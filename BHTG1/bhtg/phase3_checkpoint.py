"""Small, durable Phase3 epoch boundaries without replayable memory tables."""
import copy
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from bhtg.training_protocol import model_weights, load_model_weights


CHECKPOINT_SCHEMA = 'phase3_epoch_boundary_v1'
REPLAY_CONTEXT = {'stage': None, 'epoch': None, 'roll_sequence': 0}


def set_replay_phase(stage, epoch=None):
    """Progress instrumentation reads scalars instead of materializing frames."""
    REPLAY_CONTEXT.update(stage=stage, epoch=epoch)


def begin_memory_replay(cutoff, total_events, total_blocks=None):
    REPLAY_CONTEXT.update(roll_sequence=REPLAY_CONTEXT['roll_sequence'] + 1,
                          cutoff=float(cutoff), total_events=int(total_events),
                          total_blocks=None if total_blocks is None else int(total_blocks))


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    return copy.deepcopy(value)


def _atomic_write(path, writer, binary):
    """An interrupted serialization cannot replace the last complete artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + '.', suffix='.tmp', dir=str(target.parent))
    try:
        with os.fdopen(fd, 'wb' if binary else 'w', **({} if binary else {'encoding': 'utf-8'})) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(target))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(payload, path):
    _atomic_write(path, lambda handle: torch.save(payload, handle), True)


def write_validation_history(records, path):
    """Rewrite this small JSONL atomically; floats keep Python's full precision."""
    def write(handle):
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    _atomic_write(path, write, False)


def capture_rng():
    result = {'python': random.getstate(), 'numpy': np.random.get_state(),
              'torch': torch.get_rng_state().clone()}
    if torch.cuda.is_initialized():
        result['cuda'] = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
    return result


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise RuntimeError('Phase3 checkpoint CUDA RNG device mismatch')
        torch.cuda.set_rng_state_all(state['cuda'])


def epoch_checkpoint(epoch, contract, graph, fusion, optimizer, opt_mem,
                     best, validation_history):
    return {'checkpoint_schema': CHECKPOINT_SCHEMA, 'completed_epoch': epoch,
            'contract': copy.deepcopy(contract),
            'graph_encoder': model_weights(graph), 'fusion': model_weights(fusion),
            'optimizer': cpu_copy(optimizer.state_dict()),
            'opt_mem': None if opt_mem is None else cpu_copy(opt_mem.state_dict()),
            'best': best, 'validation_history': copy.deepcopy(validation_history),
            'rng': capture_rng(), 'runtime_memory_state_saved': False}


def restore_epoch(checkpoint, contract, graph, fusion, optimizer, opt_mem):
    if checkpoint.get('checkpoint_schema') != CHECKPOINT_SCHEMA:
        raise RuntimeError('Not a resumable Phase3 epoch checkpoint')
    if checkpoint.get('contract') != contract:
        raise RuntimeError('Phase3 checkpoint input or training contract mismatch')
    if (checkpoint.get('opt_mem') is None) != (opt_mem is None):
        raise RuntimeError('Phase3 checkpoint Joint optimizer mismatch')
    completed = checkpoint.get('completed_epoch')
    if not isinstance(completed, int) or completed < 1:
        raise RuntimeError('Invalid Phase3 completed epoch')
    if not isinstance(checkpoint.get('best'), dict):
        raise RuntimeError('Phase3 checkpoint has no selected model')
    if [record['epoch'] for record in checkpoint['validation_history']] != list(range(1, completed + 1)):
        raise RuntimeError('Phase3 checkpoint validation history is incomplete')
    load_model_weights(graph, checkpoint['graph_encoder'])
    load_model_weights(fusion, checkpoint['fusion'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if opt_mem is not None:
        opt_mem.load_state_dict(checkpoint['opt_mem'])
    restore_rng(checkpoint['rng'])
    return completed
