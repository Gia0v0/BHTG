"""Rematerialize typed Memory blocks on the installed PyTorch 1.12 runtime.

Only memory vectors are differentiable outputs. Timestamp/seen metadata are
returned from the original forward as detached constants. A scalar dummy input
keeps reentrant checkpointing active for the first block, whose prefix memory
is intentionally detached; the dummy is never a model parameter or weight.
"""
import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_memory_block(roll_fn, tables):
    """Call an event-batch-bound pure ``roll_fn(state)`` with rematerialization.

    Bind the event batch by value, for example with ``functools.partial``.
    Recomputing must not commit runtime state, update weights, or depend on
    mutable loop variables. Inference/no-grad callers run directly.
    """
    if not torch.is_grad_enabled() or not tables:
        return roll_fn(tables)
    node_types = tuple(tables)
    inputs = []
    for node_type in node_types:
        values = tables[node_type]
        if (not isinstance(values, (tuple, list)) or len(values) != 3
                or not all(torch.is_tensor(value) for value in values)):
            raise ValueError('checkpoint Memory state requires three typed tensors')
        memory, timestamps, seen = values
        if (memory.ndim != 2 or timestamps.shape != (len(memory),)
                or seen.shape != (len(memory),) or timestamps.dtype != torch.float64
                or seen.dtype != torch.bool):
            raise ValueError('checkpoint Memory state shape or metadata dtype mismatch')
        if timestamps.requires_grad or seen.requires_grad:
            raise ValueError('Memory timestamp/seen metadata must remain nondifferentiable')
        inputs.extend((memory, timestamps, seen))
    metadata = {}

    def run(*flat):
        state = {node_type: (flat[3 * index], flat[3 * index + 1].detach(),
                             flat[3 * index + 2])
                 for index, node_type in enumerate(node_types)}
        output = roll_fn(state)
        if set(output) != set(node_types):
            raise ValueError('checkpointed Memory block changed node types')
        if not torch.is_grad_enabled():
            # Reentrant backward recomputes under enable_grad, so it cannot
            # replace the metadata belonging to the original forward.
            metadata.update({node_type: (output[node_type][1].detach(),
                                         output[node_type][2].detach())
                             for node_type in node_types})
        return tuple(output[node_type][0] for node_type in node_types)

    dummy = torch.zeros((), device=inputs[0].device, requires_grad=True)
    vectors = checkpoint(run, *inputs, dummy, use_reentrant=True, preserve_rng_state=True)
    if torch.is_tensor(vectors):
        vectors = (vectors,)
    return {node_type: (vector,) + metadata[node_type]
            for node_type, vector in zip(node_types, vectors)}
