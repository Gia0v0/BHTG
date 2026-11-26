"""
BHTG 时间序边流提取 (TGN Memory 机制的数据基础)。

从 bhtg_graph.bin 提取全部边，按 timestamp 升序排列，序列化为
    (src_ntype_idx, dst_ntype_idx, src_lid, dst_lid, etype_idx, ts, value)
供 HeteroTemporalMemory 做时间序 GRU 更新。新缓存必须显式写入新版本目录；旧缓存只读。

另提供:
    - compute_phase_cutoffs()  — 与 run_phase3.py 一致的 train/val cutoff (防泄漏)
    - build_neg_index()        — 每节点类型的事件 (ts, node_id) 排序索引 (时间相关负采样)
    - sample_negatives()       — 给定时间 t 采样同型活跃节点作负样本
"""
import os
import hashlib
import json
import tempfile
from collections import defaultdict
import sys
import torch
import dgl
import numpy as np
import pandas as pd

# 独立运行时把项目根加入 sys.path (run_phase*.py 已做，这里兜底)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bhtg.config import BHTG_DATA_DIR, GRAPH_BIN, NODE_TYPES, EDGE_TYPES, LABELED_EDGES_CSV
from bhtg.temporal_time import conservative_timestamps, TIME_PROTOCOL

EDGE_STREAM_PATH = os.path.join(BHTG_DATA_DIR, 'edge_stream.pt')
# Historical count is documentation only, never the cache validation criterion.
TOTAL_EDGES_EXPECTED = 1122152
STREAM_SCHEMA = 'bhtg_edge_stream_bound_v3'
_FILE_HASH_CACHE = {}


def _canonical_edge_tensor(field, tensor):
    # DGL 1.1.1 serializes bool fields as uint8 and does not restore their dtype.
    # Accept only that exact transport encoding; the stream contract stays bool.
    if field == 'amount_known' and tensor.dtype == torch.uint8:
        if bool(((tensor != 0) & (tensor != 1)).any()):
            raise ValueError('invalid serialized boolean amount_known')
        return tensor.bool()
    return tensor


def _file_sha256(path):
    """Hash unchanged files once per process; stat changes invalidate the memo."""
    path = os.path.abspath(os.fspath(path))
    stat = os.stat(path)
    key = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _FILE_HASH_CACHE:
        digest = hashlib.sha256()
        with open(path, 'rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        _FILE_HASH_CACHE[key] = digest.hexdigest()
    return _FILE_HASH_CACHE[key]


def graph_fingerprint(g):
    """Bind topology, typed local IDs, node mappings and all numeric edge fields."""
    digest = hashlib.sha256()
    def update_tensor(tensor):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    for nt in sorted(g.ntypes):
        digest.update(json.dumps([nt, g.num_nodes(nt)]).encode())
        for field in sorted(g.nodes[nt].data):
            digest.update(field.encode())
            update_tensor(g.nodes[nt].data[field])
    for canonical in sorted(g.canonical_etypes):
        digest.update(json.dumps(canonical).encode())
        src, dst = g.edges(etype=canonical)
        update_tensor(src)
        update_tensor(dst)
        for field in sorted(g.edges[canonical].data):
            digest.update(field.encode())
            update_tensor(_canonical_edge_tensor(field, g.edges[canonical].data[field]))
    return digest.hexdigest()


def _extract_stream(g, allow_unsupported=False, relation_names=None):
    from bhtg.model_schema_v3 import relation_schema
    relations = list(EDGE_TYPES if relation_names is None else relation_schema(relation_names))
    extras = sorted({et[1] for et in g.canonical_etypes} - set(relations))
    if extras and not allow_unsupported:
        raise ValueError('unsupported Memory relation schema: %s' % extras)
    relations += extras
    fields = ('src_ntype', 'dst_ntype', 'src_lid', 'dst_lid', 'etype_idx', 'ts', 'value')
    chunks = {key: [] for key in fields}
    # Every event attribute is retained, with a presence mask where relations differ.
    attributes = sorted({key for et in g.canonical_etypes for key in g.edges[et].data
                         if key not in ('timestamp', 'value')})
    numeric_amounts = bool({'amount_z', 'amount_known'}.intersection(attributes))
    attribute_chunks, present_chunks = defaultdict(list), defaultdict(list)
    for canonical in g.canonical_etypes:
        source_type, relation, target_type = canonical
        count = g.num_edges(canonical)
        if not count:
            continue
        if source_type not in NODE_TYPES or target_type not in NODE_TYPES:
            raise ValueError('unsupported Memory node type schema')
        source, target = g.edges(etype=canonical)
        data = g.edges[canonical].data
        if 'timestamp' not in data or 'value' not in data:
            raise ValueError('edge stream requires explicit timestamp and value fields')
        if numeric_amounts and {'amount_z', 'amount_known', 'source_row_id_code'} - set(data):
            raise ValueError('continuous amount graph requires amount fields/identity on every relation')
        chunks['src_ntype'].append(torch.full((count,), NODE_TYPES.index(source_type), dtype=torch.long))
        chunks['dst_ntype'].append(torch.full((count,), NODE_TYPES.index(target_type), dtype=torch.long))
        chunks['src_lid'].append(source.cpu().long())
        chunks['dst_lid'].append(target.cpu().long())
        chunks['etype_idx'].append(torch.full((count,), relations.index(relation), dtype=torch.long))
        chunks['ts'].append(conservative_timestamps(data['timestamp'].cpu()))
        chunks['value'].append(data['value'].cpu().float())
        for attribute in attributes:
            example = next(g.edges[et].data[attribute] for et in g.canonical_etypes
                           if attribute in g.edges[et].data)
            exists = attribute in data
            attribute_chunks[attribute].append(_canonical_edge_tensor(attribute, data[attribute]).cpu() if exists else
                                               torch.zeros((count,) + tuple(example.shape[1:]), dtype=example.dtype))
            present_chunks[attribute].append(torch.full((count,), exists, dtype=torch.bool))
    stream = {key: (torch.cat(values) if values else
                   torch.empty(0, dtype=torch.float64 if key == 'ts' else
                               torch.float32 if key == 'value' else torch.long))
              for key, values in chunks.items()}
    order = torch.sort(stream['ts'], stable=True).indices
    if relation_names is not None and len(relations) == 8:
        native_fields = ('native_chain_id', 'native_block_number', 'native_transaction_index', 'native_log_index')
        if all(key in attribute_chunks for key in native_fields):
            native = {key: torch.cat(attribute_chunks[key]) for key in native_fields}
            if any(not bool(torch.cat(present_chunks[key]).all()) for key in native_fields):
                raise ValueError('v3 stream requires native order for every edge')
            keys = [np.arange(len(order))]
            keys += [native[key].numpy() for key in reversed(native_fields)]
            keys += [stream['ts'].numpy()]
            order = torch.from_numpy(np.lexsort(tuple(keys)).copy()).long()
            stream_order_policy = 'timestamp_native_chain_block_transaction_log_v3'
        else:
            # In-memory diagnostic graphs can omit native fields. Published
            # v3 preprocessing builds provide them and readiness rechecks them.
            stream_order_policy = 'timestamp_diagnostic_ties_v3'
    else:
        stream_order_policy = 'timestamp_stable_legacy'
    for key in fields:
        stream[key] = stream[key][order]
    stream['edge_attributes'] = {key: torch.cat(values)[order]
                                 for key, values in attribute_chunks.items()}
    stream['edge_attribute_present'] = {key: torch.cat(values)[order]
                                        for key, values in present_chunks.items()}
    stream['num_nodes'] = {NODE_TYPES.index(nt): g.num_nodes(nt) for nt in g.ntypes}
    stream['_meta'] = {'schema_version': STREAM_SCHEMA, 'total_edges': len(stream['ts']),
                       'timestamp_semantics': TIME_PROTOCOL, 'node_types': list(NODE_TYPES),
                       'relations': relations, 'memory_schema_supported': not extras,
                       'graph_fingerprint': graph_fingerprint(g),
                       'event_order_policy': stream_order_policy}
    if numeric_amounts:
        from bhtg.amounts import amount_metadata
        stream['_meta']['amount_protocol'] = amount_metadata()
    _validate_stream(stream)
    return stream


def _validate_stream(stream):
    fields = ('src_ntype', 'dst_ntype', 'src_lid', 'dst_lid', 'etype_idx', 'ts', 'value')
    meta = stream.get('_meta', {})
    count = meta.get('total_edges')
    if not isinstance(count, int) or count < 0:
        raise ValueError('edge stream has invalid edge-count metadata')
    for field in fields:
        if field not in stream or stream[field].ndim != 1 or len(stream[field]) != count:
            raise ValueError('edge stream field/count mismatch: %s' % field)
    if not torch.isfinite(stream['ts']).all() or not torch.isfinite(stream['value']).all():
        raise ValueError('edge stream timestamps/values must be finite')
    if count > 1 and bool((stream['ts'][1:] < stream['ts'][:-1]).any()):
        raise ValueError('edge stream must be sorted by time')
    for field in ('src_ntype', 'dst_ntype', 'src_lid', 'dst_lid', 'etype_idx'):
        if stream[field].is_floating_point() or bool((stream[field] < 0).any()):
            raise ValueError('invalid edge stream integer index: %s' % field)
    relations = meta.get('relations', EDGE_TYPES)
    if bool((stream['etype_idx'] >= len(relations)).any()):
        raise ValueError('stream relation index outside schema')
    nodes = stream.get('num_nodes', {})
    for type_field, local_field in (('src_ntype', 'src_lid'), ('dst_ntype', 'dst_lid')):
        if bool((stream[type_field] >= len(NODE_TYPES)).any()):
            raise ValueError('stream node type index outside schema')
        for type_index in torch.unique(stream[type_field]).tolist():
            if type_index not in nodes or bool((stream[local_field][stream[type_field] == type_index]
                                               >= nodes[type_index]).any()):
                raise ValueError('edge stream local ID outside graph mapping')
    for group in ('edge_attributes', 'edge_attribute_present'):
        for field, values in stream.get(group, {}).items():
            if len(values) != count:
                raise ValueError('event attribute count mismatch: %s' % field)
            if group == 'edge_attribute_present' and (values.ndim != 1 or values.dtype != torch.bool):
                raise ValueError('event attribute presence mask must be boolean: %s' % field)
    attributes = stream.get('edge_attributes', {})
    numeric = bool({'amount_z', 'amount_known'}.intersection(attributes))
    if numeric or meta.get('amount_protocol') is not None:
        from bhtg.amounts import amount_metadata
        if meta.get('amount_protocol') != amount_metadata():
            raise ValueError('continuous amount stream protocol mismatch')
        for field in ('amount_z', 'amount_known', 'source_row_id_code'):
            values = attributes.get(field)
            presence = stream.get('edge_attribute_present', {}).get(field)
            if values is None or values.ndim != 1 or len(values) != count:
                raise ValueError('continuous amount stream missing field: ' + field)
            if presence is None or presence.dtype != torch.bool or presence.shape != values.shape or not bool(presence.all()):
                raise ValueError('continuous amount stream incomplete presence: ' + field)
        z, known = attributes['amount_z'], attributes['amount_known']
        if not z.is_floating_point() or not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(z.float()).all()):
            raise ValueError('continuous amount stream amount_z must be finite float32')
        if known.dtype != torch.bool:
            raise ValueError('continuous amount stream amount_known must be boolean')
        if bool((z[~known] != 0).any()):
            raise ValueError('unknown continuous amount must use canonical zero')
        identity = attributes['source_row_id_code']
        if identity.is_floating_point() or identity.dtype == torch.bool or bool((identity < 0).any()):
            raise ValueError('continuous amount stream source identity must be integer')


def _verify_legacy_archive(cache_path, graph_path):
    # The immutable restore manifest binds the original graph, cache and address
    # mappings. This is explicitly a legacy diagnostic input, not new v3 evidence.
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(GRAPH_BIN))))
    manifest_path = os.path.join(root, 'restore_manifest.json')
    if not os.path.isfile(manifest_path):
        raise ValueError('unbound legacy edge stream: a frozen restore manifest is required')
    with open(manifest_path, encoding='utf-8') as source:
        manifest = json.load(source)
    expected = {os.path.normcase(os.path.abspath(os.path.join(root, item['path']))): item['sha256']
                for item in manifest.get('files', [])}
    data_dir = os.path.dirname(os.path.abspath(graph_path))
    paths = [cache_path, graph_path, os.path.join(data_dir, 'addr_to_id.csv')]
    paths += [os.path.join(data_dir, nt + '_gids.pt') for nt in NODE_TYPES]
    for path in paths:
        wanted = expected.get(os.path.normcase(os.path.abspath(path)))
        if not wanted or _file_sha256(path) != wanted:
            raise ValueError('legacy graph/stream/mapping provenance mismatch: %s' % path)


def build_edge_stream(g=None, force=False, cache_path=None, graph_path=None,
                      mapping_paths=None, allow_unsupported=False, relation_names=None):
    """Explicitly build a cache in a new directory, bound to the supplied graph.

    Default calls only read the frozen diagnostic stream. Supplying a new graph
    never reuses an unrelated old cache, and production input files stay immutable.
    """
    if cache_path is None:
        if force or g is not None:
            raise ValueError('explicit new cache_path required; frozen cache cannot be rebuilt')
        return load_edge_stream()
    cache_path = os.path.realpath(os.fspath(cache_path))
    if os.path.normcase(cache_path) == os.path.normcase(os.path.realpath(EDGE_STREAM_PATH)):
        raise ValueError('cannot overwrite or rebuild the frozen production edge stream')
    if g is None:
        if graph_path is None:
            raise ValueError('a graph or explicit graph_path is required')
        g = dgl.load_graphs(os.fspath(graph_path))[0][0]
    if os.path.exists(cache_path) and not force:
        return load_edge_stream(cache_path=cache_path, g=g, graph_path=graph_path,
                                allow_unsupported=allow_unsupported, relation_names=relation_names)
    stream = _extract_stream(g, allow_unsupported=allow_unsupported, relation_names=relation_names)
    if graph_path is not None:
        disk_graph = dgl.load_graphs(os.fspath(graph_path))[0][0]
        if graph_fingerprint(disk_graph) != stream['_meta']['graph_fingerprint']:
            raise ValueError('supplied graph differs from graph_path; cannot bind unrelated cache')
        stream['_meta']['graph_sha256'] = _file_sha256(graph_path)
        stream['_meta']['graph_filename'] = os.path.basename(graph_path)
    if mapping_paths:
        if any(os.path.normcase(os.path.realpath(os.path.dirname(os.fspath(path)))) !=
               os.path.normcase(os.path.realpath(os.path.dirname(cache_path))) for path in mapping_paths):
            raise ValueError('bound mapping artifacts must be beside their cache in the same bundle')
        stream['_meta']['mapping_sha256'] = {os.path.basename(path): _file_sha256(path)
                                              for path in mapping_paths}
    _validate_stream(stream)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.edge_stream_', suffix='.pt',
                                            dir=os.path.dirname(cache_path))
    os.close(descriptor)
    try:
        torch.save(stream, temporary)
        os.replace(temporary, cache_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return stream


def load_edge_stream(force_rebuild=False, cache_path=None, g=None, graph_path=None,
                     bundle_manifest=None, allow_unsupported=False, relation_names=None):
    """Read and validate a bound stream, or the hash-verified frozen legacy input."""
    if force_rebuild:
        raise ValueError('implicit stream rebuild is disabled; explicitly build a new versioned cache')
    if cache_path is None:
        from bhtg.action_csv import require_graph_binding
        require_graph_binding(graph_path)
    if bundle_manifest is None and cache_path is None:
        from bhtg import config
        bundle_manifest = getattr(config, "GRAPH_MANIFEST_PATH", None)
    path = os.path.abspath(os.fspath(cache_path or EDGE_STREAM_PATH))
    if not os.path.isfile(path):
        raise FileNotFoundError('edge stream is missing; explicitly build a new cache before training: %s' % path)
    stream = torch.load(path, map_location='cpu')
    _validate_stream(stream)
    meta = stream['_meta']
    if meta.get('schema_version') != STREAM_SCHEMA:
        if g is not None:
            raise ValueError('legacy stream cannot be reused for a supplied new graph')
        _verify_legacy_archive(path, graph_path or GRAPH_BIN)
        meta['provenance_status'] = 'frozen_legacy_diagnostic_only'
    else:
        from bhtg.model_schema_v3 import relation_schema
        expected_relations = list(EDGE_TYPES if relation_names is None else relation_schema(relation_names))
        if (not meta.get('memory_schema_supported') or meta.get('relations') != expected_relations) and not allow_unsupported:
            raise ValueError('unsupported Memory relation schema; migrate model before training this bundle')
        if meta.get('node_types') != list(NODE_TYPES):
            raise ValueError('stream node schema differs from current model')
        if g is not None and graph_fingerprint(g) != meta.get('graph_fingerprint'):
            raise ValueError('edge cache graph fingerprint mismatch; build a new cache')
        if graph_path is None and bundle_manifest is not None:
            graph_path = os.path.join(os.path.dirname(os.fspath(bundle_manifest)), 'bhtg_graph.bin')
        if graph_path is not None:
            if _file_sha256(graph_path) != meta.get('graph_sha256'):
                raise ValueError('edge cache graph file hash mismatch')
        # A supplied in-memory graph binds local/global IDs through its fingerprint;
        # independently verify saved address maps even when graph_path is omitted.
        map_dir = os.path.dirname(os.fspath(graph_path)) if graph_path is not None else os.path.dirname(path)
        for filename, expected in meta.get('mapping_sha256', {}).items():
            if os.path.basename(filename) != filename:
                raise ValueError('mapping artifact name cannot traverse directories')
            if _file_sha256(os.path.join(map_dir, filename)) != expected:
                raise ValueError('edge cache address mapping hash mismatch')
        if g is None and graph_path is None:
            raise ValueError('bound cache requires its graph or bundle manifest for verification')
        if bundle_manifest is not None:
            with open(bundle_manifest, encoding='utf-8') as source:
                manifest = json.load(source)
            if manifest.get('schema_version') != 'bhtg_graph_bundle_v3':
                raise ValueError('unsupported graph bundle manifest schema')
            required = {'bhtg_graph.bin', 'edge_stream.pt', 'addr_to_id.csv', 'token_list.csv',
                        'contract_gids.pt', 'eoa_gids.pt', 'pool_gids.pt',
                        'train_cutoff.json', 'provenance_vocabulary.json'}
            if required - set(manifest.get('files', {})):
                raise ValueError('graph bundle manifest is missing required bound artifacts')
            if manifest.get('num_nodes') != {nt: stream['num_nodes'].get(NODE_TYPES.index(nt)) for nt in NODE_TYPES}:
                raise ValueError('graph bundle/stream node mapping count mismatch')
            for filename, record in manifest.get('files', {}).items():
                if os.path.basename(filename) != filename:
                    raise ValueError('bundle artifact name cannot traverse directories')
                if _file_sha256(os.path.join(os.path.dirname(os.fspath(bundle_manifest)), filename)) != record['sha256']:
                    raise ValueError('graph bundle artifact hash mismatch: %s' % filename)
            if manifest.get('total_edges') != meta.get('total_edges'):
                raise ValueError('graph bundle/stream count mismatch')
    if meta.get('timestamp_semantics') != TIME_PROTOCOL:
        stream['ts_raw'] = stream['ts'].to(dtype=torch.float64)
        stream['ts'] = conservative_timestamps(stream['ts'])
        meta['timestamp_semantics'] = TIME_PROTOCOL
    return stream


def compute_phase_cutoffs():
    """与 run_phase3.py 一致的防泄漏 cutoff (labeled 边 timestamp 的 0.70/0.85 分位)。"""
    from bhtg.action_csv import model_edges
    t = model_edges(usecols=['timestamp'], legacy_path=LABELED_EDGES_CSV)['timestamp'].to_numpy(dtype=np.float64)
    return float(np.quantile(t, 0.70)), float(np.quantile(t, 0.85))


def split_masks(stream, train_cutoff, val_cutoff):
    """返回 train/val 布尔掩码 (val ⊇ train，单趟 roll 用)。"""
    ts = stream['ts']
    return {'train': ts < train_cutoff, 'val': ts < val_cutoff,
            'val_only': (ts >= train_cutoff) & (ts < val_cutoff)}


def iter_time_batches(timestamps, batch_size):
    """Yield ascending block bounds, extending each block through timestamp ties."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if len(timestamps) > 1 and bool((timestamps[1:] < timestamps[:-1]).any()):
        raise ValueError('timestamp batches require an ascending stream')
    start, n = 0, len(timestamps)
    while start < n:
        end = min(start + batch_size, n)
        if end < n:
            end = int(torch.searchsorted(timestamps, timestamps[end - 1], right=True))
        yield start, end
        start = end


def temporal_batches(timestamps, indices, batch_size):
    """Yield index blocks without splitting a concurrent timestamp group."""
    selected_times = timestamps[indices]
    for start, end in iter_time_batches(selected_times, batch_size):
        yield indices[start:end]


def build_neg_index(stream):
    """按 ntype 构建 (ts, node_id) 事件排序索引，供时间相关负采样。返回 {ntype_idx: (ts, ids)}。"""
    neg_index = {}
    for ti in range(len(NODE_TYPES)):
        src_mask = stream['src_ntype'] == ti
        dst_mask = stream['dst_ntype'] == ti
        ids = torch.cat([stream['src_lid'][src_mask], stream['dst_lid'][dst_mask]])
        t = torch.cat([stream['ts'][src_mask], stream['ts'][dst_mask]])
        order = torch.argsort(t)
        neg_index[ti] = (t[order], ids[order])
    return neg_index


def sample_negatives(neg_index, dst_ntype, dst_lid, ts, t_half=7 * 86400.0,
                     n_cand=32, src_ntype=None, src_lid=None,
                     return_mask=False, generator=None):
    """
    Sample same-type nodes active in the strictly past [t-t_half, t) window.
    Positive destination and same-type source are excluded. If no candidate
    exists, return -1 and a false mask; callers must omit that negative loss.
    Args:
        neg_index: build_neg_index() 输出 {ntype_idx: (ts, ids)}
        dst_ntype: (B,) 正边 dst 节点类型 idx
        dst_lid:   (B,) 正边 dst full-graph local id
        ts:        (B,) 正边时间戳
    Returns:
        neg_lid: (B,) 负样本 dst local id (≠ 正边 dst)
    """
    neg = torch.full_like(dst_lid, -1)
    valid = torch.zeros_like(dst_lid, dtype=torch.bool)
    for ti in range(len(NODE_TYPES)):
        mask = dst_ntype == ti
        B = int(mask.sum())
        if B == 0:
            continue
        t_arr, id_arr = neg_index[ti]
        q = ts[mask]
        lo = torch.searchsorted(t_arr, q - t_half)
        hi = torch.searchsorted(t_arr, q, right=False)
        if not len(id_arr):
            continue
        width = hi - lo
        offsets = (torch.rand(B, max(1, n_cand), device=q.device, generator=generator)
                   * width.clamp(min=1).unsqueeze(1)).long()
        locations = (lo.unsqueeze(1) + offsets).clamp(max=len(id_arr) - 1)
        candidates = id_arr[locations]
        allowed = (width > 0).unsqueeze(1) & (candidates != dst_lid[mask].unsqueeze(1))
        if src_ntype is not None and src_lid is not None:
            allowed &= ~((src_ntype[mask] == ti).unsqueeze(1)
                         & (candidates == src_lid[mask].unsqueeze(1)))
        has_candidate = allowed.any(dim=1)
        first = allowed.long().argmax(dim=1)
        choices = candidates[torch.arange(B, device=q.device), first]
        positions = mask.nonzero(as_tuple=True)[0]
        neg[positions[has_candidate]] = choices[has_candidate]
        valid[positions[has_candidate]] = True
        # Exhaustive fallback stays within this row's strict historical window.
        # It is needed only when randomized candidates repeatedly hit endpoints.
        for row in ((width > 0) & ~has_candidate).nonzero(as_tuple=True)[0].tolist():
            position = int(positions[row])
            candidates = id_arr[int(lo[row]):int(hi[row])]
            allowed = candidates != dst_lid[position]
            if src_ntype is not None and src_lid is not None and int(src_ntype[position]) == ti:
                allowed &= candidates != src_lid[position]
            candidates = candidates[allowed]
            if len(candidates):
                pick = int(torch.randint(len(candidates), (1,), generator=generator,
                                         device=candidates.device))
                neg[position], valid[position] = candidates[pick], True
    return (neg, valid) if return_mask else neg


if __name__ == '__main__':
    stream = build_edge_stream()
    print('[C1] total edges = %d (expect %d)' % (stream['_meta']['total_edges'], TOTAL_EDGES_EXPECTED))
    tr_c, va_c = compute_phase_cutoffs()
    print('[C1] train_cutoff=%.0f val_cutoff=%.0f' % (tr_c, va_c))
    m = split_masks(stream, tr_c, va_c)
    print('[C1] ts<train: %d (expect 984392)  ts<val: %d (expect 1051437)' % (int(m['train'].sum()), int(m['val'].sum())))
