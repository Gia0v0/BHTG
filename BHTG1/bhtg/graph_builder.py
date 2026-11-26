"""Side-effect-free heterograph construction and explicit v3 bundle conversion.

Frozen production inputs are never rebuilt in place. New bundles bind the graph,
all global ID maps, event provenance and Memory stream with content hashes.
Supply mint/burn and LP relations are kept distinct; unverified legacy chain
creation and full-history pool enrichment are rejected.
"""
import torch
import dgl
import numpy as np
import os
import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict

from bhtg.config import (
    LABELED_EDGES_CSV, UNLABELED_EDGES_CSV, ADDR_TO_ID_CSV, TOKEN_LIST_CSV, GRAPH_BIN,
    NODE_TYPES, ZERO_ADDR, DEVICE, BHTG_DATA_DIR,
)
import pandas as pd


# ============================================================
# 1. 节点类型判定 (简化版: 不需链上查询)
# ============================================================

class NodeTypeClassifier:
    """
    基于地址集合信息判定节点类型

    简化策略 (无需 bytecode):
        - 在 token_list 中的地址 → 'contract'
        - 其他 → 'eoa'
        - Pool 暂并入 contract，后续有链上数据再区分
    """

    def __init__(self, token_addrs, addr_to_id):
        self.token_addrs = set(token_addrs)
        self.addr_to_id = addr_to_id
        self.id_to_addr = {v: k for k, v in addr_to_id.items()}

    def classify(self, node_id):
        """返回 'eoa' | 'contract' | 'pool'"""
        addr = self.id_to_addr.get(node_id, None)
        if addr is None:
            return 'eoa'
        if addr in self.token_addrs:
            return 'contract'
        return 'eoa'

    def classify_all(self, num_nodes):
        """一次性分类所有节点"""
        node_types = {}
        for nid in range(num_nodes):
            node_types[nid] = self.classify(nid)
        return node_types


# ============================================================
# 2. 异构图构建
# ============================================================

GRAPH_BUNDLE_SCHEMA = 'bhtg_graph_bundle_v3'
def _mapping_fingerprint(mapping):
    return hashlib.sha256(json.dumps(sorted(mapping.items()), separators=(',', ':'),
                                      ensure_ascii=False).encode('utf-8')).hexdigest()


GRAPH_RELATIONS = ('transfer', 'mint', 'burn', 'create',
                   'lp_mint', 'lp_burn', 'lp_swap', 'lp_sync')


def build_hetero_graph(edges_dict, addr_to_id, token_list,
                      chain_node_types=None, pool_addresses=None, train_cutoff=None):
    """Build an in-memory graph without writing any production mappings.

    Numeric event fields keep their dtype, including int64 identity codes.
    String provenance must be encoded with an explicit vocabulary by the bundle
    writer. Invalid IDs and unsupported node types fail rather than dropping rows.
    """
    from bhtg.temporal_time import conservative_timestamps
    num_nodes = len(addr_to_id)
    if sorted(addr_to_id.values()) != list(range(num_nodes)):
        raise ValueError('address IDs must be unique and contiguous global IDs')
    unsupported = set(edges_dict) - set(GRAPH_RELATIONS)
    if unsupported:
        raise ValueError('unsupported graph relation schema: %s' % sorted(unsupported))
    classifier = NodeTypeClassifier(token_list, addr_to_id)
    node_type_map = classifier.classify_all(num_nodes)
    if chain_node_types is not None:
        for gid, address in classifier.id_to_addr.items():
            if address in chain_node_types:
                node_type_map[gid] = chain_node_types[address]
    if pool_addresses:
        for gid, address in classifier.id_to_addr.items():
            if address in pool_addresses:
                node_type_map[gid] = 'pool'
    if set(node_type_map.values()) - set(NODE_TYPES):
        raise ValueError('unknown node types must be verified before graph construction')
    global_to_local, counters = {}, defaultdict(int)
    for gid in range(num_nodes):
        nt = node_type_map[gid]
        global_to_local[gid] = (nt, counters[nt])
        counters[nt] += 1

    # Take the union from all relations; a transfer-only schema cannot hide LP attrs.
    extra_keys = sorted({key for fields in edges_dict.values() for key in fields
                         if key not in ('src', 'dst', 'timestamp', 'value', 'label')})
    amount_fields = {'amount_z', 'amount_known'}
    numeric_amounts = bool(amount_fields.intersection(extra_keys))
    dgl_edges, grouped_fields = {}, {}
    for relation in GRAPH_RELATIONS:
        if relation not in edges_dict:
            continue
        fields = edges_dict[relation]
        required = ('src', 'dst', 'timestamp', 'value', 'label')
        if any(key not in fields for key in required):
            raise ValueError('missing required edge field in %s' % relation)
        if numeric_amounts and (amount_fields | {'source_row_id_code'}) - set(fields):
            raise ValueError('continuous amount fields/identity required on every relation: ' + relation)
        n = len(fields['src'])
        tensors = {}
        for key in required + tuple(key for key in extra_keys if key in fields):
            try:
                tensor = torch.as_tensor(fields[key]).cpu()
            except (TypeError, ValueError):
                raise ValueError('edge field %s needs numeric encoding and provenance vocabulary' % key)
            if tensor.ndim != 1 or len(tensor) != n:
                raise ValueError('edge field lengths/shapes disagree in %s: %s' % (relation, key))
            tensors[key] = tensor
        for key in ('src', 'dst'):
            ids = tensors[key]
            if ids.is_floating_point() or bool(((ids < 0) | (ids >= num_nodes)).any()):
                raise ValueError('invalid global endpoint IDs in %s' % relation)
            tensors[key] = ids.long()
        tensors['timestamp'] = conservative_timestamps(tensors['timestamp'])
        if not bool(torch.isfinite(tensors['value']).all()):
            raise ValueError('edge values must be finite')
        if numeric_amounts:
            z, known = tensors['amount_z'], tensors['amount_known']
            if not z.is_floating_point() or not bool(torch.isfinite(z).all()):
                raise ValueError('amount_z must be finite floating point')
            if known.dtype != torch.bool:
                raise ValueError('amount_known must be boolean')
            if bool((z[~known] != 0).any()):
                raise ValueError('unknown amount_z must use canonical zero')
            tensors['amount_z'] = z.float()
            if not bool(torch.isfinite(tensors['amount_z']).all()):
                raise ValueError('amount_z overflows model float32 precision')
            identity = tensors['source_row_id_code']
            if identity.is_floating_point() or identity.dtype == torch.bool or bool((identity < 0).any()):
                raise ValueError('amount source identity code must be nonnegative integer')
        groups = defaultdict(list)
        for index, (src, dst) in enumerate(zip(tensors['src'].tolist(), tensors['dst'].tolist())):
            groups[(node_type_map[src], relation, node_type_map[dst])].append(index)
        for canonical, indices in groups.items():
            idx = torch.tensor(indices, dtype=torch.long)
            dgl_edges[canonical] = (
                torch.tensor([global_to_local[int(tensors['src'][i])][1] for i in indices]),
                torch.tensor([global_to_local[int(tensors['dst'][i])][1] for i in indices]))
            grouped_fields[canonical] = {key: value[idx] for key, value in tensors.items()
                                         if key not in ('src', 'dst')}
    if not dgl_edges:
        dgl_edges[('eoa', 'transfer', 'contract')] = (torch.empty(0, dtype=torch.long),
                                                   torch.empty(0, dtype=torch.long))
    g = dgl.heterograph(dgl_edges, num_nodes_dict={nt: counters[nt] for nt in NODE_TYPES})
    for canonical, fields in grouped_fields.items():
        for key, tensor in fields.items():
            g.edges[canonical].data[key] = tensor
    for nt in NODE_TYPES:
        gids = [gid for gid, (typ, _) in global_to_local.items() if typ == nt]
        g.nodes[nt].data['global_id'] = torch.tensor(gids, dtype=torch.long)
    features = compute_node_features(g, node_type_map, global_to_local, num_nodes, train_cutoff)
    labels = compute_node_labels(g, edges_dict, global_to_local, node_type_map)
    for nt in NODE_TYPES:
        g.nodes[nt].data['feat'] = features[nt]
        g.nodes[nt].data['label'] = labels[nt]
    g.global_to_local, g.node_type_map = global_to_local, node_type_map
    g.address_map_fingerprint = _mapping_fingerprint(addr_to_id)
    return g, features, labels


def compute_node_features(g, node_type_map, global_to_local, num_nodes, train_cutoff=None):
    """
    为三类节点计算初始特征 — 用 DGL 原生度统计，避免 NetworkX ID 错位

    Args:
        train_cutoff: float or None — 若设置，只统计 timestamp < train_cutoff 的边
                      （防止测试期边泄漏到入度特征）

    基础 8 维:
      in_degree, out_degree, total_degree,
      pool_in_degree,          (来自 pool 的入边数 — rugpull 信号)
      mint_in_degree,          (来自 mint 边的入边数)
      burn_in_degree,          (来自 burn 边的入边数)
      normalized_value_mean    (该节点相关边的平均 value)

    P0 扩展 (config.NODE_FEAT_EXTRA=True 时追加 7 维, 论文 PhishEye T1-T9/N1-N5 启发):
      in_val / out_val   (T4/T5 IVF/OVF)  入/出边 value 总和
      dtd                (T7)             total_deg / (t_last - t_first + 1) 时间密度
      cir                (T8)             邻接边中对方为 contract/pool 的比例
      ncd                (N1)             唯一邻居数（去重跨类型）
      pntd               (T9)             邻居 total_deg 最大值
      andc               (N5)             邻居 total_deg 均值

    全部只用 train_cutoff 前边计算（防泄漏）。ic/oc/dc (N2-4 归一化中心性) 未加:
    每节点类型单独 node_proj 线性投影下，度/(|V|-1) 与原始度是常数倍关系，信息冗余。
    """
    print('[graph_builder] Computing node features (DGL-native)...')

    from bhtg.config import NODE_FEAT_EXTRA

    ntypes = ['eoa', 'contract', 'pool']
    ntype_idx = {t: i for i, t in enumerate(ntypes)}
    CONTRACT_IDX, POOL_IDX = ntype_idx['contract'], ntype_idx['pool']
    ENC_SHIFT = 24          # 邻居跨类型编码: ntype_idx << 24 | local_id (eoa 最大 ~284k < 2^24)
    EXTRA_DIM = 7

    base_td = {}            # ntype -> total_deg 数组 (P0 邻居聚合要用到所有类型)
    cols_store = {}         # ntype -> 特征列 list

    # ---- Pass 1: 基础 8 维（原逻辑）----
    for ntype in ntypes:
        n = g.num_nodes(ntype)
        if n == 0:
            dim = 8 + (EXTRA_DIM if NODE_FEAT_EXTRA else 0)
            base_td[ntype] = np.zeros(0)
            cols_store[ntype] = None
            continue

        in_deg = np.zeros(n)
        out_deg = np.zeros(n)
        pool_in = np.zeros(n)
        mint_in = np.zeros(n)
        burn_in = np.zeros(n)
        val_sum = np.zeros(n)
        val_cnt = np.zeros(n)

        for etype in g.canonical_etypes:
            srct, ename, dstt = etype
            if g.num_edges(etype) == 0:
                continue
            u, v = g.edges(etype=etype)
            us = u.tolist(); vs = v.tolist()
            vals = g.edges[etype].data.get('value',
                    torch.zeros(len(us), dtype=torch.float)).tolist()
            ts = g.edges[etype].data.get('timestamp',
                    torch.zeros(len(us), dtype=torch.float)).tolist()

            # 时间过滤：若设置 train_cutoff，只用截止前的边（防泄漏）
            keep = [i for i in range(len(ts)) if train_cutoff is None or ts[i] < train_cutoff]

            # 只处理与本类型相关的边
            if srct == ntype and dstt == ntype:
                for i in keep:
                    out_deg[us[i]] += 1
                    in_deg[vs[i]] += 1
            elif srct == ntype:
                for i in keep:
                    out_deg[us[i]] += 1
            elif dstt == ntype:
                for i in keep:
                    in_deg[vs[i]] += 1

            # 边类型特定统计 (目标是 contract 的入边) — 仅在 ntype == 'contract' 时有效
            if ntype == 'contract' and dstt == 'contract':
                for i in keep:
                    if ename == 'transfer':
                        val_sum[vs[i]] += vals[i]; val_cnt[vs[i]] += 1
                    elif ename == 'mint':
                        mint_in[vs[i]] += 1
                    elif ename == 'burn':
                        burn_in[vs[i]] += 1
                    if srct == 'pool':
                        pool_in[vs[i]] += 1

        td = in_deg + out_deg
        base_td[ntype] = td
        val_mean = np.divide(val_sum, val_cnt, out=np.zeros_like(val_sum), where=val_cnt > 0)
        cols_store[ntype] = [
            in_deg, out_deg, td,
            pool_in, mint_in, burn_in, val_mean,
            np.log1p(td),  # log-scaled total degree
        ]

    # ---- Pass 2 (P0): 论文启发扩展特征 ----
    if NODE_FEAT_EXTRA:
        for ntype in ntypes:
            if cols_store[ntype] is None:
                continue
            n = g.num_nodes(ntype)
            td = base_td[ntype]
            in_val = np.zeros(n)
            out_val = np.zeros(n)
            first_ts = np.full(n, np.inf)
            last_ts = np.full(n, -np.inf)
            inc_node = []          # 本类型节点 local id (incident 边, 双向都算)
            inc_nei = []           # 邻居编码 (ntype_idx<<24 | local_id)
            inc_nei_ntype = []     # 邻居 ntype idx (cir 用)
            inc_nei_deg = []       # 邻居 total_deg (pntd/andc 用, 跨类型查 base_td)

            for etype in g.canonical_etypes:
                srct, ename, dstt = etype
                if g.num_edges(etype) == 0:
                    continue
                u, v = g.edges(etype=etype)
                us = u.tolist(); vs = v.tolist()
                vals = g.edges[etype].data.get('value',
                        torch.zeros(len(us), dtype=torch.float)).tolist()
                ts = g.edges[etype].data.get('timestamp',
                        torch.zeros(len(us), dtype=torch.float)).tolist()
                keep = [i for i in range(len(ts)) if train_cutoff is None or ts[i] < train_cutoff]
                si, di = ntype_idx[srct], ntype_idx[dstt]

                # 与 Pass 1 相同的类型门控：src 角色仅当 srct==ntype，dst 角色仅当 dstt==ntype
                if srct == ntype and dstt == ntype:
                    for i in keep:
                        # src 角色: out_val + incident(nei=dst)
                        out_val[us[i]] += vals[i]
                        if ts[i] < first_ts[us[i]]: first_ts[us[i]] = ts[i]
                        if ts[i] > last_ts[us[i]]: last_ts[us[i]] = ts[i]
                        inc_node.append(us[i])
                        inc_nei.append((di << ENC_SHIFT) | vs[i])
                        inc_nei_ntype.append(di)
                        inc_nei_deg.append(base_td[dstt][vs[i]])
                        # dst 角色: in_val + incident(nei=src)
                        in_val[vs[i]] += vals[i]
                        if ts[i] < first_ts[vs[i]]: first_ts[vs[i]] = ts[i]
                        if ts[i] > last_ts[vs[i]]: last_ts[vs[i]] = ts[i]
                        inc_node.append(vs[i])
                        inc_nei.append((si << ENC_SHIFT) | us[i])
                        inc_nei_ntype.append(si)
                        inc_nei_deg.append(base_td[srct][us[i]])
                elif srct == ntype:
                    for i in keep:
                        out_val[us[i]] += vals[i]
                        if ts[i] < first_ts[us[i]]: first_ts[us[i]] = ts[i]
                        if ts[i] > last_ts[us[i]]: last_ts[us[i]] = ts[i]
                        inc_node.append(us[i])
                        inc_nei.append((di << ENC_SHIFT) | vs[i])
                        inc_nei_ntype.append(di)
                        inc_nei_deg.append(base_td[dstt][vs[i]])
                elif dstt == ntype:
                    for i in keep:
                        in_val[vs[i]] += vals[i]
                        if ts[i] < first_ts[vs[i]]: first_ts[vs[i]] = ts[i]
                        if ts[i] > last_ts[vs[i]]: last_ts[vs[i]] = ts[i]
                        inc_node.append(vs[i])
                        inc_nei.append((si << ENC_SHIFT) | us[i])
                        inc_nei_ntype.append(si)
                        inc_nei_deg.append(base_td[srct][us[i]])

            inc_node = np.asarray(inc_node, dtype=np.int64)
            inc_nei = np.asarray(inc_nei, dtype=np.int64)
            inc_nei_ntype = np.asarray(inc_nei_ntype, dtype=np.int64)
            inc_nei_deg = np.asarray(inc_nei_deg, dtype=np.float64)

            n_inc = np.bincount(inc_node, minlength=n).astype(np.float64)

            # ncd / pntd / andc: 去重到唯一 (node, nei) 对
            if inc_node.size:
                pairs = np.stack([inc_node, inc_nei], axis=1)
                uniq_rows, first_idx = np.unique(pairs, axis=0, return_index=True)
                u_node = uniq_rows[:, 0]
                u_deg = inc_nei_deg[first_idx]
                ncd = np.bincount(u_node, minlength=n).astype(np.float64)
                pntd = np.zeros(n)
                np.maximum.at(pntd, u_node, u_deg)
                andc = np.zeros(n)
                np.add.at(andc, u_node, u_deg)
                andc = np.divide(andc, ncd, out=np.zeros(n), where=ncd > 0)
            else:
                ncd = np.zeros(n); pntd = np.zeros(n); andc = np.zeros(n)

            # cir: incident 边中对方为 contract/pool 的比例 (边加权)
            inc_mask = (inc_nei_ntype == CONTRACT_IDX) | (inc_nei_ntype == POOL_IDX)
            n_contract = np.bincount(inc_node[inc_mask], minlength=n).astype(np.float64)
            cir = np.divide(n_contract, n_inc, out=np.zeros(n), where=n_inc > 0)

            # dtd: 时间密度
            span = np.maximum(last_ts - first_ts, 0.0) + 1.0
            dtd = np.divide(td, span, out=np.zeros(n), where=np.isfinite(first_ts))

            cols_store[ntype] += [
                in_val, out_val, dtd, cir, ncd, pntd, andc,
            ]

    # ---- Pass 3: 组装 + Z-score ----
    features = {}
    for ntype in ntypes:
        if cols_store[ntype] is None:
            features[ntype] = torch.empty(0, 8 + (EXTRA_DIM if NODE_FEAT_EXTRA else 0))
            continue
        feats = np.stack(cols_store[ntype], axis=1).astype(np.float32)
        for col in range(feats.shape[1]):
            m = feats[:, col].mean(); s = feats[:, col].std()
            if s > 1e-8:
                feats[:, col] = (feats[:, col] - m) / s
        features[ntype] = torch.from_numpy(feats)
        print(f'[graph_builder]   {ntype}: {features[ntype].shape}')

    return features


# ============================================================
# 4. 节点标签
# ============================================================

def compute_node_labels(g, edges_dict, global_to_local, node_type_map):
    """为 token (contract) 节点分配 rugpull 标签"""
    node_labels = {}

    # 从原始边数据中收集每个 token 的 label
    token_label = {}
    for etype, edata in edges_dict.items():
        for i in range(len(edata['src'])):
            dst_gid = int(edata.get('token_gid', edata['dst'])[i])
            lbl = int(edata['label'][i])
            if dst_gid not in token_label or token_label[dst_gid] == -1:
                token_label[dst_gid] = lbl

    # 按类型组装
    for ntype in ['eoa', 'contract', 'pool']:
        num = g.num_nodes(ntype)
        labels = torch.full((num,), -1, dtype=torch.long)
        node_labels[ntype] = labels

    # 对有标签的 contract 节点设置标签
    labeled_contracts = []
    for gid, lbl in token_label.items():
        if lbl != -1 and node_type_map.get(gid) == 'contract':
            _, lid = global_to_local[gid]
            node_labels['contract'][lid] = lbl
            labeled_contracts.append(gid)

    rugpull = sum(1 for gid, lbl in token_label.items() if lbl == 1 and node_type_map.get(gid) == 'contract')
    benign = sum(1 for gid, lbl in token_label.items() if lbl == 0 and node_type_map.get(gid) == 'contract')
    print(f'[graph_builder]   Labeled tokens: {rugpull} rugpull + {benign} benign')
    return node_labels


# ============================================================
# 主入口
# ============================================================

def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _check_new_output(output_dir):
    output_dir = os.path.realpath(os.fspath(output_dir))
    # The active four-actions model directory is new output, not the archive.
    protected = os.path.realpath(os.path.join(os.path.dirname(__file__), 'data'))
    try:
        in_protected = os.path.normcase(os.path.commonpath([output_dir, protected])) == os.path.normcase(protected)
    except ValueError:
        in_protected = False
    if in_protected:
        raise ValueError('frozen production data directory cannot be a rebuild output')
    if os.path.exists(output_dir):
        raise FileExistsError('new graph bundle directory must not exist: %s' % output_dir)
    return output_dir


def write_graph_bundle(g, addr_to_id, token_list, output_dir, train_cutoff,
                       source_hashes=None, provenance=None, string_vocabularies=None,
                       edge_records=None, model_relations=None, node_type_records=None):
    """Atomically publish a new graph/map/stream bundle; never reuse old caches.

    LP relations remain distinct. The default diagnostic path uses the legacy
    schema; explicit model_relations enables the separate v3 eight-relation path.
    """
    from bhtg.edge_stream import build_edge_stream
    output_dir = _check_new_output(output_dir)
    if getattr(g, 'address_map_fingerprint', None) != _mapping_fingerprint(addr_to_id):
        raise ValueError('provided address map is not the one used to construct this graph')
    ids = sorted(gid for nt in NODE_TYPES for gid in g.nodes[nt].data['global_id'].tolist())
    if ids != list(range(len(addr_to_id))):
        raise ValueError('graph-local global maps must partition the address map exactly')
    if train_cutoff is None or not np.isfinite(train_cutoff):
        raise ValueError('an explicit finite training cutoff is required')
    parent = os.path.dirname(output_dir)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix='.graph_bundle_', dir=parent)
    try:
        graph_path = os.path.join(staging, 'bhtg_graph.bin')
        dgl.save_graphs(graph_path, [g])
        pd.DataFrame(sorted(addr_to_id.items(), key=lambda item: item[1]),
                     columns=['address', 'id']).to_csv(os.path.join(staging, 'addr_to_id.csv'), index=False)
        pd.DataFrame({'token_address': token_list}).to_csv(os.path.join(staging, 'token_list.csv'), index=False)
        for nt in NODE_TYPES:
            if 'global_id' not in g.nodes[nt].data:
                raise ValueError('bundle requires graph-local to global ID fields')
            torch.save(g.nodes[nt].data['global_id'].tolist(), os.path.join(staging, nt + '_gids.pt'))
        mapping_files = [os.path.join(staging, nt + '_gids.pt') for nt in NODE_TYPES]
        mapping_files.append(os.path.join(staging, 'addr_to_id.csv'))
        stream = build_edge_stream(g, cache_path=os.path.join(staging, 'edge_stream.pt'),
                                   graph_path=graph_path, mapping_paths=mapping_files,
                                   allow_unsupported=model_relations is None, relation_names=model_relations)
        with open(os.path.join(staging, 'provenance_vocabulary.json'), 'w', encoding='utf-8') as out:
            json.dump(string_vocabularies or {}, out, sort_keys=True, ensure_ascii=False, indent=2)
        if edge_records is not None:
            edge_records.to_csv(os.path.join(staging, 'edge_records.csv'), index=False)
        if node_type_records is not None:
            node_type_records.to_csv(os.path.join(staging, 'chain_node_types.csv'), index=False)
        if provenance and provenance.get('amount_audit') is not None:
            with open(os.path.join(staging, 'amount_source_audit.json'), 'w', encoding='utf-8') as out:
                json.dump(provenance['amount_audit'], out, sort_keys=True, ensure_ascii=False, indent=2)
        with open(os.path.join(staging, 'train_cutoff.json'), 'w', encoding='utf-8') as out:
            json.dump({'train_cutoff': float(train_cutoff), 'operator': 'strict_before'}, out)
        relations = sorted({et[1] for et in g.canonical_etypes})
        provenance = dict(provenance or {})
        files = {name: {'sha256': _sha256_file(os.path.join(staging, name)),
                        'bytes': os.path.getsize(os.path.join(staging, name))}
                 for name in sorted(os.listdir(staging))}
        manifest = {'schema_version': GRAPH_BUNDLE_SCHEMA,
                    'timestamp_dtype': 'float64', 'mapping_semantics': 'graph_local_to_global_v3',
                    'relations': relations, 'total_edges': sum(g.num_edges(et) for et in g.canonical_etypes),
                    'num_nodes': {nt: g.num_nodes(nt) for nt in NODE_TYPES},
                    'train_cutoff': float(train_cutoff), 'files': files,
                    'source_hashes': dict(source_hashes or {}), 'provenance': provenance,
                    'training_ready': False,
                    'training_gate_reason': 'requires verified sample/label protocol and supported model schema',
                    'memory_schema_supported': stream['_meta']['memory_schema_supported']}
        with open(os.path.join(staging, 'manifest.json'), 'w', encoding='utf-8') as out:
            json.dump(manifest, out, ensure_ascii=False, sort_keys=True, indent=2)
        if os.path.exists(output_dir):
            raise FileExistsError(output_dir)
        os.rename(staging, output_dir)
        return manifest
    except BaseException:
        # Cleanup stays inside the explicitly selected output parent.
        if os.path.dirname(os.path.realpath(staging)) != os.path.realpath(parent):
            raise RuntimeError('refusing cleanup outside graph output parent')
        shutil.rmtree(staging)
        raise


def build_graph_bundle(preprocess_dir, output_dir, node_types_path, train_cutoff, model_relations=None):
    """Convert trusted v3 event edges, preserving exact raw fields as provenance.

    Monetary model values stay masked to zero; raw uints are dictionary encoded,
    not converted to float or re-normalized. This is a diagnostic bundle pending
    the independent prospective sample/label gate.
    """
    preprocess_dir = os.path.abspath(os.fspath(preprocess_dir))
    manifest_path = os.path.join(preprocess_dir, 'manifest.json')
    with open(manifest_path, encoding='utf-8') as source:
        manifest = json.load(source)
    if (manifest.get('schema_version') != 'bhtg_events_v3' or manifest.get('status') != 'complete'
            or manifest.get('snapshot_verified') is not True):
        raise ValueError('graph bundle requires a complete trusted v3 preprocessing manifest')
    files = manifest.get('files', {})
    for filename, file_record in files.items():
        if os.path.basename(filename) != filename:
            raise ValueError('preprocessing artifact name cannot traverse directories')
        if file_record.get('sha256') != _sha256_file(os.path.join(preprocess_dir, filename)):
            raise ValueError('preprocessing artifact hash mismatch: %s' % filename)
    edges_path = os.path.join(preprocess_dir, 'edges.csv')
    entry = files.get('edges.csv', {})
    if entry.get('sha256') != _sha256_file(edges_path):
        raise ValueError('preprocessing edge hash mismatch')
    frame = pd.read_csv(edges_path, dtype=str, keep_default_na=False)
    required = {'source_address', 'target_address', 'relation', 'event_uid', 'block_timestamp'}
    if required - set(frame):
        raise ValueError('missing v3 edge identity/address fields')
    if len(frame) != int(manifest.get('trusted_edges', -1)):
        raise ValueError('preprocessing trusted edge count mismatch')
    frame['record_kind'] = 'log'
    creations_path = os.path.join(preprocess_dir, 'creations.jsonl')
    creations = []
    if int(manifest.get('trusted_creations', 0)):
        if 'creations.jsonl' not in files:
            raise ValueError('trusted creation artifact is missing from manifest')
        with open(creations_path, encoding='utf-8') as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                if (record.get('verification') not in ('verified_direct', 'verified_internal') or
                        record.get('status') != 'complete' or not record.get('evidence') or
                        not record.get('creation_uid')):
                    raise ValueError('unverified creation cannot enter a graph bundle')
                row = {key: (json.dumps(value, sort_keys=True) if isinstance(value, (dict, list))
                             else str(value)) for key, value in record.items()}
                row.update(source_address=row['creator'], target_address=row['contract_address'],
                           relation='create', edge_uid=row['creation_uid'], event_uid='',
                           record_kind='creation', leg_role='deployment')
                creations.append(row)
        if len(creations) != int(manifest['trusted_creations']):
            raise ValueError('trusted creation count mismatch')
        frame = pd.concat([frame, pd.DataFrame(creations)], ignore_index=True).fillna('')
    if frame.empty:
        raise ValueError('no trusted edges/creations are available for graph construction')
    if set(frame['relation']) - set(GRAPH_RELATIONS):
        raise ValueError('unsupported relation schema in trusted event edges')
    snapshot = manifest.get('snapshot', {})
    node_types = {}
    with open(node_types_path, encoding='utf-8') as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            address = str(record.get('address', '')).lower()
            if record.get('status') != 'complete' or record.get('node_type') not in NODE_TYPES:
                continue
            if (str(record.get('chain_id')) != str(snapshot.get('chain_id')) or
                    str(record.get('block_number')) != str(snapshot.get('end_block')) or
                    str(record.get('block_hash', '')).lower() != str(snapshot.get('end_block_hash', '')).lower()):
                raise ValueError('node type evidence snapshot differs from event source')
            if record['node_type'] == 'pool':
                from bhtg.event_schema import normalize_address
                proof = record.get('evidence', {})
                if (record.get('protocol') != 'v2' or
                        record.get('pool_verification') != 'verified_factory_pair' or
                        record.get('verification') != 'verified_v2_factory_pair' or
                        not record.get('discovery_scope') or not isinstance(proof, dict)):
                    raise ValueError('pool node requires verified Factory/pair identity proof')
                factory = normalize_address(record.get('factory'), 'factory', False)
                token0 = normalize_address(record.get('token0'), 'token0', False)
                token1 = normalize_address(record.get('token1'), 'token1', False)
                if (token0 == token1 or str(proof.get('factory_call', '')).lower() != factory or
                        str(proof.get('get_pair', '')).lower() != address or
                        str(proof.get('token0', '')).lower() != token0 or
                        str(proof.get('token1', '')).lower() != token1):
                    raise ValueError('pool Factory/pair proof contradicts node identity')
            if address in node_types and node_types[address] != record['node_type']:
                raise ValueError('conflicting node type proofs for %s' % address)
            node_types[address] = record['node_type']
    frame['source_address'] = frame['source_address'].str.lower()
    frame['target_address'] = frame['target_address'].str.lower()
    addresses = sorted(set(frame['source_address']) | set(frame['target_address']) |
                       (set(node_types) if model_relations is not None else set()))
    if set(addresses) - set(node_types):
        raise ValueError('unknown/unverified endpoint node types; do not default to EOA')
    addr_to_id = {address: index for index, address in enumerate(addresses)}
    declared_tokens = {value.lower() for value in frame.get('token_address', []) if value}
    declared_tokens.update(row['contract_address'].lower() for row in creations)
    for column in ('token0_address', 'token1_address'):
        if column in frame:
            declared_tokens.update(value.lower() for value in frame[column] if value)
    tokens = sorted(declared_tokens & set(addr_to_id))
    vocabularies, edges_dict = {}, {}
    provenance_columns = [column for column in frame if column not in required - {'event_uid'}]
    for column in provenance_columns:
        vocabularies[column] = sorted(set(frame[column]))
    vocab_maps = {column: {value: index for index, value in enumerate(values)}
                 for column, values in vocabularies.items()}
    for relation, rows in frame.groupby('relation', sort=True):
        timestamps = []
        for value in rows['block_timestamp']:
            if not value.isdigit():
                raise ValueError('block timestamps must be exact integer UTC seconds')
            timestamp = int(value)
            if timestamp > 2 ** 53:
                raise ValueError('timestamp cannot be represented exactly as float64 seconds')
            timestamps.append(timestamp)
        fields = {'src': torch.tensor([addr_to_id[value] for value in rows['source_address']]),
                  'dst': torch.tensor([addr_to_id[value] for value in rows['target_address']]),
                  'timestamp': torch.tensor(timestamps, dtype=torch.float64),
                  'value': torch.zeros(len(rows)), 'label': torch.full((len(rows),), -1, dtype=torch.long)}
        if model_relations is not None:
            from bhtg.event_schema import exact_integer
            for column in ('chain_id', 'block_number', 'transaction_index', 'log_index'):
                numbers = []
                for _, row in rows.iterrows():
                    if column == 'log_index' and row.get('record_kind') == 'creation':
                        # Sorting position for a receipt, not a fabricated native log.
                        number = -1
                    else:
                        number = exact_integer(row.get(column), column)
                    if number > 2**63-1:
                        raise ValueError('native order index exceeds int64: ' + column)
                    numbers.append(number)
                fields['native_' + column] = torch.tensor(numbers, dtype=torch.long)
            fields['native_is_creation'] = torch.tensor(
                [row.get('record_kind') == 'creation' for _,row in rows.iterrows()], dtype=torch.long)
        for column in provenance_columns:
            fields[column + '_code'] = torch.tensor([vocab_maps[column][value] for value in rows[column]],
                                                    dtype=torch.long)
        if 'token_address' in rows:
            fields['token_gid'] = torch.tensor([addr_to_id.get(value.lower(), -1)
                                               for value in rows['token_address']], dtype=torch.long)
        from bhtg.model_schema_v3 import POOL_EDGE_FEATURE_FIELDS
        if any(column in rows for column in POOL_EDGE_FEATURE_FIELDS):
            from decimal import Decimal, InvalidOperation
            pool_numeric = {}
            for column in POOL_EDGE_FEATURE_FIELDS:
                numbers = []
                for value in rows.get(column, pd.Series([''] * len(rows))):
                    try:
                        number = Decimal(str(value or '0'))
                    except InvalidOperation:
                        raise ValueError('invalid causal pool feature: ' + column)
                    if not number.is_finite() or number < 0 or number > 1:
                        raise ValueError('causal pool feature outside [0, 1]: ' + column)
                    if column.endswith('_known') and number not in (0, 1):
                        raise ValueError('pool evidence mask must be boolean')
                    numbers.append(float(number))
                pool_numeric[column] = torch.tensor(numbers, dtype=torch.float64)
            known = pool_numeric['lp_drain_known'].bool()
            if bool((known & ~pool_numeric['lp_reserve_known'].bool()).any()):
                raise ValueError('known drain requires known historical reserves')
            for column in ('lp_drain0', 'lp_drain1', 'lp_drain_max'):
                pool_numeric[column] = torch.where(known, pool_numeric[column], torch.zeros_like(pool_numeric[column]))
            if not torch.allclose(pool_numeric['lp_drain_max'][known],
                                  torch.maximum(pool_numeric['lp_drain0'][known], pool_numeric['lp_drain1'][known]),
                                  atol=1e-12, rtol=1e-12):
                raise ValueError('drain max contradicts token ratios')
            fields.update(pool_numeric)
        edges_dict[relation] = fields
    graph, _, _ = build_hetero_graph(edges_dict, addr_to_id, tokens, chain_node_types=node_types,
                                    train_cutoff=train_cutoff)
    return write_graph_bundle(graph, addr_to_id, tokens, output_dir, train_cutoff,
                              source_hashes={'preprocess_manifest': _sha256_file(manifest_path),
                                             'edges': _sha256_file(edges_path),
                                             'node_types': _sha256_file(node_types_path)},
                              provenance={'amount_policy': 'model_value_masked_zero_raw_uint_preserved',
                                          'sample_labels_verified': False, 'source_snapshot': snapshot,
                                          'source_manifest_status': manifest['status']},
                              string_vocabularies=vocabularies, edge_records=frame, model_relations=model_relations)


def build_four_actions_bundle(action_csv, output_dir, train_cutoff, use_unlabeled=False,
                              amount_mode=None, amount_policy=None, train_pool_overlay=None):
    """Explicitly build the original four-relation model from one canonical CSV.

    Source rows keep their identity when Transfer becomes two graph legs. The
    address map is always derived afresh; archived IDs and embeddings are never
    reused. This is an original-model input conversion, not the strict eight-
    relation v3 evidence/label pipeline.
    """
    from bhtg.action_csv import iter_action_chunks, _graph_legs, active_actions_selection
    from bhtg import config
    if amount_mode is None:
        amount_mode = 'continuous_zscore' if getattr(config, 'AMOUNT_EXPERIMENT', False) else 'legacy'
    if amount_mode not in ('legacy', 'disabled', 'continuous_zscore'):
        raise ValueError('unsupported four-actions graph amount mode')
    numeric = amount_mode != 'legacy'
    _check_new_output(output_dir)
    if train_cutoff is None or not np.isfinite(train_cutoff):
        raise ValueError('an explicit finite training cutoff is required')
    action_csv = os.path.abspath(os.fspath(action_csv))
    source_hash = _sha256_file(action_csv)
    pool_overlay = None
    if train_pool_overlay is not None:
        from bhtg.pool_graph_overlay import load_train_pool_overlay
        pool_overlay = load_train_pool_overlay(train_pool_overlay, source_hash, train_cutoff)
    roles = ['labeled', 'creation'] + (['unlabeled'] if use_unlabeled else [])
    columns = ['edge_type', 'src_addr', 'dst_addr', 'token_addr', 'timestamp', 'value', 'label',
               'source_row_id', 'source_role', 'native_order_available',
               'creation_kind', 'verification', 'factory_address']
    amount_audit = None
    if numeric:
        from bhtg.amounts import source_policy_for_actions, amount_metadata, audit_actions
        amount_policy = source_policy_for_actions(action_csv) if amount_policy is None else amount_policy
        amount_audit = audit_actions(action_csv, policy=amount_policy)
        columns += ['amount_z', 'amount_known']
    frames, seen_sources, creation_targets, internal_factories = [], set(), set(), set()
    token_labels, canonical_counts = {}, defaultdict(int)
    for canonical in iter_action_chunks(action_csv, source_roles=roles, columns=columns,
                                        amount_mode=amount_mode, amount_policy=amount_policy):
        identities = canonical['source_row_id'].tolist()
        if seen_sources.intersection(identities):
            raise ValueError('source-row identity repeated across four-actions CSV chunks')
        seen_sources.update(identities)
        if not canonical['native_order_available'].isin(['true', 'false']).all():
            raise ValueError('invalid native-order availability flag')
        for token, label in canonical[['token_addr', 'label']].itertuples(index=False, name=None):
            if label != -1:
                if token in token_labels and token_labels[token] != label:
                    raise ValueError('conflicting legacy registry labels for ' + token)
                token_labels[token] = label
        creations = canonical[canonical['edge_type'] == 'create']
        for token, creator, kind, verification, factory in creations[
                ['token_addr', 'src_addr', 'creation_kind', 'verification', 'factory_address']
                ].itertuples(index=False, name=None):
            if token in creation_targets:
                raise ValueError('multiple verified creations for the same target')
            creation_targets.add(token)
            if (kind, verification) not in (('direct', 'verified_direct'),
                                            ('internal', 'verified_internal')):
                raise ValueError('unverified creation cannot enter a four-actions graph')
            if kind == 'internal':
                if factory.lower() != creator:
                    raise ValueError('internal creator differs from verified factory address')
                internal_factories.add(creator)
            elif factory:
                raise ValueError('direct creation cannot carry a factory address')
        for relation, count in canonical['edge_type'].value_counts().items():
            canonical_counts[relation] += int(count)
        frames.append(_graph_legs(canonical))
    if not frames:
        raise ValueError('no selected actions are available for graph construction')
    # A source change during conversion invalidates the mapping/provenance binding.
    if _sha256_file(action_csv) != source_hash:
        raise ValueError('four-actions CSV changed during graph conversion')
    frame = pd.concat(frames, ignore_index=True)
    token_addresses = set(frame['token_addr'])
    tokens = sorted(token_addresses)
    if bool((frame['timestamp'] > 2 ** 53).any()):
        raise ValueError('timestamp cannot be represented exactly as float64 seconds')
    addresses = sorted(set(frame['src_addr']) | set(frame['dst_addr']) | set(tokens))
    if ZERO_ADDR in addresses:
        raise ValueError('zero-address event marker must not become a graph node')
    mapping = {address: index for index, address in enumerate(addresses)}
    selection = active_actions_selection()
    selected_source = bool(selection is not None and
        os.path.normcase(os.path.realpath(selection['actions_csv']['path'])) ==
        os.path.normcase(os.path.realpath(action_csv)) and
        selection['actions_csv']['sha256'] == source_hash and
        selection['dataset_manifest'].get('creation_reverified') == len(creation_targets))
    node_types = {address: 'contract' for address in token_addresses | internal_factories}
    node_type_records = pd.DataFrame([
        {'address': address, 'node_type': node_types.get(address, 'eoa'),
         'type_source': ('token_registry_structural' if address in token_addresses else
                         ('verified_internal_creation_factory' if selected_source else
                          'diagnostic_asserted_internal_creation_factory') if address in internal_factories else
                         'legacy_structural_eoa_fallback'),
         'chain_type_verified': selected_source and address in internal_factories}
        for address in addresses])
    pool_overlay_audit = None
    if pool_overlay is not None:
        from bhtg.pool_graph_overlay import select_train_pool_overlay
        accepted_pools, pool_overlay_audit = select_train_pool_overlay(
            pool_overlay, frame, token_addresses, internal_factories)
        node_types.update({address: 'pool' for address in accepted_pools})
        pool_records = {pool['address']: pool for pool in pool_overlay['pools']}
        decisions = {row['address']: row for row in pool_overlay_audit['records']}
        # A pool role does not replace a supervised target's contract node type.
        # Its train-snapshot role remains visible in this hash-bound sidecar.
        node_type_records['pool_role_at_train_snapshot'] = node_type_records['address'].isin(pool_records)
        node_type_records['pool_static_overlay_accepted'] = node_type_records['address'].isin(accepted_pools)
        for column in ('pool_overlay_decision', 'pool_creation_block_number',
                       'pool_role_valid_from_timestamp', 'pool_verified_snapshot',
                       'pool_verification_block_number', 'pool_verification_block_hash',
                       'pool_candidate_identity_sha256', 'pool_same_second_order_unknown',
                       'pool_identity_scope'):
            node_type_records[column] = None
        record_indices = dict(zip(node_type_records['address'], node_type_records.index))
        for address, pool in pool_records.items():
            index = record_indices.get(address)
            if index is None:
                continue
            decision = decisions[address]
            node_type_records.at[index, 'pool_overlay_decision'] = decision['reason']
            node_type_records.at[index, 'pool_creation_block_number'] = pool['creation_block_number']
            node_type_records.at[index, 'pool_role_valid_from_timestamp'] = pool['creation_block_timestamp']
            node_type_records.at[index, 'pool_verified_snapshot'] = 'train'
            node_type_records.at[index, 'pool_verification_block_number'] = pool['verification_block_number']
            node_type_records.at[index, 'pool_verification_block_hash'] = pool['verification_block_hash']
            node_type_records.at[index, 'pool_candidate_identity_sha256'] = pool['candidate_identity_sha256']
            node_type_records.at[index, 'pool_same_second_order_unknown'] = decision['same_second_order_unknown']
            node_type_records.at[index, 'pool_identity_scope'] = 'train_snapshot_retrospective_identity'
            if address in accepted_pools:
                node_type_records.at[index, 'node_type'] = 'pool'
                node_type_records.at[index, 'type_source'] = 'verified_train_pool_overlay'
                node_type_records.at[index, 'chain_type_verified'] = True
    provenance_columns = ['source_row_id', 'source_role', 'leg_role', 'native_order_available']
    vocabularies = {column: sorted(set(frame[column])) for column in provenance_columns}
    vocab_maps = {column: {value: index for index, value in enumerate(values)}
                 for column, values in vocabularies.items()}
    edges = {}
    for relation, rows in frame.groupby('edge_type', sort=True):
        fields = {'src': torch.tensor([mapping[value] for value in rows['src_addr']], dtype=torch.long),
                  'dst': torch.tensor([mapping[value] for value in rows['dst_addr']], dtype=torch.long),
                  'token_gid': torch.tensor([mapping[value] for value in rows['token_addr']], dtype=torch.long),
                  'timestamp': torch.tensor(rows['timestamp'].to_numpy(), dtype=torch.float64),
                  'value': torch.zeros(len(rows), dtype=torch.float32),
                  'label': torch.tensor(rows['label'].to_numpy(), dtype=torch.long)}
        if numeric:
            fields['amount_z'] = torch.tensor(rows['amount_z'].to_numpy(), dtype=torch.float32)
            fields['amount_known'] = torch.tensor(rows['amount_known'].to_numpy(), dtype=torch.bool)
        for column in provenance_columns:
            fields[column + '_code'] = torch.tensor(
                [vocab_maps[column][value] for value in rows[column]], dtype=torch.long)
        edges[relation] = fields
    graph, _, _ = build_hetero_graph(edges, mapping, tokens, chain_node_types=node_types,
                                    train_cutoff=train_cutoff)
    provenance = {
        'source': 'canonical_four_actions_csv',
        'source_path': action_csv,
        'action_identity': 'source_row_id; transfer legs share one source identity',
        'transfer_representation': 'sender_to_token_and_token_to_receiver',
        'event_order': 'timestamp_only; CSV row and graph leg order are not native chain order',
        'amount_policy': 'masked_zero_original_source_text_retained_in_actions_csv',
        'sample_labels_verified': False,
        'label_source': 'legacy_unverified_registry',
        'creation_verified': selected_source,
        'creation_count': len(creation_targets),
        'verified_creation_count': len(creation_targets) if selected_source else 0,
        'creation_verification_source': ('hash_bound_reverified_active_dataset' if selected_source else
                                         'diagnostic_csv_assertions_only'),
        'canonical_action_counts': dict(canonical_counts),
        'selected_source_roles': roles,
        'node_type_policy': 'token_registry_structural_plus_internal_factories; other_addresses_eoa_fallback',
        'chain_endpoint_types_fully_verified': False,
        'old_mapping_reused': False,
    }
    if numeric:
        provenance.update(amount_policy='continuous_saved_zscore_separate_from_zero_legacy_value',
                          amount_protocol=amount_metadata(), amount_audit=amount_audit)
    if pool_overlay is not None:
        provenance['node_type_policy'] = (
            'token_registry_structural_plus_internal_factories_plus_authenticated_train_pool_overlay; '
            'other_addresses_eoa_fallback; supervised_target_contract_type_preserved')
        provenance['train_pool_overlay'] = {
            'schema': pool_overlay['schema'], 'sha256': pool_overlay['sha256'],
            'source_actions_sha256': pool_overlay['source_actions_sha256'],
            'snapshot': pool_overlay['snapshot'], 'audit': pool_overlay_audit,
            'scope': 'train_snapshot_retrospective_identity; not_earlier_block_by_block_verification',
            'event_order': 'timestamp_only; same_second_creation_action_order_unknown'}
        if pool_overlay['pool_roles_by_snapshot'] is not None:
            provenance['train_pool_overlay']['pool_roles_by_snapshot'] = pool_overlay['pool_roles_by_snapshot']
    edge_records = frame[['edge_type', 'src_addr', 'dst_addr', 'token_addr', 'timestamp',
                          'label', 'value'] + (['amount_z', 'amount_known'] if numeric else []) + provenance_columns]
    if _sha256_file(action_csv) != source_hash:
        raise ValueError('four-actions CSV changed before graph publication')
    source_hashes = {'actions_csv': source_hash}
    if pool_overlay is not None:
        if _sha256_file(pool_overlay['path']) != pool_overlay['sha256']:
            raise ValueError('train pool overlay changed before graph publication')
        source_hashes['train_pool_overlay'] = pool_overlay['sha256']
        role_table = pool_overlay['pool_roles_by_snapshot']
        if role_table is not None:
            if _sha256_file(role_table['path']) != role_table['sha256']:
                raise ValueError('pool snapshot roles changed before graph publication')
            source_hashes['pool_roles_by_snapshot'] = role_table['sha256']
    if selected_source:
        source_hashes['actions_manifest'] = selection['manifest']['sha256']
    manifest = write_graph_bundle(graph, mapping, tokens, output_dir, train_cutoff,
                                  source_hashes=source_hashes, provenance=provenance,
                                  string_vocabularies=vocabularies, edge_records=edge_records,
                                  node_type_records=node_type_records)
    return graph, manifest


def run_graph_builder(use_unlabeled=False, output_dir=None, train_cutoff=None):
    """Explicitly convert the selected CSV input into a new graph directory."""
    if output_dir is None:
        raise ValueError('choose a new output_dir; frozen production inputs cannot be rebuilt in place')
    _check_new_output(output_dir)
    from bhtg import config
    if getattr(config, 'ACTION_CSV', None):
        graph, _ = build_four_actions_bundle(config.ACTION_CSV, output_dir, train_cutoff,
                                              use_unlabeled=use_unlabeled)
        return graph
    frames = [pd.read_csv(LABELED_EDGES_CSV)]
    if use_unlabeled:
        frames.append(pd.read_csv(UNLABELED_EDGES_CSV))
    frame = pd.concat(frames, ignore_index=True)
    addresses = pd.read_csv(ADDR_TO_ID_CSV)
    mapping = dict(zip(addresses['address'], addresses['id']))
    tokens = pd.read_csv(TOKEN_LIST_CSV)['token_address'].tolist()
    edges = {}
    for relation, rows in frame.groupby('edge_type', sort=True):
        edges[relation] = {'src': torch.tensor(rows['src_id'].values, dtype=torch.long),
                           'dst': torch.tensor(rows['dst_id'].values, dtype=torch.long),
                           'timestamp': torch.tensor(rows['timestamp'].values, dtype=torch.float64),
                           'value': torch.tensor(rows['value'].values, dtype=torch.float32),
                           'label': torch.tensor(rows['label'].values, dtype=torch.long)}
    graph, _, _ = build_hetero_graph(edges, mapping, tokens, train_cutoff=train_cutoff)
    write_graph_bundle(graph, mapping, tokens, output_dir, train_cutoff,
                       provenance={'source': 'legacy_csv_diagnostic_only', 'creation_verified': False})
    return graph


def rebuild_with_chain_data(chain_dir=None, **kwargs):
    """Reject the obsolete unverified creation/full-history LP backfill path."""
    raise RuntimeError('legacy chain rebuild is unsafe: unverified creation and future LP statistics; '
                       'use build_graph_bundle with trusted v3 events and verified node type proofs')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build an explicit graph bundle in a fresh directory; no training')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--preprocess-dir', help='trusted v3 preprocessing directory')
    inputs.add_argument('--actions-csv', help='canonical four-actions CSV; unselected files remain diagnostic')
    inputs.add_argument('--selected-actions', action='store_true', help='use the hash-bound active four-actions CSV')
    parser.add_argument('--node-types', help='verified fixed-snapshot node_types.jsonl; required for v3 inputs')
    parser.add_argument('--output-dir', required=True, help='fresh directory outside frozen bhtg/data')
    parser.add_argument('--train-cutoff', required=True, type=float, help='explicit strict-before UTC seconds cutoff')
    parser.add_argument('--model-schema', choices=('legacy', 'v3'))
    parser.add_argument('--use-unlabeled', action='store_true', help='include unlabeled source rows in four-action inputs')
    parser.add_argument('--train-pool-overlay', help='source-bound authenticated train-snapshot pool identity JSON')
    args = parser.parse_args(argv)
    if args.preprocess_dir:
        if args.train_pool_overlay:
            parser.error('--train-pool-overlay is only supported for canonical four-actions inputs')
        if not args.node_types:
            parser.error('--preprocess-dir requires --node-types verified fixed-snapshot proofs')
        if args.use_unlabeled:
            parser.error('--use-unlabeled is only supported for canonical four-actions inputs')
        from bhtg.model_schema_v3 import V3_EDGE_TYPES
        manifest = build_graph_bundle(args.preprocess_dir, args.output_dir, args.node_types, args.train_cutoff,
                                      model_relations=V3_EDGE_TYPES if args.model_schema != 'legacy' else None)
    else:
        if args.node_types or args.model_schema == 'v3':
            parser.error('four-actions CSV supports the original legacy relations; use trusted --preprocess-dir for v3')
        from bhtg import config
        action_csv = config.ACTION_CSV if args.selected_actions else args.actions_csv
        if not action_csv:
            parser.error('no hash-bound active four-actions CSV is selected')
        _, manifest = build_four_actions_bundle(action_csv, args.output_dir, args.train_cutoff,
                                               use_unlabeled=args.use_unlabeled,
                                               train_pool_overlay=args.train_pool_overlay)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return manifest


if __name__ == '__main__':
    main()
