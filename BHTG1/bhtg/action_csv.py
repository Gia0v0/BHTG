"""Read canonical four-action CSVs and derive model graph legs explicitly.

One saved source row is one action. Graph legs are a separate representation;
missing chain/log identity in legacy rows is never reconstructed from row order.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[1]
ZERO = "0x" + "0" * 40
BASE_COLUMNS = ["edge_type", "src_addr", "dst_addr", "token_addr", "timestamp", "label", "value", "source_role", "source_row_id"]
ROLES = {"labeled", "unlabeled", "creation"}
OPERATIONS = {"transfer", "mint", "burn", "create"}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_actions_selection():
    if os.environ.get("BHTG_USE_AGGREGATED_ACTIONS", "1") == "0":
        return None
    path = Path(os.environ.get("BHTG_ACTIONS_MANIFEST", str(PROJECT / "data_versions/active_four_actions.json")))
    if not path.exists():
        if "BHTG_ACTIONS_MANIFEST" in os.environ:
            raise ValueError("explicit four-actions selection is missing")
        return None
    selected = json.loads(path.read_text(encoding="utf-8-sig"))
    if selected.get("schema_version") != "bhtg_active_four_actions_v1":
        raise ValueError("invalid four-actions selection schema")
    for key in ("actions_csv", "manifest"):
        record = selected[key]
        target = Path(record["path"])
        if not target.is_absolute():
            target = path.parent / target
        target = target.resolve()
        if not target.is_file() or file_sha(target) != record["sha256"]:
            raise ValueError("four-actions selected file hash mismatch: " + key)
        selected[key] = dict(record, path=str(target))
    manifest = json.loads(Path(selected["manifest"]["path"]).read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != "bhtg_four_actions_csv_v1" or manifest.get("status") != "complete":
        raise ValueError("four-actions CSV is not complete")
    if manifest["actions_csv"]["sha256"] != selected["actions_csv"]["sha256"]:
        raise ValueError("four-actions CSV/manifest binding mismatch")
    selected["dataset_manifest"] = manifest
    return selected


def _normalize(frame):
    import pandas as pd
    if set(BASE_COLUMNS) - set(frame.columns):
        raise ValueError("four-actions CSV is missing canonical fields")
    if set(frame["edge_type"]) - OPERATIONS or set(frame["source_role"]) - ROLES:
        raise ValueError("invalid four-actions operation or source role")
    if ((frame["edge_type"] == "create") != (frame["source_role"] == "creation")).any():
        raise ValueError("creation action/source role mismatch")
    for column in ("src_addr", "dst_addr", "token_addr"):
        frame[column] = frame[column].str.lower()
        if not frame[column].str.fullmatch(r"0x[0-9a-f]{40}").all():
            raise ValueError("invalid four-actions address")
    if (frame["token_addr"] == ZERO).any():
        raise ValueError("zero token address")
    for column in ("timestamp", "label"):
        if not frame[column].str.fullmatch(r"-?[0-9]+").all():
            raise ValueError("four-actions integer field is invalid: " + column)
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    if (frame["timestamp"] < 0).any() or not frame["label"].isin([-1, 0, 1]).all():
        raise ValueError("invalid timestamp or label")
    if ((frame["source_role"] == "unlabeled") & (frame["label"] != -1)).any():
        raise ValueError("unlabeled source cannot provide a label")
    if ((frame["source_role"] != "unlabeled") & ~frame["label"].isin([0, 1])).any():
        raise ValueError("labeled/creation source requires a binary registry label")
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    if (frame["value"] != 0).any():
        raise ValueError("four-actions model amount must remain masked zero")
    if (frame["source_row_id"] == "").any() or frame["source_row_id"].duplicated().any():
        raise ValueError("missing or repeated source-row identity")
    ordinary = frame["edge_type"] != "create"
    src_zero, dst_zero = frame["src_addr"] == ZERO, frame["dst_addr"] == ZERO
    if (ordinary & src_zero & dst_zero).any():
        raise ValueError("double-zero legacy row")
    expected = frame["edge_type"].copy()
    expected.loc[ordinary & src_zero] = "mint"
    expected.loc[ordinary & ~src_zero & dst_zero] = "burn"
    expected.loc[ordinary & ~src_zero & ~dst_zero] = "transfer"
    if (expected != frame["edge_type"]).any():
        raise ValueError("legacy action/zero-address semantics mismatch")
    create = ~ordinary
    if (create & (src_zero | (frame["dst_addr"] != frame["token_addr"]))).any():
        raise ValueError("invalid creation endpoints")
    return frame


def _graph_legs(frame):
    import pandas as pd
    other = frame[frame["edge_type"] != "transfer"].copy()
    other["leg_role"] = other["edge_type"]
    other.loc[other["edge_type"] == "mint", "src_addr"] = other.loc[other["edge_type"] == "mint", "token_addr"]
    other.loc[other["edge_type"] == "burn", "dst_addr"] = other.loc[other["edge_type"] == "burn", "token_addr"]
    transfers = frame[frame["edge_type"] == "transfer"]
    incoming, outgoing = transfers.copy(), transfers.copy()
    incoming["dst_addr"] = incoming["token_addr"]
    outgoing["src_addr"] = outgoing["token_addr"]
    incoming["leg_role"], outgoing["leg_role"] = "sender_to_token", "token_to_receiver"
    # source_row_id remains the action identity on both legs; no native order is added.
    return pd.concat([other, incoming, outgoing], ignore_index=True)


def iter_action_chunks(path, source_roles=None, columns=None, chunksize=100000, graph_legs=False,
                       amount_mode='legacy', amount_policy=None):
    import pandas as pd
    roles = set(source_roles) if source_roles is not None else ROLES
    if roles - ROLES:
        raise ValueError("unsupported source roles")
    if amount_mode not in ('legacy', 'disabled', 'continuous_zscore'):
        raise ValueError('unsupported action amount mode')
    numeric = amount_mode != 'legacy'
    requested = list(columns) if columns is not None else None
    derived = {'leg_role', 'amount_z', 'amount_known', 'amount_reason'} if numeric else {'leg_role'}
    fields = None if requested is None else sorted(set(BASE_COLUMNS) | (set(requested) - derived))
    if numeric:
        from bhtg.amounts import SOURCE_COLUMNS, source_policy_for_actions, parse_amount_frame
        amount_policy = source_policy_for_actions(path) if amount_policy is None else amount_policy
        if fields is not None:
            fields = sorted(set(fields) | set(SOURCE_COLUMNS))
    seen = set()
    for frame in pd.read_csv(path, dtype=str, keep_default_na=False, usecols=fields, chunksize=chunksize):
        frame = _normalize(frame)
        identities = frame['source_row_id'].tolist()
        if seen.intersection(identities):
            raise ValueError('source-row identity repeated across CSV chunks')
        seen.update(identities)
        frame = frame[frame["source_role"].isin(roles)].copy()
        if numeric:
            frame = parse_amount_frame(frame, amount_policy)
        if graph_legs:
            frame = _graph_legs(frame)
        if requested is not None:
            frame = frame.loc[:, requested]
        if len(frame):
            yield frame


def read_actions(path, source_roles=None, columns=None, chunksize=None, graph_legs=False,
                 amount_mode='legacy', amount_policy=None):
    import pandas as pd
    frames = iter_action_chunks(path, source_roles, columns, chunksize or 100000, graph_legs,
                                amount_mode, amount_policy)
    if chunksize is not None:
        return frames
    frames = list(frames)
    names = columns if columns is not None else BASE_COLUMNS + (["leg_role"] if graph_legs else [])
    if columns is None and amount_mode != 'legacy':
        names += ['amount_z', 'amount_known', 'amount_reason']
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=names)


def model_edges(labeled=True, usecols=None, chunksize=None, legacy_path=None):
    import pandas as pd
    from bhtg import config
    action_path = getattr(config, "ACTION_CSV", None)
    if action_path:
        # Label and timestamp scans need the same ordered graph-leg rows, but
        # their projections do not use the derived amount columns. The graph
        # manifest still binds the amount inputs consumed by the model.
        amount_fields = {'amount_z', 'amount_known', 'amount_reason'}
        needs_amounts = (getattr(config, 'AMOUNT_EXPERIMENT', False) and
                         (usecols is None or not amount_fields.isdisjoint(usecols)))
        return read_actions(action_path, source_roles=["labeled", "creation"] if labeled else ["unlabeled"],
                            columns=usecols, chunksize=chunksize, graph_legs=True,
                            amount_mode='continuous_zscore' if needs_amounts else 'legacy')
    path = legacy_path or (config.LABELED_EDGES_CSV if labeled else config.UNLABELED_EDGES_CSV)
    return pd.read_csv(path, usecols=usecols, chunksize=chunksize)


def model_action_frames(chunksize=100000, legacy_paths=None):
    from bhtg import config
    action_path = getattr(config, "ACTION_CSV", None)
    columns = ["edge_type", "src_addr", "dst_addr", "token_addr", "timestamp"]
    if action_path:
        numeric = getattr(config, 'AMOUNT_EXPERIMENT', False)
        if numeric:
            columns += ['amount_z', 'amount_known', 'source_row_id']
        yield from iter_action_chunks(action_path, columns=columns, chunksize=chunksize,
                                      amount_mode='continuous_zscore' if numeric else 'legacy')
    else:
        for labeled in (True, False):
            path = legacy_paths[0 if labeled else 1] if legacy_paths else None
            yield from model_edges(labeled=labeled, usecols=columns, chunksize=chunksize, legacy_path=path)


def creator_times_from_actions(path):
    times = {}
    for frame in iter_action_chunks(path, source_roles=["creation"], columns=["token_addr", "timestamp"]):
        for token, timestamp in frame.itertuples(index=False, name=None):
            if token in times:
                raise ValueError("multiple creations for the same target")
            times[token] = int(timestamp)
    return times, {"source": str(Path(path).resolve()), "sha256": file_sha(path),
                   "kind": "verified_creation_rows_in_four_actions", "records": len(times)}


def require_graph_binding(graph_path=None):
    from bhtg import config
    actions = getattr(config, "ACTION_CSV", None)
    if not actions:
        return
    graph = Path(graph_path or config.GRAPH_BIN)
    if graph.name != "bhtg_graph.bin":
        raise RuntimeError("four-actions graph must use the bound bhtg_graph.bin artifact")
    manifest_path = graph.parent / "manifest.json"
    if not graph.is_file() or not manifest_path.is_file():
        raise RuntimeError("The four-actions CSV is connected; build its new graph and mappings before training. "
                           "Frozen old graphs and weights cannot be used with the new CSV.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_hashes", {}).get("actions_csv") != file_sha(actions):
        raise RuntimeError("graph is not bound to the selected four-actions CSV")
    if manifest.get("schema_version") != "bhtg_graph_bundle_v3":
        raise RuntimeError("unsupported four-actions graph manifest")
    amount_protocol = manifest.get('provenance', {}).get('amount_protocol')
    if getattr(config, 'AMOUNT_EXPERIMENT', False):
        from bhtg.amounts import amount_metadata
        if amount_protocol != amount_metadata():
            raise RuntimeError('four-actions graph amount protocol mismatch; build a new amount graph')
        audit = manifest.get('provenance', {}).get('amount_audit', {})
        if (audit.get('actions_sha256') != manifest['source_hashes']['actions_csv'] or
                audit.get('policy', {}).get('verified') is not True):
            raise RuntimeError('four-actions graph lacks a verified amount source audit')
    elif amount_protocol is not None:
        raise RuntimeError('continuous amount graph requires explicit new amount modes')
    required = {'bhtg_graph.bin', 'addr_to_id.csv', 'token_list.csv',
                'contract_gids.pt', 'eoa_gids.pt', 'pool_gids.pt',
                'chain_node_types.csv', 'edge_stream.pt', 'train_cutoff.json',
                'provenance_vocabulary.json'}
    if required - set(manifest.get('files', {})):
        raise RuntimeError("four-actions graph manifest is missing required mapping/artifact hashes")
    for name, record in manifest['files'].items():
        target = graph.parent / name
        if Path(name).name != name or not target.is_file() or file_sha(target) != record["sha256"]:
            raise RuntimeError("four-actions graph artifact hash mismatch: " + name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--csv")
    args = parser.parse_args(argv)
    selected = active_actions_selection()
    path = args.csv or (selected["actions_csv"]["path"] if selected else None)
    if not path:
        parser.error("no selected four-actions CSV")
    operations, roles, identities, creation_targets = Counter(), Counter(), set(), set()
    for frame in iter_action_chunks(path, columns=BASE_COLUMNS):
        operations.update(frame["edge_type"]);roles.update(frame["source_role"])
        ids = frame["source_row_id"].tolist()
        if identities.intersection(ids):
            raise ValueError("source-row identities repeat across chunks")
        identities.update(ids)
        targets = frame.loc[frame["edge_type"] == "create", "token_addr"].tolist()
        if creation_targets.intersection(targets) or len(set(targets)) != len(targets):
            raise ValueError("multiple creation rows for a target")
        creation_targets.update(targets)
    print(json.dumps({"csv": str(Path(path).resolve()), "sha256": file_sha(path),
        "rows": sum(operations.values()), "operations": dict(operations), "source_roles": dict(roles),
        "creation_target_count": len(creation_targets), "training_started": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
