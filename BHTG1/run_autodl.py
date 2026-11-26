"""Build the four-action graph and train every stage in a new AutoDL run."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from datetime import datetime


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_sha256(path, expected):
    if not path.is_file():
        raise FileNotFoundError("Missing input: %s" % path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("Input SHA256 mismatch: %s" % path)


def check_inputs(root):
    from bhtg.amounts import AUDIT_EVIDENCE, PREPROCESS_CODE
    from bhtg.pool_graph_overlay import load_train_pool_overlay

    actions = root / "data" / "actions"
    sources = root / "data" / "source"
    selection = read_json(actions / "active_actions.json")
    if selection["schema_version"] != "bhtg_active_four_actions_v1":
        raise ValueError("Unexpected active actions schema")
    for key, filename in (("actions_csv", "actions.csv"), ("manifest", "manifest.json")):
        record = selection[key]
        if record["path"] != filename:
            raise ValueError("Expected a portable path for %s: %s" % (key, filename))
        verify_sha256(actions / filename, record["sha256"])
    dataset = read_json(actions / "manifest.json")
    if (dataset["schema_version"] != "bhtg_four_actions_csv_v1" or
            dataset["status"] != "complete" or
            dataset["actions_csv"]["sha256"] != selection["actions_csv"]["sha256"]):
        raise ValueError("Actions and dataset manifest do not match")
    for filename in ("source_manifest.json", "train_pool_overlay.json",
                     "pool_roles_by_snapshot.jsonl"):
        verify_sha256(actions / filename, dataset["files"][filename]["sha256"])
    ledger = read_json(actions / "source_manifest.json")
    if ledger["schema_version"] != "bhtg_four_actions_sources_v1":
        raise ValueError("Unexpected source ledger schema")
    audit = sources / "transfer_preprocess_audit.json"
    verify_sha256(audit, AUDIT_EVIDENCE)
    evidence = {item["sha256"]: item for item in read_json(audit)["files"] if item.get("sha256")}
    audited_code = sources / "load_data.py"
    verify_sha256(audited_code, PREPROCESS_CODE)
    aliases = {
        ledger["expected_source_sha_from"]: audit,
        evidence[PREPROCESS_CODE]["path"]: audited_code,
    }
    for role, record, filename in (
        ("labeled", ledger["sources"]["labeled"], "transfer_std_label.csv"),
        ("unlabeled", ledger["sources"]["unlabeled"], "token_transfer.csv"),
        ("creation", ledger["creation"], "creation_verified.csv"),
    ):
        if record != dataset["source_inputs"][role]:
            raise ValueError("Dataset and source ledger differ for %s" % role)
        path = sources / filename
        verify_sha256(path, record["sha256"])
        aliases[record["path"]] = path
    for old_path in aliases:
        # These archived Windows paths are relative single components to POSIX.
        if not old_path or "\0" in old_path or "/" in old_path or (
                PurePosixPath(old_path).name != old_path or old_path in {".", ".."}):
            raise ValueError("Cannot alias this archived source path: %s" % old_path)
    overlay_file = actions / "train_pool_overlay.json"
    overlay = read_json(overlay_file)
    if overlay["pool_roles_by_snapshot"]["path"] != "pool_roles_by_snapshot.jsonl":
        raise ValueError("Expected the snapshot-role file beside the pool overlay")
    load_train_pool_overlay(overlay_file, selection["actions_csv"]["sha256"], 1620305417)
    print("Dataset, source audit and train pool evidence verified.", flush=True)
    return aliases


def train(root, aliases):
    if os.name != "posix":
        raise SystemExit("Run the full pipeline on AutoDL Linux. Use --check-only for local checks.")
    env = os.environ.copy()
    for key in ("BHTG_BUNDLE_DIR", "PHASE1_TAG", "PHASE3_GRAPH_CKPT"):
        env.pop(key, None)
    env.update({
        "PYTHONPATH": str(root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "DGLBACKEND": "pytorch",
        "BHTG_USE_AGGREGATED_ACTIONS": "1",
        "BHTG_ACTIONS_MANIFEST": str(root / "data" / "actions" / "active_actions.json"),
        "BHTG_NODE_FEATURE_MODE": "behavior9",
        "BHTG_EXPERIMENT_PROTOCOL": "unlabeled_phase2_full_history_phase3_v1",
        "BHTG_SEMANTIC_AMOUNT_MODE": "continuous_zscore",
        "BHTG_MEMORY_AMOUNT_MODE": "continuous_zscore",
        "BHTG_AMOUNT_SOURCE_POLICY": "rugscreener_saved_zscore_v1",
        "BHTG_AMOUNT_MASK_RATE": "0.15",
        "BHTG_AMOUNT_LOSS_WEIGHT": "0.1",
        "BHTG_AMOUNT_MLP_HIDDEN": "64",
        "BHTG_NODE_FEATURE_WINDOW_SECONDS": "604800",
        "BHTG_NODE_FEATURE_MIN_HISTORY": "5",
        "PHASE2_TAG": "amount_causal_v2",
        "PHASE2_EPOCHS": "10",
        "PHASE2_MAX_BATCHES": "600",
        "PHASE2_SSL": "1",
        "PHASE2_STRICT_TEMPORAL": "1",
        "PHASE3_TAG": "amount_causal_v2",
        "PHASE3_ALL_TOKENS": "1",
        "PHASE3_EVAL_PROTOCOL": "token_stratified",
        "PHASE3_STRICT_TEMPORAL": "1",
        "PHASE3_PERMUTE_LABELS": "0",
        "RELATION_BALANCED_SAMPLING": "0",
        "HOMO_ABLATION": "0",
    })
    python = [sys.executable, "-u", "-B"]
    subprocess.run(
        python + ["-c",
                  "import torch, dgl, numpy, pandas, sklearn, transformers\n"
                  "if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')\n"
                  "print('DGL CUDA check:', dgl.graph(([0], [1])).to('cuda').device)"],
        cwd=root, env=env, check=True,
    )
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("autodl_%Y%m%d_%H%M%S_"),
                               dir=output))
    graph = run / "graph"
    weights = run / "training" / "behavior9_unlabeled_full_history"
    weights.mkdir(parents=True)
    env["BHTG_FOUR_ACTIONS_GRAPH_DIR"] = str(graph)
    env["BHTG_OUTPUT_DIR"] = str(weights)
    warmup = [str(root / "warmup_training.py")]
    commands = (
        ("Build graph", ["-m", "bhtg.graph_builder", "--selected-actions",
                         "--output-dir", str(graph), "--train-cutoff", "1620305417",
                         "--use-unlabeled", "--train-pool-overlay",
                         str(root / "data" / "actions" / "train_pool_overlay.json")]),
        ("Phase 1", warmup + [str(root / "run_phase1_lmae.py"), "--epochs", "10",
                              "--batch", "32"]),
        ("Phase 2", warmup + [str(root / "run_phase2.py")]),
        ("Memory pretraining", warmup + [str(root / "run_phase2_memory.py"), "--epochs", "8",
                                         "--batch", "2048"]),
        ("Full-history semantic export",
         [str(root / "export_semantic_history.py"), "--checkpoint",
          str(weights / "phase1_sem_lmae_strict.pt"), "--output-dir", str(weights),
          "--full-history-cutoff", "auto"]),
        ("Phase 3 + Joint Memory",
         warmup + [str(root / "run_phase3.py"), "--use-memory", "--joint-memory",
                   "--strict-temporal", "--validation-threshold", "--eval-protocol",
                   "token_stratified", "--graph-path", str(weights / "phase2_graph_amount_causal_v2.pt"),
                   "--sem-path", str(weights / "h_sem_full_history.pt"), "--epochs", "50"]),
    )
    print("Run folder: %s" % run, flush=True)
    # Keep byte-verified manifests unchanged; give legacy source paths a
    # temporary name beside the subprocess working directory on Linux.
    with tempfile.TemporaryDirectory(prefix=".source_paths_", dir=run) as cwd:
        for old_path, path in aliases.items():
            (Path(cwd) / old_path).symlink_to(path)
        for stage, args in commands:
            print("\n=== %s ===" % stage, flush=True)
            subprocess.run(python + args, cwd=cwd, env=env, check=True)
    print("All stages finished. Results: %s" % weights, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify files without starting CUDA or training")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    aliases = check_inputs(root)
    if not args.check_only:
        train(root, aliases)


if __name__ == "__main__":
    main()
