[CmdletBinding()]
param(
    [ValidateSet('Inspect','Reuse','All','Phase1','Phase2','Memory','Phase3')]
    [string]$Phase = 'Inspect',
    [string]$RunDir = (Join-Path $PSScriptRoot ('experiments\behavior9_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))),
    [string]$GraphDir = (Join-Path $PSScriptRoot 'experiments\pool_create_amount_joint_20261111\graph'),
    [string]$ActionsManifest = (Join-Path $PSScriptRoot 'experiments\pool_create_amount_joint_20261111\active_actions.json'),
    [string]$UpstreamDir = (Join-Path $PSScriptRoot 'experiments\pool_create_amount_semantic_joint_restart_20261114_2124\outputs'),
    [string]$PythonExe = 'D:\anaconda3\envs\rug\python.exe',
    [ValidateSet('A','B','C','D')][string]$Group = 'D',
    [ValidateSet('behavior9','legacy15')][string]$NodeFeatureMode = 'behavior9',
    [ValidateRange(1,2147483647)][int]$Phase1Epochs = 10,
    [ValidateRange(1,2147483647)][int]$Phase2Epochs = 10,
    [ValidateRange(1,2147483647)][int]$Phase2MaxBatches = 600,
    [ValidateRange(1,2147483647)][int]$MemoryEpochs = 8,
    [ValidateRange(1,2147483647)][int]$Phase3Epochs = 50,
    [ValidateRange(1,2147483647)][int]$WindowSeconds = 604800,
    [ValidateRange(1,2147483647)][int]$MinHistory = 5,
    [switch]$FreshUpstream,
    [switch]$JointMemory,
    [switch]$LegacyTimeProtocol,
    [string]$Phase1Checkpoint,
    [string]$ResumePhase3
)
$ErrorActionPreference = 'Stop'
# The default is read-only inspection. All training stages are explicit, use
# the existing amount graph and separate their outputs by node feature mode.
$taskNames = @('PYTHONPATH','DGLBACKEND','PYTHONDONTWRITEBYTECODE','BHTG_BUNDLE_DIR','BHTG_ACTIONS_MANIFEST',
    'BHTG_USE_AGGREGATED_ACTIONS','BHTG_SEMANTIC_AMOUNT_MODE','BHTG_MEMORY_AMOUNT_MODE',
    'BHTG_EXPERIMENT_PROTOCOL',
    'BHTG_FOUR_ACTIONS_GRAPH_DIR','BHTG_OUTPUT_DIR','BHTG_NODE_FEATURE_MODE',
    'BHTG_NODE_FEATURE_WINDOW_SECONDS','BHTG_NODE_FEATURE_MIN_HISTORY',
    'PHASE2_TAG','PHASE3_TAG','PHASE2_EPOCHS','PHASE2_MAX_BATCHES',
    'PHASE2_STRICT_TEMPORAL','PHASE3_ALL_TOKENS','PHASE3_EVAL_PROTOCOL',
    'PHASE3_STRICT_TEMPORAL','PHASE3_GRAPH_CKPT','PHASE3_PERMUTE_LABELS',
    'RELATION_BALANCED_SAMPLING','HOMO_ABLATION')
$taskSaved = @{}
foreach ($taskName in $taskNames) { $taskSaved[$taskName] = [Environment]::GetEnvironmentVariable($taskName,'Process') }
function Invoke-BehaviorPython([string[]]$Arguments) {
    if ($Arguments[0] -in @('run_phase2.py','run_phase3.py')) {
        & $PythonExe -u -B (Join-Path $PSScriptRoot 'warmup_training.py') @Arguments
    } else {
        & $PythonExe -u -B @Arguments
    }
    if ($LASTEXITCODE -ne 0) { throw "Stage failed (Python exit $LASTEXITCODE); subsequent stages stopped." }
}
function Assert-NewStage([string[]]$Names) {
    foreach ($taskFile in $Names) {
        $taskPath = Join-Path $taskOutput $taskFile
        if (Test-Path -LiteralPath $taskPath) { throw "Existing stage artifact: $taskPath. Use a fresh RunDir or select the next stage." }
    }
}
$taskReuseCode = @'
import json, os, shutil, sys, tempfile
from pathlib import Path
import torch
from bhtg import config
from bhtg.amount_protocol import experiment_metadata, validate_amount_metadata, file_sha256
from bhtg.edge_stream import compute_phase_cutoffs
from bhtg.training_protocol import validate_semantic_bundle
from bhtg.memory import HeteroTemporalMemory, MEMORY_AMOUNT_SEMANTICS
kind, source, destination = sys.argv[1:]
source, destination = Path(source).resolve(), Path(destination).resolve()
if destination.exists():
    source = destination  # Validate and retain an already prepared artifact.
try:
    if not source.is_file():
        raise ValueError('source checkpoint is unavailable')
    checkpoint = torch.load(str(source), map_location='cpu')
    expected = experiment_metadata()
    validate_amount_metadata(checkpoint.get('amount_experiment'), expected)
    if checkpoint.get('smoke_subset') or checkpoint.get('smoke'):
        raise ValueError('smoke checkpoint is not reusable for a full experiment')
    train, validation = compute_phase_cutoffs()
    if kind == 'semantic':
        count = len(torch.load(os.path.join(config.BHTG_DATA_DIR, 'contract_gids.pt'), map_location='cpu'))
        validate_semantic_bundle(checkpoint, count, config.HIDDEN_DIM, train, validation,
                                 semantic_amount_mode=config.SEMANTIC_AMOUNT_MODE,
                                 amount_experiment=expected)
    else:
        if (checkpoint.get('memory_semantics') != MEMORY_AMOUNT_SEMANTICS or
                checkpoint.get('train_cutoff') != train or checkpoint.get('val_cutoff') != validation or
                checkpoint.get('amount_mode') != config.MEMORY_AMOUNT_MODE or
                not checkpoint.get('memory_state')):
            raise ValueError('Memory protocol, amount mode, or observation cutoffs differ')
        # Validate every schema field and actual weight shape without allocating
        # the full graph's runtime memory tables.
        memory = HeteroTemporalMemory({i: 0 for i in range(len(config.NODE_TYPES))},
                                      mem_dim=config.MEM_DIM, temporal_dim=config.TEMPORAL_DIM,
                                      amount_mode=config.MEMORY_AMOUNT_MODE)
        memory.load_weights(checkpoint)
    if source != destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp',
                                         dir=str(destination.parent))
        try:
            with os.fdopen(fd, 'wb') as target, source.open('rb') as original:
                shutil.copyfileobj(original, target)
                target.flush()
                os.fsync(target.fileno())
            if file_sha256(temporary) != file_sha256(source):
                raise ValueError('checkpoint changed or copied incompletely')
            # On Windows rename is atomic and refuses to replace an existing
            # destination, preserving another process's prepared checkpoint.
            os.rename(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        record = {'source': str(source), 'destination': str(destination),
                  'sha256': file_sha256(destination), 'kind': kind,
                  'node_features_independent': True}
        destination.with_suffix(destination.suffix + '.reuse.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Compatible upstream retained: ' + str(destination))
except Exception as exc:
    print('Upstream not reused (%s): %s' % (source, exc))
    sys.exit(2)
'@
function Use-CompatibleUpstream([string]$Kind, [string]$Name) {
    $taskDestination = Join-Path $taskOutput $Name
    if ($FreshUpstream -and -not (Test-Path -LiteralPath $taskDestination)) { return $false }
    $taskMessages = & $PythonExe -B -c $taskReuseCode $Kind (Join-Path $UpstreamDir $Name) $taskDestination 2>&1
    $taskCode = $LASTEXITCODE
    foreach ($taskMessage in $taskMessages) { Write-Host $taskMessage }
    if ($taskCode -notin @(0,2)) { throw "Upstream validation failed (Python exit $taskCode)." }
    if ($taskCode -eq 2 -and (Test-Path -LiteralPath $taskDestination)) {
        throw "Existing incompatible artifact: $taskDestination. Use a fresh RunDir."
    }
    return ($taskCode -eq 0)
}
try {
    Push-Location -LiteralPath $PSScriptRoot
    if (-not (Test-Path -LiteralPath $PythonExe)) { throw "Python not found: $PythonExe" }
    $taskRoot = [IO.Path]::GetFullPath($RunDir)
    $taskMode = $NodeFeatureMode.ToLowerInvariant()
    $taskOutput = Join-Path (Join-Path $taskRoot 'outputs') $taskMode
    $taskFullHistory = ($taskMode -eq 'behavior9') -and (-not $LegacyTimeProtocol)
    Remove-Item Env:BHTG_EXPERIMENT_PROTOCOL -ErrorAction SilentlyContinue
    if ($taskFullHistory) {
        $env:BHTG_EXPERIMENT_PROTOCOL = 'unlabeled_phase2_full_history_phase3_v1'
        $taskOutput += '_unlabeled_full_history'
    }
    $env:PYTHONPATH = $env:BHTG_EXPERIMENT_DGL_RUNTIME
    $env:DGLBACKEND = 'pytorch'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    Remove-Item Env:BHTG_BUNDLE_DIR -ErrorAction SilentlyContinue
    $env:BHTG_USE_AGGREGATED_ACTIONS = '1'
    $env:BHTG_ACTIONS_MANIFEST = [IO.Path]::GetFullPath($ActionsManifest)
    $env:BHTG_SEMANTIC_AMOUNT_MODE = if ($Group -in @('B','D')) { 'continuous_zscore' } else { 'disabled' }
    $env:BHTG_MEMORY_AMOUNT_MODE = if ($Group -in @('C','D')) { 'continuous_zscore' } else { 'disabled' }
    $env:BHTG_FOUR_ACTIONS_GRAPH_DIR = [IO.Path]::GetFullPath($GraphDir)
    $env:BHTG_OUTPUT_DIR = $taskOutput
    $env:BHTG_NODE_FEATURE_MODE = $taskMode
    $env:BHTG_NODE_FEATURE_WINDOW_SECONDS = [string]$WindowSeconds
    $env:BHTG_NODE_FEATURE_MIN_HISTORY = [string]$MinHistory
    $env:PHASE2_TAG = 'amount_causal_v2'
    $env:PHASE3_TAG = 'amount_causal_v2'
    $env:PHASE2_EPOCHS = [string]$Phase2Epochs
    $env:PHASE2_MAX_BATCHES = [string]$Phase2MaxBatches
    $env:PHASE2_STRICT_TEMPORAL = '1'
    $env:PHASE3_ALL_TOKENS = '1'
    $env:PHASE3_EVAL_PROTOCOL = 'token_stratified'
    $env:PHASE3_STRICT_TEMPORAL = '1'
    $env:PHASE3_PERMUTE_LABELS = '0'
    Remove-Item Env:PHASE3_GRAPH_CKPT -ErrorAction SilentlyContinue
    $env:RELATION_BALANCED_SAMPLING = '0'
    $env:HOMO_ABLATION = '0'
    Invoke-BehaviorPython -Arguments @('-c', @'
import json
import os
import sys
from pathlib import Path
from bhtg import config
from bhtg.action_csv import require_graph_binding
from bhtg.node_feature_provider import feature_spec
require_graph_binding()
with open(config.GRAPH_MANIFEST_PATH, encoding='utf-8') as handle:
    graph_manifest = json.load(handle)
if graph_manifest.get('source_hashes', {}).get('actions_manifest') != config.ACTIONS_SELECTION['manifest']['sha256']:
    raise RuntimeError('graph provenance differs from the selected actions manifest')
protocol, names = feature_spec()
print(json.dumps({'graph': config.GRAPH_BIN, 'outputs': config.CHECKPOINT_DIR,
                  'actions_manifest': config.ACTIONS_SELECTION['manifest'],
                  'upstream_directory': str(Path(sys.argv[1]).resolve()),
                  'node_feature_mode': config.NODE_FEATURE_MODE,
                   'experiment_protocol': os.environ.get('BHTG_EXPERIMENT_PROTOCOL', 'causal_training_v2'),
                  'feature_protocol': protocol, 'input_dim': len(names), 'input_names': names,
                  'window_seconds': int(os.environ['BHTG_NODE_FEATURE_WINDOW_SECONDS'])
                                    if config.NODE_FEATURE_MODE == 'behavior9' else None,
                  'min_amount_history': int(os.environ['BHTG_NODE_FEATURE_MIN_HISTORY'])
                                        if config.NODE_FEATURE_MODE == 'behavior9' else None,
                  'amount_modes': [config.SEMANTIC_AMOUNT_MODE, config.MEMORY_AMOUNT_MODE]},
                 ensure_ascii=False, indent=2))
'@, $UpstreamDir)
    Write-Host "RunDir: $taskRoot"
    if ($Phase -eq 'Inspect') { Write-Host 'Inspection complete. No graph rebuild, file copy, or training started.'; return }
    New-Item -ItemType Directory -Path $taskOutput -Force | Out-Null
    if ($Phase -in @('All','Phase2')) {
        Assert-NewStage -Names @('phase2_graph_amount_causal_v2.pt','phase2_log_amount_causal_v2.txt')
    }
    if ($Phase -in @('All','Phase3') -and -not $ResumePhase3) {
        Assert-NewStage -Names @('phase3_mem_best_amount_causal_v2.pt','phase3_mem_latest_amount_causal_v2.pt',
            'phase3_mem_log_amount_causal_v2.txt','phase3_log_amount_causal_v2.txt',
            'phase3_mem_results_amount_causal_v2.json','phase3_split_amount_causal_v2.pt')
    }
    if ($Phase -in @('Reuse','All','Phase1')) {
        $taskSemanticReady = Use-CompatibleUpstream -Kind 'semantic' -Name 'h_sem_temporal_lmae_strict.pt'
        if (-not $taskSemanticReady -and $Phase -ne 'Reuse') {
            Assert-NewStage -Names @('phase1_lmae_log_strict.txt','phase1_sem_lmae_strict.pt')
            Invoke-BehaviorPython -Arguments @('run_phase1_lmae.py','--epochs',([string]$Phase1Epochs),'--batch','32')
        }
    }
    if ($Phase -in @('All','Phase2')) { Invoke-BehaviorPython -Arguments @('run_phase2.py') }
    if ($Phase -in @('Reuse','All','Memory')) {
        $taskMemoryReady = Use-CompatibleUpstream -Kind 'memory' -Name 'phase2_memory.pt'
        if (-not $taskMemoryReady -and $Phase -ne 'Reuse') {
            Assert-NewStage -Names @('phase2_memory.pt','phase2_memory_log.txt')
            Invoke-BehaviorPython -Arguments @('run_phase2_memory.py','--epochs',([string]$MemoryEpochs),'--batch','2048')
        }
    }
    if ($Phase -eq 'Reuse') {
        if (-not ($taskSemanticReady -and $taskMemoryReady)) {
            throw "Upstream reuse incomplete (Phase1=$taskSemanticReady, Memory=$taskMemoryReady). Run the missing stage in this RunDir."
        }
        Write-Host 'Phase1 and Memory artifacts are ready.'
        return
    }
    if ($Phase -in @('All','Phase3')) {
        $taskRequiredFiles = @('phase2_memory.pt','phase2_graph_amount_causal_v2.pt')
        if (-not $taskFullHistory) { $taskRequiredFiles += 'h_sem_temporal_lmae_strict.pt' }
        foreach ($taskRequired in $taskRequiredFiles) {
            if (-not (Test-Path -LiteralPath (Join-Path $taskOutput $taskRequired))) {
                throw "Missing prerequisite: $taskRequired. Run Reuse/Phase1/Phase2/Memory first."
            }
        }
        $taskSemanticPath = Join-Path $taskOutput 'h_sem_temporal_lmae_strict.pt'
        if ($taskFullHistory) {
            $taskSemanticPath = Join-Path $taskOutput 'h_sem_full_history.pt'
            if (-not (Test-Path -LiteralPath $taskSemanticPath)) {
                $taskEncoderName = 'phase1_sem_lmae_strict.pt'
                $taskEncoderPath = Join-Path $taskOutput $taskEncoderName
                if ($Phase1Checkpoint) {
                    $taskEncoderPath = [IO.Path]::GetFullPath($Phase1Checkpoint)
                } elseif (-not $FreshUpstream -and -not (Test-Path -LiteralPath $taskEncoderPath)) {
                    $taskEncoderPath = Join-Path $UpstreamDir $taskEncoderName
                    if (-not (Test-Path -LiteralPath $taskEncoderPath)) {
                        $taskEncoderPath = Join-Path $PSScriptRoot "experiments\pool_create_amount_joint_20261111\outputs\$taskEncoderName"
                    }
                }
                if (-not (Test-Path -LiteralPath $taskEncoderPath -PathType Leaf)) {
                    throw 'Missing Phase1 encoder weights for full-history export. Run Phase1 or pass -Phase1Checkpoint.'
                }
                Invoke-BehaviorPython -Arguments @('export_semantic_history.py','--checkpoint',$taskEncoderPath,
                    '--output-dir',$taskOutput,'--full-history-cutoff','auto')
            }
        }
        $taskArguments = @('run_phase3.py','--use-memory','--strict-temporal','--validation-threshold',
            '--eval-protocol','token_stratified',
            '--graph-path',(Join-Path $taskOutput 'phase2_graph_amount_causal_v2.pt'),
            '--sem-path',$taskSemanticPath,'--epochs',([string]$Phase3Epochs))
        if ($JointMemory) { $taskArguments += '--joint-memory' }
        if ($ResumePhase3) { $taskArguments += @('--resume-phase3',$ResumePhase3) }
        Invoke-BehaviorPython -Arguments $taskArguments
    }
} finally {
    foreach ($taskName in $taskNames) {
        [Environment]::SetEnvironmentVariable($taskName,$taskSaved[$taskName],'Process')
    }
    Pop-Location
}
