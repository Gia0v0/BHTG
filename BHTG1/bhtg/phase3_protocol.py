"""Observation-based supervised cohorts and explicitly separate metrics."""
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, recall_score, roc_auc_score,
)


OBSERVATION_PROTOCOL = 'existing_token_active_history_v1'
FIXED_THRESHOLD = 0.5
FIXED_THRESHOLD_POLICY = 'fixed_0.5'
VALIDATION_THRESHOLD_POLICY = 'validation_active_history_max_bac_v1'


def partition_observed_splits(raw_split, observation_index, cutoffs, seed):
    """Keep each token's original assignment; never redistribute excluded IDs.

    Only active-history tokens enter supervised training and the primary
    validation/test metrics. Legitimate cold starts have their own cohort.
    Query-time eligibility and ordering do not inspect class labels.
    """
    split, cold_start, excluded, records, summaries = {}, {}, {}, {}, {}
    assigned = set()
    for offset, name in enumerate(('train', 'val', 'test')):
        ids = torch.as_tensor(raw_split[name], dtype=torch.long).cpu()
        values = ids.tolist()
        if len(values) != len(set(values)) or assigned.intersection(values):
            raise ValueError('Supervised token assignments must be unique and disjoint')
        assigned.update(values)
        rows = observation_index.query('contract', values, cutoffs[name])
        if [row['node_id'] for row in rows] != values:
            raise ValueError('Observation records are not aligned with requested token IDs')
        groups = {'active_history': [], 'cold_start': [], 'excluded': []}
        for row in rows:
            group = row['group'] if row['eligible'] else 'excluded'
            if group not in groups:
                raise ValueError('Unknown observation group: %s' % group)
            groups[group].append(int(row['node_id']))
        # Avoid class-concatenated evaluation batches even when the caller's
        # initial stratification concatenates classes. No label enters this RNG.
        rng = np.random.RandomState(int(seed) + 1009 * (offset + 1))
        for group in groups.values():
            rng.shuffle(group)
        split[name] = torch.tensor(groups['active_history'], dtype=torch.long)
        cold_start[name] = torch.tensor(groups['cold_start'], dtype=torch.long)
        excluded[name] = torch.tensor(groups['excluded'], dtype=torch.long)
        records[name] = rows
        summaries[name] = {
            'candidates': len(values),
            'active_history': len(groups['active_history']),
            'cold_start': len(groups['cold_start']),
            'excluded': len(groups['excluded']),
            'excluded_reasons': dict(Counter(row['reason'] for row in rows
                                             if not row['eligible'])),
            'existence_sources': dict(Counter(row['provenance'] for row in rows)),
        }
    return {
        'split': split, 'cold_start': cold_start, 'excluded': excluded,
        'observation_records': records,
        'observation_meta': {
            'protocol': OBSERVATION_PROTOCOL,
            'cutoffs': {name: float(value) for name, value in cutoffs.items()},
            'primary_cohort': 'active_history',
            'cold_start_policy': 'separate_report_excluded_from_training_and_selection',
            'ordering': 'seeded_label_independent_permutation',
            'summaries': summaries,
        },
    }


def _classification_inputs(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(labels) != len(probabilities):
        raise ValueError('Prediction count differs from the requested cohort')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Evaluation requires binary labels')
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Evaluation requires finite probabilities in [0, 1]')
    return labels, probabilities


def select_validation_threshold(labels, probabilities):
    """Maximize validation BAC, favoring the threshold nearest 0.5 then the lower one."""
    labels, probabilities = _classification_inputs(labels, probabilities)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError('Threshold selection requires both validation classes')
    order = np.argsort(probabilities)
    scores = probabilities[order]
    # With a >= decision rule, the lower end of an interval is the first
    # representable float strictly above the preceding score.
    above = np.nextafter(scores, np.inf)
    candidates = np.unique(np.r_[FIXED_THRESHOLD, scores, above[above <= 1.0]])
    boundaries = np.searchsorted(scores, candidates, side='left')
    cumulative_positive = np.r_[0, labels[order].cumsum()]
    false_negative = cumulative_positive[boundaries]
    true_positive = positives - false_negative
    true_negative = boundaries - false_negative
    # BAC has a common denominator, so compare exact integer numerators.
    bac_numerator = true_positive * negatives + true_negative * positives
    best = candidates[bac_numerator == bac_numerator.max()]
    return float(best[np.argmin(np.abs(best - FIXED_THRESHOLD))])


def classification_metrics(labels, probabilities, *, bac_fn=balanced_accuracy_score,
                           threshold=FIXED_THRESHOLD):
    """Report coverage and undefined metrics honestly for empty/one-class groups."""
    labels, probabilities = _classification_inputs(labels, probabilities)
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Classification threshold must be finite and in [0, 1]')
    result = {
        'count': len(labels), 'predicted': len(probabilities), 'missing_predictions': 0,
        'benign': int((labels == 0).sum()), 'rug': int((labels == 1).sum()),
        'bac': None, 'f1': None, 'ap': None, 'auc': None, 'recall': None,
        'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0,
    }
    if not len(labels):
        result['metric_status'] = 'empty_cohort'
        return result
    predicted = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    result.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    if len(np.unique(labels)) < 2:
        result['metric_status'] = 'single_class_cohort'
        return result
    result.update(
        bac=float(bac_fn(labels, predicted)),
        f1=float(f1_score(labels, predicted, zero_division=0)),
        ap=float(average_precision_score(labels, probabilities)),
        auc=float(roc_auc_score(labels, probabilities)),
        recall=float(recall_score(labels, predicted, zero_division=0)),
        metric_status='both_classes',
    )
    return result
