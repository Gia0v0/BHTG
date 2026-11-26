"""Explicit saved-zscore edge messages for the four-actions graph encoder."""
import torch


GRAPH_AMOUNT_EDGE_FIELDS = ('amount_z', 'amount_known')
GRAPH_AMOUNT_PROTOCOL = 'four_actions_graph_saved_zscore_known_v1'


def graph_amount_fields(memory_amount_mode):
    """The graph and Memory amount ablation share the same explicit switch."""
    if memory_amount_mode not in ('legacy', 'disabled', 'continuous_zscore'):
        raise ValueError('unsupported graph/Memory amount mode')
    return GRAPH_AMOUNT_EDGE_FIELDS if memory_amount_mode == 'continuous_zscore' else ()


def amount_edge_features(edge_data, count, device):
    """Return [z, known] with missing amounts masked, without treating z as volume.

    DGL can deserialize boolean fields as uint8. Boolean and numeric 0/1 masks
    are accepted; other flags, missing fields, and float32 overflow fail closed.
    A known zero is [0, 1], distinct from an unknown amount's [0, 0].
    """
    z, known = (edge_data.get(field) for field in GRAPH_AMOUNT_EDGE_FIELDS)
    if (not isinstance(z, torch.Tensor) or z.ndim != 1 or len(z) != count or
            not z.is_floating_point()):
        raise ValueError('graph amount_z must be a floating tensor of shape [E]')
    if (not isinstance(known, torch.Tensor) or known.ndim != 1 or len(known) != count or
            known.is_complex()):
        raise ValueError('graph amount_known must be a boolean or 0/1 tensor of shape [E]')
    z = z.to(device=device, dtype=torch.float32)
    known = known.to(device=device)
    if not bool(torch.isfinite(z).all()):
        raise ValueError('graph amount_z must be finite float32 saved z-scores')
    if (not bool(torch.isfinite(known).all()) or
            bool(((known != 0) & (known != 1)).any())):
        raise ValueError('graph amount_known flags must be boolean or numeric 0/1')
    mask = known.bool()
    return torch.stack((torch.where(mask, z, torch.zeros_like(z)), mask.float()), dim=1)


def graph_amount_metadata(memory_amount_mode):
    fields = graph_amount_fields(memory_amount_mode)
    return {'protocol': GRAPH_AMOUNT_PROTOCOL, 'enabled': bool(fields),
            'fields': list(fields), 'architecture_version': 4 if fields else 2,
            'message': '[z,1] if known; [0,0] if missing',
            'amount_semantics': 'saved_standardized_zscore_not_raw_volume',
            'switch': 'memory_amount_mode_continuous_zscore'}
