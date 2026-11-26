"""Conservative visibility times for graph archives with rounded timestamps.

Float32 cannot represent individual seconds at contemporary Unix epochs.  Its
stored value therefore denotes a rounding interval, not an exact event time.
Only make such an event visible once the entire interval is before a cutoff.
Exact integer and float64 timestamps remain unchanged.
"""
import torch


TIME_PROTOCOL = 'strict_before_conservative_rounding_upper_bound_v1'


def conservative_timestamps(timestamps):
    """Return float64 upper bounds on potentially rounded event timestamps.

    Uses the larger adjacent spacing to also cover powers of two and negative
    values.  Existing float64 values are understood to be exact; callers should
    retain the original dtype until this conversion has been performed.
    """
    values = torch.as_tensor(timestamps)
    if not bool(torch.isfinite(values).all()):
        raise ValueError('timestamps must be finite')
    precise = values.to(dtype=torch.float64)
    if not values.is_floating_point() or values.dtype == torch.float64:
        return precise
    upper = torch.nextafter(values, torch.full_like(values, float('inf')))
    lower = torch.nextafter(values, torch.full_like(values, -float('inf')))
    spacing = torch.maximum(upper.to(torch.float64) - precise,
                            precise - lower.to(torch.float64))
    return precise + spacing / 2
