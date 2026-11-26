"""Explicit v3 relations and causal pool feature contract; legacy stays four."""
LEGACY_EDGE_TYPES = ("transfer", "mint", "burn", "create")
V3_EDGE_TYPES = LEGACY_EDGE_TYPES + ("lp_mint", "lp_burn", "lp_swap", "lp_sync")
POOL_EDGE_FEATURE_FIELDS = ("lp_drain0", "lp_drain1", "lp_drain_max",
                           "lp_drain_known", "lp_reserve_known")

def relation_schema(names):
    names = tuple(names)
    if names not in (LEGACY_EDGE_TYPES, V3_EDGE_TYPES):
        raise ValueError("unregistered or reordered model relation schema")
    return names
