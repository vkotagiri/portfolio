def exact_or_na(val):
    return val if val is not None else "Data not available"

def reconcile_total(mkt_values: list[float], total: float) -> bool:
    if total == 0: return False
    s = sum(mkt_values)
    return abs(s - total) <= max(0.001 * total, 0.01)
