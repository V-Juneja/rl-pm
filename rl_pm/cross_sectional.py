import numpy as np


RANK_FEATURES = (
    (2, 16, "rank_log_return_21"),
    (3, 17, "rank_log_return_63"),
    (4, 18, "rank_log_return_252"),
    (7, 21, "rank_volatility_21"),
    (9, 23, "rank_downside_volatility_21"),
)


def extra_feature_count(feature_count):
    return len(RANK_FEATURES) if feature_count >= 24 else 0


def augment_features(values, eligible, held=None):
    values = np.asarray(values, dtype=np.float32)
    eligible = np.asarray(eligible, dtype=bool)

    if values.ndim != 2 or eligible.shape != (values.shape[0],):
        raise ValueError("Expected one date of N×F features and an N mask")

    count = extra_feature_count(values.shape[1])
    if not count:
        return values

    comparison = eligible.copy()
    if held is not None:
        held = np.asarray(held, dtype=bool)
        if held.shape != comparison.shape:
            raise ValueError("Held mask shape mismatch")
        comparison |= held

    ranks = np.full((len(values), count), 0.5, dtype=np.float32)
    for rank_column, (column, availability_column, _) in enumerate(RANK_FEATURES):
        valid = comparison & (values[:, availability_column] > 0)
        indices = np.flatnonzero(valid)
        if len(indices) < 2:
            continue

        order = np.argsort(values[indices, column], kind="stable")
        sorted_indices = indices[order]
        sorted_values = values[sorted_indices, column]
        _, starts, counts = np.unique(
            sorted_values, return_index=True, return_counts=True,
        )
        average_ranks = (starts + (counts - 1) / 2) / (len(indices) - 1)
        ranks[sorted_indices, rank_column] = np.repeat(average_ranks, counts)

    return np.concatenate([values, ranks], axis=1)
