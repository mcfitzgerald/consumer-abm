import numpy as np


def generate_promo_calendar(promo_depths, promo_frequencies):
    """
    Generates a promotion calendar for 52 weeks.

    promo_depths: list of percentage discounts to be taken off base
    promo_frequencies: list of probabilities reflecting percentage of occasions depth will be applied

    Returns a dictionary where keys are week numbers and values are promo depths.
    """
    if len(promo_depths) != len(promo_frequencies):
        raise ValueError(
            "promo_depths and promo_frequencies must be of the same length"
        )

    if not np.isclose(sum(promo_frequencies), 1):
        raise ValueError("promo_frequencies must sum to 1")

    calendar = {}
    for week in range(1, 53):
        promo_depth = np.random.choice(promo_depths, p=promo_frequencies)
        calendar[week] = promo_depth

    return calendar
