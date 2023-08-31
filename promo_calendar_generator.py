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


def generate_ad_calendar(
    ad_spend: float, ad_block: int, ad_channels: list, ad_frequencies: list
):
    """
    Generates an advertising calendar for 52 weeks that distributes budget (ad_spend) across channels and blocks of advertising

    ad_spend: float representing total advertising budget, to be distributed across channels and blocks
    ad_block: int representing duration of campaign in weeks
    ad_channels: list of channels to where ads will be run
    ad_frequencies: proportion of budget to be spent on each channel, total spends per channel will reflect this proportion

    Returns a dictionary where keys are week numbers value is a dict of channels and spend per channel
    """
    if len(ad_channels) != len(ad_frequencies):
        raise ValueError("ad_channels and ad_frequencies must be of the same length")

    if not np.isclose(sum(ad_frequencies), 1):
        raise ValueError("ad_frequencies must sum to 1")

    calendar = {}
    total_spend = 0
    for week in range(1, 53):
        if week % ad_block == 0 and total_spend < ad_spend:
            week_spend = {}
            for channel, frequency in zip(ad_channels, ad_frequencies):
                channel_spend = ad_spend * frequency / (52 / ad_block)
                total_spend += channel_spend
                if total_spend > ad_spend:
                    channel_spend -= total_spend - ad_spend
                    total_spend = ad_spend
                week_spend[channel] = channel_spend
            calendar[week] = week_spend

    return calendar
