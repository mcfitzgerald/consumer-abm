import pandas as pd
import numpy as np


def generate_campaign_schedule(
    media_channel_id: str,
    budget: float,
    campaign_duration: int,
    campaign_frequency: int,
):
    # Calculate total number of campaign weeks
    total_campaign_weeks = campaign_duration * campaign_frequency

    # Calculate weekly budget
    weekly_budget = budget / total_campaign_weeks

    # Calculate the number of weeks between each campaign
    weeks_between_campaigns = 52 // campaign_frequency

    # Create a list of weeks when each campaign starts
    campaign_start_weeks = list(range(1, 53, weeks_between_campaigns))

    # Initialize a DataFrame to store the campaign schedule
    campaign_schedule = pd.DataFrame(index=range(1, 53), columns=[media_channel_id])
    campaign_schedule = campaign_schedule.fillna(0)

    # Distribute the budget across the campaigns
    for start_week in campaign_start_weeks:
        end_week = start_week + campaign_duration - 1
        if end_week > 52:
            end_week = 52
        campaign_schedule.loc[start_week:end_week, media_channel_id] = weekly_budget

    return campaign_schedule


def distribute_ad_budget(
    total_ad_budget: float, media_channels: list[str], channel_proportions: list[float]
) -> dict:
    """
    Distributes the total advertising budget across media channels.

    total_ad_budget: float representing total advertising budget
    media_channels: list of media channels
    channel_proportions: list of proportions of budget to be spent on each media channel

    Returns a dictionary where keys are media channels and values are the budget to be spent on each channel.
    """
    if len(media_channels) != len(channel_proportions):
        raise ValueError(
            "media_channels and channel_proportions must be of the same length"
        )

    if not np.isclose(sum(channel_proportions), 1):
        raise ValueError("channel_proportions must sum to 1")

    ad_budget = {}
    for channel, proportion in zip(media_channels, channel_proportions):
        channel_budget = total_ad_budget * proportion
        ad_budget[channel] = channel_budget

    return ad_budget


def generate_multi_channel_schedule(
    total_ad_budget: float,
    media_channels: list[str],
    channel_proportions: list[float],
    campaign_durations: list[int],
    campaign_frequencies: list[int],
) -> pd.DataFrame:
    """
    Generates a campaign schedule for multiple channels.

    total_ad_budget: float representing total advertising budget
    media_channels: list of media channels
    channel_proportions: list of proportions of budget to be spent on each media channel
    campaign_durations: list of campaign durations for each media channel
    campaign_frequencies: list of campaign frequencies for each media channel

    Returns a DataFrame representing the campaign schedule for each channel.
    """
    if not all(
        len(lst) == len(media_channels)
        for lst in [channel_proportions, campaign_durations, campaign_frequencies]
    ):
        raise ValueError("All input lists must be of the same length as media_channels")

    # Distribute the total ad budget across the channels
    ad_budgets = distribute_ad_budget(
        total_ad_budget, media_channels, channel_proportions
    )

    # Initialize an empty DataFrame to store the multi-channel campaign schedule
    multi_channel_schedule = pd.DataFrame()

    # Generate the campaign schedule for each channel
    for channel in media_channels:
        channel_budget = ad_budgets[channel]
        campaign_duration = campaign_durations[media_channels.index(channel)]
        campaign_frequency = campaign_frequencies[media_channels.index(channel)]
        channel_schedule = generate_campaign_schedule(
            channel, channel_budget, campaign_duration, campaign_frequency
        )
        multi_channel_schedule = pd.concat(
            [multi_channel_schedule, channel_schedule], axis=1
        )

    return multi_channel_schedule
