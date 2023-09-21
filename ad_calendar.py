import pandas as pd
import numpy as np
from typing import Dict, List, Callable


def prepare_ad_schedule_variables(brand: str, config: Dict) -> tuple:
    """config is a dictionary that contains the information for all brands and their
    advertising campaigns, it should be read from a toml file prior to calling function
    """
    # Extract the relevant information for the specified brand
    brand_info = config["brands"][brand]
    ad_info = brand_info["advertising"]

    # Prepare the variables for the generate_brand_ad_schedule function
    budget = ad_info["budget"]
    media_channels = ad_info["channels"]
    priority = ad_info["priority"]
    schedule = {
        channel: config["ad_campaigns"][campaign]
        for channel, campaign in ad_info["ad_campaigns"].items()
    }

    return budget, media_channels, priority, schedule


def generate_brand_ad_schedule(
    budget: int,
    media_channels: List[str],
    priority: Dict[str, int],
    schedule: Dict[str, List[int]],
    spending_strategy: Callable[[int], List[float]] = None,
) -> pd.DataFrame:
    """
    Generate a marketing ad schedule.

    Parameters:
    budget (int): The total marketing budget.
    media_channels (List[str]): A list of media channels.
    priority (Dict[str, int]): A dictionary where the keys are the media channels and the values are the priority of each channel.
    schedule (Dict[str, List[int]]): A dictionary where the keys are the media channels and the values are lists of weeks during which the campaigns are run.
    spending_strategy (Callable[[int], List[float]]): A function that takes the number of weeks and returns a list of proportions that represent the spending distribution across the weeks.

    Returns:
    pd.DataFrame: A DataFrame that represents the ad schedule.
    """
    # Ensure the priority values sum to 100
    assert sum(priority.values()) == 100, "Priority values must sum to 100"

    # Create a DataFrame with 52 weeks and the media channels as columns
    df = pd.DataFrame(index=pd.RangeIndex(1, 53), columns=media_channels)

    # Initialize the DataFrame with zeros
    df = df.fillna(0)

    # Distribute the budget across the media channels based on their priority and schedule
    for channel in media_channels:
        if len(schedule[channel]) == 0:
            print(f"Warning: No schedule defined for {channel}. Skipping this channel.")
            continue
        if spending_strategy is None:
            spending_distribution = [1 / len(schedule[channel])] * len(
                schedule[channel]
            )
        else:
            spending_distribution = spending_strategy(len(schedule[channel]))
        for week, proportion in zip(schedule[channel], spending_distribution):
            df.loc[week, channel] = budget * (priority[channel] / 100) * proportion

    # Replace NaN values with 0
    df = df.replace(np.nan, 0)

    return df


# def generate_brand_ad_schedule(
#     budget: int,
#     media_channels: List[str],
#     priority: Dict[str, int],
#     schedule: Dict[str, List[int]],
# ) -> pd.DataFrame:
#     """
#     Generate a marketing ad schedule.

#     Parameters:
#     budget (int): The total marketing budget.
#     media_channels (List[str]): A list of media channels.
#     priority (Dict[str, int]): A dictionary where the keys are the media channels and the values are the priority of each channel.
#     schedule (Dict[str, List[int]]): A dictionary where the keys are the media channels and the values are lists of weeks during which the campaigns are run.

#     Returns:
#     pd.DataFrame: A DataFrame that represents the ad schedule.
#     """
#     # Ensure the priority values sum to 100
#     assert sum(priority.values()) == 100, "Priority values must sum to 100"

#     # Create a DataFrame with 52 weeks and the media channels as columns
#     df = pd.DataFrame(index=pd.RangeIndex(1, 53), columns=media_channels)

#     # Initialize the DataFrame with zeros
#     df = df.fillna(0)

#     # Distribute the budget across the media channels based on their priority and schedule
#     for channel in media_channels:
#         if len(schedule[channel]) == 0:
#             print(f"Warning: No schedule defined for {channel}. Skipping this channel.")
#             continue
#         df.loc[schedule[channel], channel] = (
#             budget * (priority[channel] / 100) / len(schedule[channel])
#         )

#     # Replace NaN values with 0
#     df = df.replace(np.nan, 0)

#     return df
