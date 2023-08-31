import pandas as pd
import numpy as np
from typing import Dict, List


def generate_ad_schedule(
    budget: int,
    media_channels: List[str],
    priority: Dict[str, int],
    schedule: Dict[str, List[int]],
) -> pd.DataFrame:
    """
    Generate a marketing ad schedule.

    Parameters:
    budget (int): The total marketing budget.
    media_channels (List[str]): A list of media channels.
    priority (Dict[str, int]): A dictionary where the keys are the media channels and the values are the priority of each channel.
    schedule (Dict[str, List[int]]): A dictionary where the keys are the media channels and the values are lists of weeks during which the campaigns are run.

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
        df.loc[schedule[channel], channel] = (
            budget * (priority[channel] / 100) / len(schedule[channel])
        )

    # Replace NaN values with 0
    df = df.replace(np.nan, 0)

    return df


# Define the budget and the media channels
budget = 52000
media_channels = ["TV", "Radio", "Social Media", "Print"]

# Define the priority of each media channel
priority = {"TV": 40, "Radio": 30, "Social Media": 20, "Print": 10}

# Define the schedule for each media channel
schedule = {
    "TV": [1, 2, 13, 14, 26, 27, 39, 40],
    "Radio": [5, 6, 7],
    "Social Media": [5, 32, 33, 45],
    "Print": [5, 12, 15],
}

try:
    df = generate_ad_schedule(budget, media_channels, priority, schedule)
    print(df)
except Exception as e:
    print(f"An error occurred: {e}")

try:
    df.sum().sum() == budget
    print("The budget is fully allocated.")
except Exception as e:
    print(f"An error occurred: {e}")
