import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Optional, Tuple


def prepare_ad_schedule_variables(
    brand: str, config: Dict[str, Any]
) -> Optional[Tuple[int, List[str], Dict[str, int], Dict[str, List[int]]]]:
    """
    Prepare the variables needed for the generate_brand_ad_schedule function.

    Parameters:
    brand (str): The brand for which the ad schedule is to be prepared.
    config (Dict[str, Any]): A dictionary that contains the information for all brands and their advertising campaigns.
                             It should be read from a toml file prior to calling this function.

    Returns:
    Tuple[int, List[str], Dict[str, int], Dict[str, List[int]]]: A tuple containing the budget, media channels, priority, and schedule for the specified brand.
                                                                 Returns None if a KeyError or any other exception occurs.
    """
    try:
        # Extract the relevant information for the specified brand from the config dictionary
        brand_info = config["brands"][brand]
        ad_info = brand_info["advertising"]

        # Extract the budget, media channels, and priority for the specified brand
        budget = ad_info["budget"]
        media_channels = ad_info["channels"]
        priority = ad_info["priority"]

        # Construct the schedule dictionary where the keys are the media channels and the values are the corresponding ad campaigns
        schedule = {
            channel: config["campaign_library"][campaign]
            for channel, campaign in ad_info["ad_campaigns"].items()
        }

        # Return the prepared variables as a tuple
        return budget, media_channels, priority, schedule
    except KeyError as e:
        print(f"KeyError: {e} is not found in the config.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def generate_brand_ad_schedule(
    budget: int,
    media_channels: List[str],
    priority: Dict[str, int],
    schedule: Dict[str, List[int]],
    spending_strategy: Optional[Callable[[int], List[float]]] = None,
) -> Optional[pd.DataFrame]:
    """
    Generate a marketing ad schedule.

    This function creates a DataFrame that represents the ad schedule for a brand.
    The ad schedule is created based on the budget, media channels, priority, schedule, and spending strategy.

    Parameters:
    budget (int): The total marketing budget.
    media_channels (List[str]): A list of media channels.
    priority (Dict[str, int]): A dictionary where the keys are the media channels and the values are the priority of each channel.
    schedule (Dict[str, List[int]]): A dictionary where the keys are the media channels and the values are lists of weeks during which the campaigns are run.
    spending_strategy (Callable[[int], List[float]], optional): A function that takes the number of weeks and returns a list of proportions that represent the spending distribution across the weeks. Defaults to None.

    Returns:
    pd.DataFrame: A DataFrame that represents the ad schedule. Returns None if an error occurs.
    """
    try:
        # Check if the sum of priority values is 100
        assert sum(priority.values()) == 100, "Priority values must sum to 100"

        # Create a DataFrame with 52 weeks as index and the media channels as columns
        df = pd.DataFrame(index=pd.RangeIndex(1, 53), columns=media_channels)

        # Fill the DataFrame with zeros
        df = df.fillna(0)

        # Loop over each media channel
        for channel in media_channels:
            # Skip the channel if no schedule is defined
            if len(schedule[channel]) == 0:
                print(
                    f"Warning: No schedule defined for {channel}. Skipping this channel."
                )
                continue

            # If no spending strategy is provided, distribute the budget evenly across the weeks
            if spending_strategy is None:
                spending_distribution = [1 / len(schedule[channel])] * len(
                    schedule[channel]
                )
            else:
                spending_distribution = spending_strategy(len(schedule[channel]))

            # Distribute the budget across the weeks based on the spending distribution
            for week, proportion in zip(schedule[channel], spending_distribution[:-1]):
                df.loc[week, channel] = round(
                    budget * (priority[channel] / 100) * proportion, 2
                )

            # Adjust the budget for the last week to ensure the total budget for the channel is correct
            df.loc[schedule[channel][-1], channel] = (
                budget * (priority[channel] / 100) - df[channel].sum()
            )

        # Replace any remaining NaN values with 0
        df = df.replace(np.nan, 0)

        return df
    except AssertionError as e:
        print(f"AssertionError: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
