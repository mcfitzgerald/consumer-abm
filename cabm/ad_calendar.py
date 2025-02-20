# cabm/ad_calendar.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Optional, Tuple


def prepare_ad_schedule_variables(
    brand: str, config: Dict[str, Any]
) -> Optional[Tuple[float, List[str], Dict[str, int], Dict[str, List[int]]]]:
    """
    Prepare the variables needed for the generate_brand_ad_schedule function
    from the config. For a brand, read:
      - total budget
      - channels
      - channel priorities
      - campaign schedule
    """
    try:
        brand_info = config["brands"][brand]
        ad_info = brand_info["advertising"]

        budget = ad_info["budget"]
        media_channels = ad_info["channels"]
        priority = ad_info["priority"]

        # Map channel->week list from campaign_library
        schedule = {
            channel: config["campaign_library"][campaign]
            for channel, campaign in ad_info["ad_campaigns"].items()
        }

        return budget, media_channels, priority, schedule
    except KeyError as e:
        print(f"KeyError: {e} not found in the config.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def generate_brand_ad_schedule(
    budget: float,
    media_channels: List[str],
    priority: Dict[str, int],
    schedule: Dict[str, List[int]],
    spending_strategy: Optional[Callable[[int], List[float]]] = None,
) -> Optional[pd.DataFrame]:
    """
    Generate a marketing ad schedule for a brand across 52 weeks, distributing
    the brand's budget by channel, week, etc. Returns a DataFrame with columns
    for each media channel, index=1..52 weeks.
    """
    try:
        assert sum(priority.values()) == 100, "Priority values must sum to 100"
        df = pd.DataFrame(index=pd.RangeIndex(1, 53), columns=media_channels)
        df = df.fillna(0.0)

        for channel in media_channels:
            if channel not in schedule or len(schedule[channel]) == 0:
                continue

            if spending_strategy is None:
                # evenly
                n = len(schedule[channel])
                spending_distribution = [1 / n] * n
            else:
                spending_distribution = spending_strategy(len(schedule[channel]))

            # sum-of-priority for channel
            channel_budget = budget * (priority[channel] / 100.0)

            # fill each scheduled week
            for idx, week in enumerate(schedule[channel]):
                if idx < len(spending_distribution) - 1:
                    df.loc[week, channel] = round(
                        channel_budget * spending_distribution[idx], 2
                    )
                else:
                    # final leftover to ensure sum matches
                    spent_so_far = df[channel].sum()
                    df.loc[week, channel] = round(channel_budget - spent_so_far, 2)

        return df
    except AssertionError as e:
        print(f"AssertionError: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
