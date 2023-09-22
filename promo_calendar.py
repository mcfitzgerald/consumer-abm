import pandas as pd
from typing import Dict, List


def prepare_promo_schedule_variables(brand: str, config: Dict) -> tuple:
    """
    This function prepares the variables needed for the promo schedule.

    Args:
        brand (str): The brand for which the promo schedule is being prepared.
        config (Dict): The configuration dictionary containing all the necessary data.

    Returns:
        tuple: A tuple containing the base product price, promo depths, promo frequencies, and promo weeks.

    Raises:
        KeyError: If a necessary key is not found in the configuration.
        Exception: If an unexpected error occurs.
    """
    try:
        brand_info = config["brands"][brand]
        promo_info = brand_info["promotions"]
        base_product_price = brand_info["base_product_price"]

        promo_depths = promo_info["promo_depths"]
        promo_frequencies = promo_info["promo_frequencies"]
        promo_calendar = promo_info["promo_calendar"]
        promo_weeks = list(
            set(
                [
                    week
                    for campaign in promo_calendar
                    for week in config["ad_campaigns"][campaign]
                ]
            )
        )

        return base_product_price, promo_depths, promo_frequencies, promo_weeks
    except KeyError as e:
        print(f"KeyError: {e} is not found in the configuration.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def generate_brand_promo_schedule(
    base_product_price: float,
    promo_depths: Dict[str, float],
    promo_frequencies: Dict[str, float],
    promo_weeks: List[int],
) -> pd.DataFrame:
    """
    This function generates a brand's promotional schedule.

    Args:
        base_product_price (float): The base price of the product.
        promo_depths (Dict[str, float]): A dictionary mapping promo names to their depths.
        promo_frequencies (Dict[str, float]): A dictionary mapping promo names to their frequencies.
        promo_weeks (List[int]): A list of weeks when promotions are scheduled.

    Returns:
        pd.DataFrame: A DataFrame representing the promotional schedule, with weeks as the index and prices as the values.

    Raises:
        Exception: If an unexpected error occurs.
    """
    try:
        df = pd.DataFrame(index=pd.RangeIndex(1, 53), columns=["price"])

        # Calculate the number of weeks for each promo depth based on their frequencies
        weeks_per_promo = [round(freq * 52) for freq in promo_frequencies]

        # Create a list of prices based on promo depths and their frequencies
        prices = []
        for depth, weeks in zip(promo_depths, weeks_per_promo):
            prices.extend([base_product_price * (1 - depth)] * weeks)

        # Sort the prices in ascending order
        prices.sort()

        # Assign the lowest prices to the weeks indicated in promo_weeks
        for week in promo_weeks:
            df.loc[week, "price"] = prices.pop(0)

        # Distribute the remaining prices in even blocks across remaining weeks
        remaining_weeks = df[df["price"].isna()].index.tolist()
        block_size = len(remaining_weeks) // len(prices)

        for i in range(0, len(remaining_weeks), block_size):
            for week in remaining_weeks[i : i + block_size]:
                df.loc[week, "price"] = prices.pop(0)

        # If there are any remaining prices, assign them to the remaining weeks
        for week in remaining_weeks[i + block_size :]:
            df.loc[week, "price"] = prices.pop(0)

        return df
    except Exception as e:
        print(f"An unexpected error occurred in generate_brand_promo_schedule: {e}")
        return pd.DataFrame()
