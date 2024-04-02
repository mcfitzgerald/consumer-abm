import pandas as pd
from typing import Dict, List, Any, Tuple


def prepare_promo_schedule_variables(
    brand: str, config: Dict[str, Any]
) -> Tuple[float, Dict[int, float]]:
    """
    This function prepares the variables needed for the promo schedule.

    Args:
        brand (str): The brand for which the promo schedule is being prepared.
        config (Dict[str, Any]): The configuration dictionary containing all the necessary data, read from a toml file.

    Returns:
        Tuple[float, Dict[int, float]]: A tuple containing the base product price and a dictionary mapping week numbers to discounted prices.

    Raises:
        KeyError: If a necessary key is not found in the configuration.
        Exception: If an unexpected error occurs.
    """
    try:
        # Extract brand and promotion information from the configuration
        brand_info = config["brands"][brand]
        promo_info = brand_info["promotions"]

        # Extract base product price
        base_product_price = brand_info["base_product_price"]

        # Extract promo calendar and initialize promo weeks dict
        promo_calendar_config = promo_info["promo_calendar"]
        promo_calendar = {}

        # For each campaign in the promo calendar, check if it exists in the configuration
        # If it does, update the promo calendar dict with the weeks of the campaign and the discounted price
        for campaign, depth in promo_calendar_config.items():
            if campaign not in config["campaign_library"]:
                raise KeyError(f"{campaign} is not found in the configuration.")
            for week in config["campaign_library"][campaign]:
                discounted_price = base_product_price * (1 - depth)
                # If the week already exists in the promo calendar and the new discounted price is lower, update it
                if (
                    week not in promo_calendar
                    or discounted_price < promo_calendar[week]
                ):
                    promo_calendar[week] = discounted_price

        return base_product_price, promo_calendar
    except KeyError as e:
        print(f"KeyError: {e} is not found in the configuration.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


# def generate_brand_promo_schedule(
#     base_product_price: float,
#     promo_depths: Dict[str, float],
#     promo_frequencies: Dict[str, float],
#     promo_weeks: List[int],
# ) -> pd.DataFrame:
#     """
#     Generates a brand's promotional schedule.

#     Args:
#         base_product_price (float): The base price of the product.
#         promo_depths (Dict[str, float]): A dictionary mapping promo names to their depths.
#         promo_frequencies (Dict[str, float]): A dictionary mapping promo names to their frequencies.
#         promo_weeks (List[int]): A list of weeks when promotions are scheduled.

#     Returns:
#         pd.DataFrame: A DataFrame representing the promotional schedule, with weeks as the index and prices as the values.

#     Raises:
#         Exception: If an unexpected error occurs.
#     """
#     try:
#         # Initialize a DataFrame with 52 weeks and a 'price' column
#         df = pd.DataFrame(index=pd.RangeIndex(1, 53), columns=["price"])

#         # Calculate the number of weeks for each promo depth based on their frequencies
#         weeks_per_promo = [round(freq * 52) for freq in promo_frequencies]

#         # Create a list of prices based on promo depths and their frequencies
#         prices = []
#         for depth, weeks in zip(promo_depths, weeks_per_promo):
#             # Extend the prices list with the discounted price for the number of weeks
#             prices.extend([base_product_price * (1 - depth)] * weeks)

#         # Sort the prices in ascending order
#         prices.sort()

#         # Assign the lowest prices to the weeks indicated in promo_weeks
#         for week in promo_weeks:
#             df.loc[week, "price"] = prices.pop(0)

#         # Get the list of remaining weeks where price is not assigned
#         remaining_weeks = df[df["price"].isna()].index.tolist()
#         # Calculate the block size for distributing the remaining prices
#         block_size = len(remaining_weeks) // len(prices)

#         # Distribute the remaining prices in even blocks across remaining weeks
#         for i in range(0, len(remaining_weeks), block_size):
#             for week in remaining_weeks[i : i + block_size]:
#                 df.loc[week, "price"] = prices.pop(0)

#         # If there are any remaining prices, assign them to the remaining weeks
#         for week in remaining_weeks[i + block_size :]:
#             df.loc[week, "price"] = prices.pop(0)

#         return df
#     except Exception as e:
#         print(f"An unexpected error occurred in generate_brand_promo_schedule: {e}")
#         # Return an empty DataFrame in case of an error
#         return pd.DataFrame()


def generate_brand_promo_schedule(
    base_product_price: float,
    promo_calendar: Dict[int, float],
) -> pd.DataFrame:
    """
    Generates a brand's promotional schedule.

    Args:
        base_product_price (float): The base price of the product without any promotions.
        promo_calendar (Dict[int, float]): A dictionary mapping weeks to their discounted prices.

    Returns:
        pd.DataFrame: A DataFrame representing the promotional schedule, with weeks as the index and prices as the values.
    """

    # Initialize a DataFrame with 52 weeks and a 'price' column
    df = pd.DataFrame(index=pd.RangeIndex(1, 53), columns=["price"])

    # Populate the DataFrame
    for week in df.index:
        if week in promo_calendar:
            # If there is a promo this week, assign the discounted price
            df.loc[week, "price"] = promo_calendar[week]
        else:
            # If there is no promo this week, use the base product price
            df.loc[week, "price"] = base_product_price

    return df
