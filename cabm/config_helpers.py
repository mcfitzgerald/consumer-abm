import pandas as pd
from typing import Dict, List
from ad_calendar import (
    prepare_ad_schedule_variables,
    generate_brand_ad_schedule,
)
from promo_calendar import (
    prepare_promo_schedule_variables,
    generate_brand_promo_schedule,
)


# Ad and Promo Schedule


def generate_joint_ad_promo_schedule(brands: List[str], config: Dict) -> pd.DataFrame:
    """
    This function generates a joint ad and promo schedule for a list of brands.

    Parameters:
    brands (List[str]): A list of brand names.
    config (Dict): A dictionary containing configuration parameters.

    Returns:
    pd.DataFrame: A DataFrame containing the joint ad and promo schedule for the brands.
    """
    joint_schedule = pd.DataFrame()

    for brand in brands:
        try:
            # Prepare variables for promo schedule
            (
                base_product_price,
                promo_depths,
                promo_frequencies,
                promo_weeks,
            ) = prepare_promo_schedule_variables(brand, config)

            # Generate promo schedule
            promo_schedule = generate_brand_promo_schedule(
                base_product_price, promo_depths, promo_frequencies, promo_weeks
            )

            # Prepare variables for ad schedule
            (
                budget,
                media_channels,
                priority,
                schedule,
            ) = prepare_ad_schedule_variables(brand, config)

            # Generate ad schedule
            ad_schedule = generate_brand_ad_schedule(
                budget, media_channels, priority, schedule
            )

            # Merge promo and ad schedules
            brand_schedule = pd.concat([promo_schedule, ad_schedule], axis=1)

            # Create a MultiIndex for the columns
            brand_schedule.columns = pd.MultiIndex.from_product(
                [[brand], brand_schedule.columns]
            )

            # Concatenate the joint_schedule DataFrame with the new DataFrame for the brand
            joint_schedule = pd.concat([joint_schedule, brand_schedule], axis=1)
        except Exception as e:
            print(f"An error occurred while generating schedule for {brand}: {e}")

    return joint_schedule


# Brand-Ad Channel Map


def generate_brand_ad_channel_map(brand_list: List[str], config: Dict) -> Dict:
    """
    This function generates a map of brands to their advertising channels.

    Parameters:
    brand_list (list): A list of brands.
    config (dict): A dictionary containing the configuration settings.

    Returns:
    dict: A dictionary mapping brands to their advertising channels.
    """
    brand_ad_channel_map = {}
    try:
        for brand in brand_list:
            brand_ad_channel_map[brand] = config["brands"][brand]["advertising"][
                "channels"
            ]
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return brand_ad_channel_map


# Configuration class


# class Configuration:
#     def __init__(self, config: Dict):
#         # Set up household parameters
#         self.household_sizes = config["household"]["household_sizes"]
#         self.household_size_distribution = config["household"][
#             "household_size_distribution"
#         ]
#         self.base_consumption_rate = config["household"]["base_consumption_rate"]
#         self.pantry_min_percent = config["household"]["pantry_min_percent"]

#         # Set up retail environment
#         self.brand_list = list(config["brands"].keys())
#         try:
#             self.brand_market_share = [
#                 config["brands"][brand]["current_market_share"]
#                 for brand in self.brand_list
#             ]
#             assert round(sum(self.brand_market_share), 2) == 1.0
#         except AssertionError:
#             raise ValueError("Error: Brand market shares do not sum to 1.")
#         try:
#             self.brand_base_price = {
#                 brand: config["brands"][brand]["base_product_price"]
#                 for brand in self.brand_list
#             }
#         except KeyError:
#             raise ValueError(
#                 "Error: Base price for one or more brands is missing in the config."
#             )

#         # Set up advertising and promotion
#         self.ad_decay_factor = config["household"]["ad_decay_factor"]
#         self.joint_calendar = generate_joint_ad_promo_schedule(self.brand_list, config)
#         self.brand_channel_map = generate_brand_ad_channel_map(self.brand_list, config)
#         self.loyalty_alpha = config["household"]["loyalty_alpha"]
#         self.loyalty_beta = config["household"]["loyalty_beta"]
#         self.sensitivity_alpha = config["household"]["sensitivity_alpha"]
#         self.sensitivity_beta = config["household"]["sensitivity_beta"]
#         self.channel_set = set(
#             channel
#             for channels in self.brand_channel_map.values()
#             for channel in channels
#         )
#         self.channel_priors = [
#             config["household"]["base_channel_preferences"][channel]
#             for channel in self.channel_set
#         ]


class Configuration:
    def __init__(self, config: Dict):
        self.config = config

        # Set up household parameters
        self.household_sizes = config["household"]["household_sizes"]
        self.household_size_distribution = config["household"][
            "household_size_distribution"
        ]
        self.base_consumption_rate = config["household"]["base_consumption_rate"]
        self.pantry_min_percent = config["household"]["pantry_min_percent"]

        # Set up retail environment
        self.brand_list = list(config["brands"].keys())
        self.brand_market_share = {
            brand: info["current_market_share"]
            for brand, info in config["brands"].items()
        }
        self.brand_base_price = {
            brand: info["base_product_price"]
            for brand, info in config["brands"].items()
        }

        # Set up advertising and promotion
        self.ad_decay_factor = config["household"]["ad_decay_factor"]
        self.joint_calendar = generate_joint_ad_promo_schedule(self.brand_list, config)
        self.brand_channel_map = generate_brand_ad_channel_map(self.brand_list, config)
        self.loyalty_alpha = config["household"]["loyalty_alpha"]
        self.loyalty_beta = config["household"]["loyalty_beta"]
        self.sensitivity_alpha = config["household"]["sensitivity_alpha"]
        self.sensitivity_beta = config["household"]["sensitivity_beta"]
        self.channel_set = set(
            channel
            for channels in self.brand_channel_map.values()
            for channel in channels
        )
        self.channel_priors = {
            channel: config["household"]["base_channel_preferences"][channel]
            for channel in self.channel_set
        }
