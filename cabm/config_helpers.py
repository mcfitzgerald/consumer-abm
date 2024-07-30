import pandas as pd
from typing import Dict, List
from .ad_calendar import (
    prepare_ad_schedule_variables,
    generate_brand_ad_schedule,
)
from .promo_calendar import (
    prepare_promo_schedule_variables,
    generate_brand_promo_schedule,
)


# Ad and Promo Schedule
def generate_joint_ad_promo_schedule(brands: List[str], config: Dict) -> pd.DataFrame:
    """
    Generates a joint ad and promo schedule for a list of brands.

    Args:
        brands (List[str]): A list of brand names.
        config (Dict): A dictionary containing configuration parameters.

    Returns:
        pd.DataFrame: A DataFrame containing the joint ad and promo schedule for the brands.
    """
    joint_schedule = pd.DataFrame()

    for brand in brands:
        try:
            # Prepare variables for promo schedule
            (base_product_price, promo_calendar) = prepare_promo_schedule_variables(
                brand, config
            )

            # Generate promo schedule
            promo_schedule = generate_brand_promo_schedule(
                base_product_price, promo_calendar
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
    Generates a map of brands to their advertising channels.

    Args:
        brand_list (List[str]): A list of brands.
        config (Dict): A dictionary containing the configuration settings.

    Returns:
        Dict: A dictionary mapping brands to their advertising channels.
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


class Configuration:
    """
    A class to represent the configuration of the model. Fed by config.toml file.

    Attributes:
        config (Dict): A dictionary containing the configuration settings.
        household_sizes (List[int]): A list of household sizes.
        household_size_distribution (List[float]): A list of household size distributions.
        base_consumption_rate (float): The base consumption rate.
        pantry_min_percent (float): The minimum pantry percent.
        brand_list (List[str]): A list of brands.
        brand_market_share (Dict[str, float]): A dictionary mapping brands to their market shares.
        brand_base_price (Dict[str, float]): A dictionary mapping brands to their base prices.
        ad_decay_factor (float): The ad decay factor.
        joint_calendar (pd.DataFrame): A DataFrame representing the joint calendar.
        brand_channel_map (Dict[str, List[str]]): A dictionary mapping brands to their channels.
        loyalty_alpha (float): The loyalty alpha.
        loyalty_beta (float): The loyalty beta.
        channel_set (set): A set of channels.
        channel_priors (Dict[str, float]): A dictionary mapping channels to their priors.
    """

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

        self.consumption_rate_override = config["household"][
            "consumption_rate_override"
        ]
        self.ad_decay_override = config["household"]["ad_decay_override"]
        self.loyalty_rate_override = config["household"]["loyalty_rate_override"]

        # Set up advertising and promotion
        self.ad_decay_factor = config["household"]["ad_decay_factor"]
        self.joint_calendar = generate_joint_ad_promo_schedule(self.brand_list, config)
        self.brand_channel_map = generate_brand_ad_channel_map(self.brand_list, config)
        self.price_increase_sensitivity = config["household"][
            "price_increase_sensitivity"
        ]
        self.price_decrease_sensitivity = config["household"][
            "price_decrease_sensitivity"
        ]
        self.price_threshold = config["household"]["price_threshold"]
        self.loyalty_alpha = config["household"]["loyalty_alpha"]
        self.loyalty_beta = config["household"]["loyalty_beta"]
        self.purchase_history_range_lower = config["household"][
            "purchase_history_range_lower"
        ]
        self.purchase_history_range_upper = config["household"][
            "purchase_history_range_upper"
        ]
        self.channel_set = set(
            channel
            for channels in self.brand_channel_map.values()
            for channel in channels
        )
        self.channel_priors = {
            channel: config["household"]["base_channel_preferences"][channel]
            for channel in self.channel_set
        }
