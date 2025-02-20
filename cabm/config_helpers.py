# cabm/config_helpers.py

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


def generate_joint_ad_promo_schedule(brands: List[str], config: Dict) -> pd.DataFrame:
    """
    Creates a single DataFrame with columns (brand, 'price'), (brand, channel1), ...
    Indices = 1..52 for weeks.
    """
    joint_schedule = pd.DataFrame()

    for brand in brands:
        try:
            base_price, promo_calendar = prepare_promo_schedule_variables(brand, config)
            promo_schedule = generate_brand_promo_schedule(base_price, promo_calendar)

            out = promo_schedule  # has 1 col named 'price'
            out.columns = pd.MultiIndex.from_product([[brand], out.columns])

            # Ad schedule
            ad_vars = prepare_ad_schedule_variables(brand, config)
            if ad_vars is not None:
                (budget, media_channels, priority, schedule) = ad_vars
                ad_schedule = generate_brand_ad_schedule(
                    budget, media_channels, priority, schedule
                )
                # also rename with brand as level 0
                ad_schedule.columns = pd.MultiIndex.from_product(
                    [[brand], ad_schedule.columns]
                )
                out = pd.concat([out, ad_schedule], axis=1)

            joint_schedule = pd.concat([joint_schedule, out], axis=1)
        except Exception as e:
            print(f"Error generating schedule for brand={brand}: {e}")

    return joint_schedule


def generate_brand_ad_channel_map(brand_list: List[str], config: Dict) -> Dict:
    """
    brand -> list of channels it advertises on
    """
    brand_ad_channel_map = {}
    for b in brand_list:
        try:
            brand_info = config["brands"][b]
            ch = brand_info["advertising"]["channels"]
            brand_ad_channel_map[b] = ch
        except KeyError:
            brand_ad_channel_map[b] = []
    return brand_ad_channel_map


class Configuration:
    """
    Wraps the raw TOML config dict and provides direct attributes for model usage.
    """

    def __init__(self, config: Dict):
        self.config = config

        # Household
        hh = config["household"]
        self.household_sizes = hh["household_sizes"]
        self.household_size_distribution = hh["household_size_distribution"]
        self.base_consumption_rate = hh["base_consumption_rate"]
        self.pantry_min_percent = hh["pantry_min_percent"]
        self.consumption_rate_override = hh["consumption_rate_override"]
        self.ad_decay_override = hh["ad_decay_override"]
        self.loyalty_rate_override = hh["loyalty_rate_override"]

        self.ad_decay_factor = hh["ad_decay_factor"]
        self.adstock_incremental_sensitivity = hh["adstock_incremental_sensitivty"]
        self.adstock_incremental_midpoint = hh["adstock_incremental_midpoint"]

        self.loyalty_alpha = hh["loyalty_alpha"]
        self.loyalty_beta = hh["loyalty_beta"]
        self.purchase_history_range_lower = hh["purchase_history_range_lower"]
        self.purchase_history_range_upper = hh["purchase_history_range_upper"]

        self.price_increase_sensitivity = hh["price_increase_sensitivity"]
        self.price_decrease_sensitivity = hh["price_decrease_sensitivity"]
        self.price_threshold = hh["price_threshold"]

        # Brands
        self.brand_list = list(config["brands"].keys())
        self.brand_market_share = {
            brand: config["brands"][brand]["current_market_share"]
            for brand in self.brand_list
        }
        self.brand_base_price = {
            brand: config["brands"][brand]["base_product_price"]
            for brand in self.brand_list
        }

        # Build the joint calendar
        self.joint_calendar = generate_joint_ad_promo_schedule(self.brand_list, config)

        # brand-> channels
        self.brand_channel_map = generate_brand_ad_channel_map(self.brand_list, config)

        # channel set + base preferences
        self.channel_set = set(
            ch for chlist in self.brand_channel_map.values() for ch in chlist
        )
        self.channel_priors = {}
        base_cp = hh["base_channel_preferences"]
        for c in self.channel_set:
            self.channel_priors[c] = base_cp.get(c, 0.0)
