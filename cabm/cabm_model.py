# cabm/cabm_model.py

import mesa
import toml
import logging
from typing import List
from .config_helpers import Configuration
from .cabm_agent import ConsumerAgent
from .model_functions import (
    compute_total_purchases,
    compute_average_price,
    compute_average_purchase_probability,
)


class ConsumerModel(mesa.Model):
    """
    A model that simulates a consumer market with a specified number of agents.
    """

    def __init__(
        self,
        N: int,
        config_file: str,
        enable_ads: bool = False,
        compare_brand_prices: bool = False,
        enable_ad_increment: bool = False,
        enable_elasticity: bool = False,
    ):
        super().__init__()
        raw_conf = toml.load(config_file)
        self.config: Configuration = Configuration(raw_conf)

        self.num_agents: int = N
        self.schedule: mesa.time.RandomActivation = mesa.time.RandomActivation(self)
        self.week_number: int = 1
        self.enable_ads = enable_ads
        self.compare_brand_prices = compare_brand_prices
        self.enable_ad_increment = enable_ad_increment
        self.enable_elasticity = enable_elasticity

        self.brand_list: List[str] = self.config.brand_list

        # Create agents
        for i in range(self.num_agents):
            agent = ConsumerAgent(i, self, self.config)
            self.schedule.add(agent)

        # DataCollector
        agent_reporters = {
            "Household_Size": "household_size",
            "Consumption_Rate": "consumption_rate",
            "Brand_Preference": "brand_preference",
            "Brand_Choice": "brand_choice",
            "Loyalty_Rate": "loyalty_rate",
            "Purchase_Probabilities": "purchase_probabilities",
            "Enable_Ads": "enable_ads",
            "Ad_Decay_Factor": "ad_decay_factor",
            "Ad_Channel_Preference": "ad_channel_preference",
            "Adstock": "adstock",
            "Pantry_Min": "pantry_min",
            "Pantry_Max": "pantry_max",
            "Pantry_Stock": "pantry_stock",
            "Purchased_This_Step": "purchased_this_step",
            "Current_Price": "current_price",
            "Last_Product_Price": "last_product_price",  # if needed
            "Step_Min": "step_min",
            "Step_Max": "step_max",
            "Units_to_Purchase": "units_to_purchase",
            "Baseline_Units": "baseline_units",
            "Incremental_Promo_Units": "incremental_promo_units",
            "Incremental_Ad_Units": "incremental_ad_units",
            "Decremental_Units": "decremental_units",
            "Price_Change": "price_change",
        }

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Total_Purchases": compute_total_purchases,
                "Average_Product_Price": compute_average_price,
                "Average_Purchase_Probability": compute_average_purchase_probability,
                "Week_Number": "week_number",
            },
            agent_reporters=agent_reporters,
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.week_number += 1
        if self.week_number > 52:
            self.week_number = 1
