import mesa
import toml
import logging
import datetime
from typing import List
from .config_helpers import Configuration
from .cabm_agent import ConsumerAgent
from .model_functions import (
    compute_total_purchases,
    compute_average_price,
    compute_average_purchase_probability,
)

# Set up logger
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

# # Create a log file with current date and time
# logfile = datetime.datetime.now().strftime("log_%m%d%y%H%M%p.log")

# # Set up file handler for logger
# file_handler = logging.FileHandler(logfile)
# file_handler_format = (
#     "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(levelname)s: %(message)s"
# )
# file_handler.setFormatter(logging.Formatter(file_handler_format))
# logger.addHandler(file_handler)


class ConsumerModel(mesa.Model):
    """
    A model that simulates a consumer market with a specified number of agents.
    Each agent represents a consumer with unique characteristics and behaviors.
    The model simulates the interactions between these consumers and the market,
    including their purchasing decisions and responses to advertising.

    Attributes:
        num_agents (int): The number of agents in the model.
        schedule (mesa.time.RandomActivation): The schedule of agent activation.
        week_number (int): The current week number in the simulation.
        enable_ads (bool): Flag to enable ads.
        compare_brand_prices (bool): Flag to enable pricepoint.
        brand_list (List[str]): List of available brands in the market.
        datacollector (mesa.DataCollector): Data collector to collect model and agent level data.

    Methods:
        __init__(self, N: int, config_file: str, enable_ads: bool = True, enable_pricepoint: bool = True):
            Initializes the ConsumerModel with the specified number of agents, configuration file, and flags for ads and pricepoint.
    """

    def __init__(
        self,
        N: int,
        config_file: str,
        enable_ads: bool = True,
        compare_brand_prices: bool = True,
        enable_ad_increment: bool = False,
    ):
        """
        Initialize the ConsumerModel.

        Args:
            N (int): Number of agents.
            config_file (str): Path to the configuration file.
            enable_ads (bool, optional): Flag to enable impact of adstock on purchase probability. Defaults to true.
            enable_compare_brand_prices (bool, optional): Flag to enable impact of pricepoint on purchase probability. Defaults to True.
            enable_ad_increment (bool, optional): Flag to enable expanded consumption (incremental purchase) based on advertisting. Defaults to False.
        """
        super().__init__()

        # Load CABM configuration
        config = toml.load(config_file)
        self.config: Configuration = Configuration(config)

        self.num_agents: int = N
        self.schedule: mesa.time.RandomActivation = mesa.time.RandomActivation(self)
        self.week_number: int = 1  # Initialize week_number attribute
        self.enable_ads: bool = enable_ads
        self.compare_brand_prices: bool = compare_brand_prices
        self.enable_ad_increment = enable_ad_increment
        self.brand_list: List[str] = self.config.brand_list

        # Create agents
        for i in range(self.num_agents):
            a: ConsumerAgent = ConsumerAgent(i, self, self.config)
            self.schedule.add(a)

        # Initialize DataCollector with dynamic agent reporters
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
            "Last_Product_Price": "last_product_price",
            "Step_Min": "step_min",
            "Step_Max": "step_max",
            "Baseline_Units": "baseline_units",
            "Incremental_Promo_Units": "incremental_promo_units",
            "Incremental_Ad_Units": "incremental_ad_units",
            "Decremental_Units": "decremental_units",
            "Price_Change": "price_change",
        }

        for brand, attribute in self.config.joint_calendar.columns:
            property_name = f"{attribute.lower()}_{brand.upper()}"
            agent_reporters[property_name] = property_name

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
        """
        Advance the model by one step and collect data.
        """
        self.datacollector.collect(self)
        self.schedule.step()
        self.week_number += 1  # Increment week_number each step
        if self.week_number == 53:  # Reset week_number to 1 after 52 weeks
            self.week_number = 1
