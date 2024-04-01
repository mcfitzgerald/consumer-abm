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
)

# Set up logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a log file with current date and time
logfile = datetime.datetime.now().strftime("log_%m%d%y%H%M%p.log")

# Set up file handler for logger
file_handler = logging.FileHandler(logfile)
file_handler_format = (
    "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(levelname)s: %(message)s"
)
file_handler.setFormatter(logging.Formatter(file_handler_format))
logger.addHandler(file_handler)


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
        enable_pricepoint (bool): Flag to enable pricepoint.
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
        enable_pricepoint: bool = True,
    ):
        """
        Initialize the ConsumerModel.

        Args:
            N (int): Number of agents.
            config_file (str): Path to the configuration file.
            enable_ads (bool, optional): Flag to enable ads. Defaults to True.
            enable_pricepoint (bool, optional): Flag to enable pricepoint. Defaults to True.
        """
        super().__init__()

        # Load CABM configuration
        config = toml.load(config_file)
        self.config: Configuration = Configuration(config)

        self.num_agents: int = N
        self.schedule: mesa.time.RandomActivation = mesa.time.RandomActivation(self)
        self.week_number: int = 1  # Initialize week_number attribute
        self.enable_ads: bool = enable_ads
        self.enable_pricepoint: bool = enable_pricepoint
        self.brand_list: List[str] = self.config.brand_list

        # Create agents
        for i in range(self.num_agents):
            a: ConsumerAgent = ConsumerAgent(i, self, self.config)
            self.schedule.add(a)

        # Set up data collector
        self.datacollector: mesa.DataCollector = mesa.DataCollector(
            model_reporters={
                "Total_Purchases": compute_total_purchases,
                "Average_Product_Price": compute_average_price,
                "Week_Number": "week_number",
            },
            agent_reporters={
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
                "Ad_Sensitivity": "ad_sensitivity",
                "Pantry_Min": "pantry_min",
                "Pantry_Max": "pantry_max",
                "Pantry_Stock": "pantry_stock",
                "Purchased_This_Step": "purchased_this_step",
                "Current_Price": "current_price",
                "Last_Product_Price": "last_product_price",
                "Purchase_Behavior": "purchase_behavior",
                "Step_Min": "step_min",
                "Step_Max": "step_max",
            },
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
