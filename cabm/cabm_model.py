import mesa
import toml
import logging
import datetime
from .config_helpers import Configuration
from .cabm_agent import ConsumerAgent

from .model_functions import (
    compute_total_purchases,
    compute_average_price,
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logfile = datetime.datetime.now().strftime("log_%m%d%y%H%M%p.log")

file_handler = logging.FileHandler(logfile)
file_handler_format = (
    "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(levelname)s: %(message)s"
)
file_handler.setFormatter(logging.Formatter(file_handler_format))
logger.addHandler(file_handler)


class ConsumerModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, config_file, enable_ads=True):
        # Initialize base class -- new requirement
        super().__init__()
        # Load CABM configuration
        config = toml.load(config_file)
        self.config = Configuration(config)

        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)
        self.week_number = 1  # Add week_number attribute
        self.enable_ads = enable_ads
        self.brand_list = self.config.brand_list

        # Create agents
        for i in range(self.num_agents):
            a = ConsumerAgent(i, self, self.config)
            self.schedule.add(a)

        self.datacollector = mesa.DataCollector(
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
        self.datacollector.collect(self)
        """Advance the model by one step and collect data"""
        self.schedule.step()
        self.week_number += 1  # Increment week_number each step
        if self.week_number == 53:  # Reset week_number to 1 after 52 weeks
            self.week_number = 1
