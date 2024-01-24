import math
import mesa
import toml
import warnings
import logging
import numpy as np
from cabm_function_library.joint_calendar import generate_joint_ad_promo_schedule
from cabm_function_library.ad_helpers import (
    generate_brand_ad_channel_map,
    assign_weights,
    calculate_adstock,
    update_adstock,
    get_purchase_probabilities,
    ad_decay,
)
from cabm_function_library.agent_and_model_functions import (
    get_pantry_max,
    get_current_price,
    compute_total_purchases,
    compute_average_price,
)

# config logger
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.ERROR)

file_handler = logging.FileHandler("cabm.log")
file_handler.setLevel(logging.DEBUG)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[file_handler, stream_handler],
)

# create logger
logger = logging.getLogger(__name__)

logger.info("CABM RUNTIME STARTED")

config = toml.load("config.toml")

# Set up household parameters
household_sizes = config["household"]["household_sizes"]
household_size_distribution = config["household"]["household_size_distribution"]
base_consumption_rate = config["household"]["base_consumption_rate"]
pantry_min_percent = config["household"]["pantry_min_percent"]

# Set up retail environment
brand_list = list(config["brands"].keys())
brand_market_share = [
    config["brands"][brand]["current_market_share"] for brand in brand_list
]
try:
    assert round(sum(brand_market_share), 2) == 1.0
except AssertionError:
    print("Error: Brand market shares do not sum to 1.")

# DEBUG PRINT STATEMENT
# print(brand_market_share)


# Set up advertising and promotion
ad_decay_factor = config["household"]["ad_decay_factor"]
joint_calendar = generate_joint_ad_promo_schedule(brand_list, config)
brand_channel_map = generate_brand_ad_channel_map(brand_list, config)
loyalty_alpha = config["household"]["loyalty_alpha"]
loyalty_beta = config["household"]["loyalty_beta"]
sensitivity_alpha = config["household"]["sensitivity_alpha"]
sensitivity_beta = config["household"]["sensitivity_beta"]

channel_set = set(
    channel for channels in brand_channel_map.values() for channel in channels
)
channel_priors = [
    config["household"]["base_channel_preferences"][channel] for channel in channel_set
]


### TEST USING A BETA SAMPLER THAT AVOIDS SMALL VALUES
def sample_beta_min(alpha, beta, min_value=0.05, override=None):
    """Sample from a beta distribution, rejecting values less than min_value.
    If override is specified, return only that value."""
    if override is not None:
        warnings.warn("Beta Sampler Override is in effect.")
        return override
    sample = np.random.beta(alpha, beta)
    while abs(sample) < min_value:
        sample = np.random.beta(alpha, beta)
    return sample


class ConsumerAgent(mesa.Agent):
    """Consumer of products"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.household_size = np.random.choice(
            household_sizes, p=household_size_distribution
        )
        self.consumption_rate = abs(
            np.random.normal(base_consumption_rate, 1)
        )  # Applied at household level
        self.brand_preference = np.random.choice(
            self.model.brand_list, p=brand_market_share
        )
        self.loyalty_rate = sample_beta_min(loyalty_alpha, loyalty_beta)
        self.enable_ads = self.model.enable_ads
        self.ad_decay_factor = abs(np.random.normal(ad_decay_factor, 1))
        self.ad_channel_preference = assign_weights(channel_set, channel_priors)
        self.adstock = {i: 0 for i in self.model.brand_list}
        self.ad_sensitivity = sample_beta_min(sensitivity_alpha, sensitivity_beta)
        self.purchase_probabilities = {
            brand: self.loyalty_rate
            if brand == self.brand_preference
            else (1 - self.loyalty_rate) / (len(self.model.brand_list) - 1)
            for brand in self.model.brand_list
        }
        self.pantry_min = (
            self.household_size * pantry_min_percent
        )  # Forces must-buy when stock drops percentage of household size
        self.pantry_max = get_pantry_max(self.household_size, self.pantry_min)
        self.pantry_stock = self.pantry_max  # Start with a fully stocked pantry
        self.purchased_this_step = {brand: 0 for brand in self.model.brand_list}
        self.current_price = config["brands"][self.brand_preference][
            "base_product_price"
        ]
        self.last_product_price = config["brands"][self.brand_preference][
            "base_product_price"
        ]
        self.purchase_behavior = "buy_minimum"
        self.step_min = (
            0  # fewest number of products needed to bring stock above pantry minimum
        )
        self.step_max = 0

        # logger.debug(f"Initial Household Size: {self.household_size}")

        # self.DEBUG_print_initial_variables()

    # def DEBUG_print_initial_variables(self):
    #     print("Initialized Variables:")
    #     print(f"Household Size: {self.household_size}")
    #     print(f"Consumption Rate: {self.consumption_rate}")
    #     print(f"Brand Preference: {self.brand_preference}")
    #     print(f"Loyalty Rate: {self.loyalty_rate}")
    #     print(f"Ad Decay Factor: {self.ad_decay_factor}")
    #     print(f"Ad Channel Preference: {self.ad_channel_preference}")
    #     print(f"Adstock: {self.adstock}")
    #     print(f"Ad Sensitivity: {self.ad_sensitivity}")
    #     print(f"Purchase Probabilities: {self.purchase_probabilities}")
    #     print(f"Pantry Min: {self.pantry_min}")
    #     print(f"Pantry Max: {self.pantry_max}")
    #     print(f"Pantry Stock: {self.pantry_stock}")
    #     print(f"Purchased This Step: {self.purchased_this_step}")
    #     print(f"Current Price: {self.current_price}")
    #     print(f"Last Product Price: {self.last_product_price}")
    #     print(f"Purchase Behavior: {self.purchase_behavior}")
    #     print(f"Step Min: {self.step_min}")
    #     print(f"Step Max: {self.step_max}")

    def consume(self):
        try:
            self.pantry_stock = self.pantry_stock - (
                self.household_size / self.consumption_rate
            )
        except ZeroDivisionError:
            print("Error: Consumption rate cannot be zero.")
        except Exception as e:
            print("An unexpected error occurred:", e)

    def ad_exposure(self):
        """
        This function handles the ad exposure for the consumer agent.
        It decays the current adstock, calculates the weekly adstock,
        updates the adstock, generates purchase probabilities, and
        updates the preferred brand based on advertising effects.
        """
        try:
            # 1) Decay current self.adstock
            self.adstock = ad_decay(self.adstock, self.ad_decay_factor)

            # 2) Calculate this week's adstock
            weekly_adstock = calculate_adstock(
                self.model.week_number,
                joint_calendar,
                brand_channel_map,
                self.ad_channel_preference,
            )

            # 3) Update self.adstock
            self.adstock = update_adstock(self.adstock, weekly_adstock)

            # 4) Generate purchase probabilities
            self.purchase_probabilities = get_purchase_probabilities(
                self.adstock,
                self.brand_preference,
                self.loyalty_rate,
                self.ad_sensitivity,
            )
            # 5) Update preferred brand based purchase probabilities
            brands = list(self.purchase_probabilities.keys())
            probabilities = list(self.purchase_probabilities.values())
            self.brand_preference = np.random.choice(brands, p=probabilities)

            ## DEBUG PRINT STATEMENTS
            print(f"Agent: {self.unique_id}; Adstock: {self.adstock}")
        except ZeroDivisionError:
            print("Error: Division by zero in ad_exposure.")
        except KeyError as e:
            print(f"KeyError occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred in ad_exposure: {e}")

    def set_purchase_behavior(self):
        try:
            self.current_price = get_current_price(
                self.model.week_number, joint_calendar, self.brand_preference
            )
            price_dropped = self.current_price < self.last_product_price
            if self.pantry_stock <= self.pantry_min:
                self.purchase_behavior = (
                    "buy_maximum" if price_dropped else "buy_minimum"
                )
            elif self.pantry_min < self.pantry_stock < self.pantry_max:
                self.purchase_behavior = (
                    "buy_maximum" if price_dropped else "buy_some_or_none"
                )
            elif self.pantry_stock >= self.pantry_max:
                self.purchase_behavior = "buy_none"
        except Exception as e:
            print("An unexpected error occurred in set_purchase_behavior:", e)

    def purchase(self):
        """
        This method simulates the purchase behavior of the consumer agent.
        It first resets the purchase count for the current step.
        Then, it determines the minimum and maximum possible purchases for the step based on the current pantry stock.
        Depending on the purchase behavior, it updates the purchase count and the pantry stock.
        """
        try:
            self.purchased_this_step = {
                brand: 0 for brand in self.model.brand_list
            }  # Reset purchase count
            # Determine purchase needed this step to maintain pantry_min or above
            if self.pantry_stock <= self.pantry_min:
                self.step_min = math.ceil(self.pantry_min - self.pantry_stock)
            else:
                self.step_min = 0
            # Set max possible purchase for step
            self.step_max = math.floor(self.pantry_max - self.pantry_stock)
            # Update purchase count based on purchase behavior
            if self.purchase_behavior == "buy_minimum":
                self.purchased_this_step[self.brand_preference] += self.step_min
            elif self.purchase_behavior == "buy_maximum":
                self.purchased_this_step[self.brand_preference] += self.step_max
            elif self.purchase_behavior == "buy_some_or_none":
                # Include 0 as a possible purchase even if pantry not full
                self.purchased_this_step[self.brand_preference] += np.random.choice(
                    list(range(0, (self.step_max + 1)))
                )
            elif self.purchase_behavior == "buy_none":
                self.purchased_this_step[self.brand_preference] += 0  # No purchase
            # Update pantry stock
            self.pantry_stock += sum(self.purchased_this_step.values())
        except Exception as e:
            print("An unexpected error occurred in purchase:", e)

    def step(self):
        self.consume()
        if self.model.enable_ads:
            self.ad_exposure()
        self.set_purchase_behavior()
        self.purchase()


class ConsumerModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, enable_ads=True):
        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)
        self.week_number = 1  # Add week_number attribute
        self.brand_list = brand_list
        self.enable_ads = enable_ads

        # Create agents
        for i in range(self.num_agents):
            a = ConsumerAgent(i, self)
            self.schedule.add(a)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Total_Purchases": compute_total_purchases,
                "Average_Product_Price": compute_average_price,
                "Week_Number": "week_number",
            },
            agent_reporters={
                "Household_Size": "household_size",
                "Purchased_This_Step": "purchased_this_step",
                "Pantry_Stock": "pantry_stock",
                "Pantry_Max": "pantry_max",
                "Pantry_Min": "pantry_min",
                "Purchase_Behavior": "purchase_behavior",
                "Minimum_Purchase_Needed": "step_min",
                "Current_Product_Price": "current_price",
                "Last_Product_Price": "last_product_price",
                "Brand_Preference": "brand_preference",
            },
        )

    def step(self):
        self.datacollector.collect(self)
        """Advance the model by one step and collect data"""
        self.schedule.step()
        self.week_number += 1  # Increment week_number each step
        if self.week_number == 53:  # Reset week_number to 1 after 52 weeks
            self.week_number = 1
