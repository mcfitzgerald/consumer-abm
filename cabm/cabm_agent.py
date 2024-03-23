import math
import mesa
import toml
import numpy as np
from .cabm_helpers.config_helpers import Configuration
from .cabm_helpers.ad_helpers import (
    assign_weights,
    calculate_adstock,
    update_adstock,
    get_purchase_probabilities,
    ad_decay,
)
from .cabm_helpers.agent_and_model_functions import (
    sample_normal_min,
    sample_beta_min,
    get_pantry_max,
    get_current_price,
    compute_total_purchases,
    compute_average_price,
)


# Instantiate agents
class ConsumerAgent(mesa.Agent):
    """Consumer of products"""

    def __init__(self, unique_id, model, config):
        super().__init__(unique_id, model)
        self.config = config

        self.initialize_household()
        self.initialize_brand_preference()
        self.initialize_ad_preferences()
        self.initialize_pantry()
        self.initialize_prices()

    def initialize_household(self):
        self.household_size = np.random.choice(
            self.config.household_sizes, p=self.config.household_size_distribution
        )
        self.consumption_rate = sample_normal_min(self.config.base_consumption_rate)

    def initialize_brand_preference(self):
        self.brand_preference = np.random.choice(
            list(self.config.brand_market_share.keys()),
            p=list(self.config.brand_market_share.values()),
        )
        self.loyalty_rate = sample_beta_min(
            self.config.loyalty_alpha, self.config.loyalty_beta, override=0.99
        )
        self.purchase_probabilities = {
            brand: (
                self.loyalty_rate
                if brand == self.brand_preference
                else (1 - self.loyalty_rate) / (len(self.config.brand_list) - 1)
            )
            for brand in self.config.brand_list
        }
        # Adding a purchase choice in addition to preference so that preference remains but choice can change
        self.purchase_choice = self.brand_preference

    def initialize_ad_preferences(self):
        self.enable_ads = self.model.enable_ads
        self.ad_decay_factor = sample_normal_min(
            self.config.ad_decay_factor, override=2
        )
        self.ad_channel_preference = assign_weights(
            list(self.config.channel_priors.keys()),
            list(self.config.channel_priors.values()),
        )
        self.adstock = {i: 0 for i in self.config.brand_list}
        self.ad_sensitivity = sample_beta_min(
            self.config.sensitivity_alpha, self.config.sensitivity_beta
        )

    def initialize_pantry(self):
        self.pantry_min = (
            self.household_size * self.config.pantry_min_percent
        )  # Forces must-buy when stock drops percentage of household size
        self.pantry_max = get_pantry_max(self.household_size, self.pantry_min)
        self.pantry_stock = self.pantry_max  # Start with a fully stocked pantry
        self.purchased_this_step = {brand: 0 for brand in self.config.brand_list}

    def initialize_prices(self):
        self.current_price = self.config.brand_base_price[self.brand_preference]
        self.last_product_price = self.config.brand_base_price[self.brand_preference]
        self.purchase_behavior = "buy_minimum"
        self.step_min = (
            0  # fewest number of products needed ot bring stock above pantry min
        )
        self.step_max = 0

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
            # print(f"TRACE FOR WEEK = {self.model.week_number}")
            # print(f"step 0 - current adstock: {self.adstock}")
            self.adstock = ad_decay(self.adstock, self.ad_decay_factor)
            # print(f"step 1 - decayed adstock: {self.adstock}")

            # 2) Calculate this week's adstock
            weekly_adstock = calculate_adstock(
                self.model.week_number,
                self.config.joint_calendar,
                self.config.brand_channel_map,
                self.ad_channel_preference,
            )
            # print(f"step 2 - new adstock this week: {weekly_adstock}")

            # 3) Update self.adstock
            self.adstock = update_adstock(self.adstock, weekly_adstock)
            # print(f"step 3 - updated adstock: {self.adstock}")

            # 4) Generate purchase probabilities
            # breakpoint()
            self.purchase_probabilities = get_purchase_probabilities(
                self.adstock,
                self.brand_preference,
                self.loyalty_rate,
                self.ad_sensitivity,
            )
            # 5) Update preferred brand based purchase probabilities
            brands = list(self.purchase_probabilities.keys())
            probabilities = list(self.purchase_probabilities.values())
            self.brand_choice = np.random.choice(brands, p=probabilities)

        except ZeroDivisionError:
            print("Error: Division by zero in ad_exposure.")
        except KeyError as e:
            print(f"KeyError occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred in ad_exposure: {e}")

    def set_purchase_behavior(self):
        try:
            self.current_price = get_current_price(
                self.model.week_number,
                self.config.joint_calendar,
                self.brand_choice,
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
                brand: 0 for brand in self.config.brand_list
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
                self.purchased_this_step[self.brand_choice] += self.step_min
            elif self.purchase_behavior == "buy_maximum":
                self.purchased_this_step[self.brand_choice] += self.step_max
            elif self.purchase_behavior == "buy_some_or_none":
                # Include 0 as a possible purchase even if pantry not full
                self.purchased_this_step[self.brand_choice] += np.random.choice(
                    list(range(0, (self.step_max + 1)))
                )
            elif self.purchase_behavior == "buy_none":
                self.purchased_this_step[self.brand_choice] += 0  # No purchase
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
