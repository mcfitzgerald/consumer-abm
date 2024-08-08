import math
import mesa
import logging
import numpy as np

from cabm.config_helpers import Configuration

from .agent_functions import (
    sample_normal_min,
    sample_beta_min,
    get_pantry_max,
    get_current_price,
    assign_media_channel_weights,
    calculate_adstock,
    decay_adstock,
    update_adstock,
    get_ad_impact_on_purchase_probabilities,
    get_price_impact_on_brand_choice_probabilities,
    get_probability_of_change_in_units_purchased_due_to_price,
    get_probability_of_change_in_units_purchased_due_to_adstock,
)


# Instantiate agents
class ConsumerAgent(mesa.Agent):
    """
    A class to represent a consumer of products.

    Attributes
    ----------
    unique_id : int
        Unique identifier for the agent
    model : mesa.Model
        The model instance in which the agent lives
    config : Configuration
        An instance of the Configuration class containing configuration parameters for the agent

    Methods
    -------
    initialize_household():
        Initializes the household size and consumption rate for the agent
    initialize_brand_preference():
        Initializes the brand preference and loyalty rate for the agent
    initialize_ad_preferences():
        Initializes the ad preferences for the agent
    initialize_pantry():
        Initializes the pantry for the agent
    initialize_prices():
        Initializes the prices for the agent
    consume():
        Simulates the consumption of products by the agent
    ad_exposure():
        Handles the ad exposure for the agent
    price_exposure():
        Adjusts the purchase probabilities based on price impact
    set_brand_choice():
        Updates the brand choice based on purchase probabilities
    purchase():
        Simulates the purchase behavior of the agent
    step():
        Defines the sequence of actions for the agent in each step of the simulation
    """

    def __init__(
        self,
        unique_id: int,
        model: mesa.Model,
        config: Configuration,
    ):
        super().__init__(unique_id, model)
        self.config = config

        # Initialize agent's household, brand preference, ad preferences, pantry, and prices
        self.initialize_household()
        self.initialize_brand_preference()
        self.initialize_ad_preferences()
        self.initialize_pantry()
        self.initialize_prices()
        self.add_joint_calendar_properties()

    def initialize_household(self):
        """Initializes the household size and consumption rate for the agent"""
        self.household_size = np.random.choice(
            self.config.household_sizes, p=self.config.household_size_distribution
        )
        self.consumption_rate = sample_normal_min(
            self.config.base_consumption_rate,
            override=self.config.consumption_rate_override,
        )

    def initialize_brand_preference(self):
        """Initializes the brand preference and loyalty rate for the agent"""
        self.brand_preference = np.random.choice(
            list(self.config.brand_market_share.keys()),
            p=list(self.config.brand_market_share.values()),
        )
        self.loyalty_rate = sample_beta_min(
            self.config.loyalty_alpha,
            self.config.loyalty_beta,
            override=self.config.loyalty_rate_override,
        )
        self.purchase_probabilities = {
            brand: (
                self.loyalty_rate
                if brand == self.brand_preference
                else (1 - self.loyalty_rate) / (len(self.config.brand_list) - 1)
            )
            for brand in self.config.brand_list
        }
        # Adding a brand choice in addition to preference so that preference remains but choice can change
        self.brand_choice = self.brand_preference

        # Purchase history is the last three brands purchased - used to reset brand preference if switching is persistent
        self.purchase_history_window_length = np.random.uniform(
            self.config.purchase_history_range_lower,
            self.config.purchase_history_range_upper,
        )
        self.purchase_history = [
            self.brand_choice for i in range(self.purchase_history_window_length)
        ]

    def initialize_ad_preferences(self):
        """Initializes the ad preferences for the agent"""
        self.enable_ads = self.model.enable_ads
        self.ad_decay_factor = sample_normal_min(
            self.config.ad_decay_factor, override=self.config.ad_decay_override
        )
        self.ad_channel_preference = assign_media_channel_weights(
            list(self.config.channel_priors.keys()),
            list(self.config.channel_priors.values()),
        )
        self.adstock = {i: 0 for i in self.config.brand_list}
        self.adstock_incremental_sensitivity = sample_normal_min(self.config.adstock_incremental_sensitivity)
        self.adstock_incremental_midpoint = sample_normal_min(self.config.adstock_incremental_midpoint, std_dev=(self.config.adstock_incremental_midpoint/10.0))

    def initialize_pantry(self):
        """Initializes the pantry for the agent"""
        self.pantry_min = (
            self.household_size * self.config.pantry_min_percent
        )  # Forces must-buy when stock drops percentage of household size
        self.pantry_max = get_pantry_max(self.household_size, self.pantry_min)
        self.pantry_stock = self.pantry_max  # Start with a fully stocked pantry
        self.step_min = (
            0  # fewest number of products needed ot bring stock above pantry min
        )
        self.step_max = 0
        self.baseline_units = 0
        self.incremental_promo_units = 0
        self.incremental_ad_units = 0
        self.decremental_units = 0
        self.purchased_this_step = {brand: 0 for brand in self.config.brand_list}

    def initialize_prices(self):
        """Initializes the prices for the agent"""
        self.current_price = self.config.brand_base_price[self.brand_choice]
        # deprecated? #self.last_product_price = self.config.brand_base_price[self.brand_choice]
        self.price_change = "no_price_change"
        self.price_increase_sensitivity = sample_normal_min(self.config.price_increase_sensitivity)
        self.price_decrease_sensitivity = sample_normal_min(self.config.price_decrease_sensitivity)
        self.price_threshold = self.config.price_threshold

    def _create_joint_calendar_property(self, property_name, brand, attribute):
        def getter(self):
            return self.model.config.joint_calendar.loc[
                self.model.week_number, (brand, attribute)
            ]

        setattr(self.__class__, property_name, property(getter))

    def add_joint_calendar_properties(self):
        for brand, attribute in self.model.config.joint_calendar.columns:
            property_name = f"{attribute.lower()}_{brand.upper()}"
            self._create_joint_calendar_property(property_name, brand, attribute)

    def consume(self):
        """Simulates the consumption of products by the agent"""
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
            self.adstock = decay_adstock(self.adstock, self.ad_decay_factor)

            # 2) Calculate this week's adstock
            weekly_adstock = calculate_adstock(
                self.model.week_number,
                self.config.joint_calendar,
                self.config.brand_channel_map,
                self.ad_channel_preference,
            )

            # 3) Update self.adstock
            self.adstock = update_adstock(self.adstock, weekly_adstock)

            # 4) Generate purchase probabilities
            logging.debug("*** NEXT AGENT OR STEP ***")
            logging.debug(f"Agent ID: {self.unique_id}, Step: {self.model.week_number}")
            self.purchase_probabilities = get_ad_impact_on_purchase_probabilities(
                self.adstock, self.brand_preference, self.loyalty_rate
            )

        except ZeroDivisionError:
            print("Error: Division by zero in ad_exposure.")
        except KeyError as e:
            print(f"KeyError occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred in ad_exposure: {e}")

    def compare_brand_prices(self):
        """
        This function adjusts the purchase probabilities based on price impact.
        It first calculates the price impact probabilities, then adjusts the purchase probabilities accordingly.
        Finally, it normalizes the adjusted purchase probabilities so they sum to 1.
        """
        try:
            price_impact_probabilities = get_price_impact_on_brand_choice_probabilities(
                self.model.week_number,
                self.config.brand_list,
                self.config.joint_calendar,
                self.brand_preference,
                self.loyalty_rate,
            )

            # Averaging ad impact and price impact
            self.purchase_probabilities = {
                brand: (
                    price_impact_probabilities[brand]
                    + self.purchase_probabilities[brand]
                )
                / 2
                for brand in self.purchase_probabilities
            }

        except ZeroDivisionError:
            print("Error: Division by zero in price_exposure.")
        except KeyError as e:
            print(f"KeyError occurred in price_exposure: {e}")
        except Exception as e:
            print(f"An unexpected error occurred in price_exposure: {e}")

    def set_brand_choice(self):
        """Updates the brand choice based on purchase probabilities"""
        brands = list(self.purchase_probabilities.keys())
        probabilities = list(self.purchase_probabilities.values())
        self.brand_choice = np.random.choice(brands, p=probabilities)

        logging.debug(f"Purchase probabilities: {self.purchase_probabilities}")
        logging.debug(f"Brand choice: {self.brand_choice}")

    def reset_purchased_this_step(self):
        self.purchased_this_step = {
            brand: 0 for brand in self.config.brand_list
        }  # Reset purchase count

    def get_step_min_and_max_units(self):
        # Determine the minimum and maximum possible purchases for the step
        self.step_min = max(0, math.ceil(self.pantry_min - self.pantry_stock))
        self.step_max = max(0, math.floor(self.pantry_max - self.pantry_stock))

        if self.step_min > self.step_max:
            # If step_min is greater than step_max, set both to 0
            self.step_min = self.step_max = 0

    def get_baseline_units_to_purchase(self):
        """
        This method simulates the purchase behavior of the consumer agent.
        It uses a triangle distribution to set the number of units purchased,
        while respecting the pantry_min and pantry_max constraints.
        """
        try:
            if self.step_max > 0:
                if self.step_min == self.step_max:
                    # If min and max are the same, set units to purchase to that value
                    self.baseline_units = self.step_min
                else:
                    # Ensure mode is between step_min and step_max
                    mode = self.step_min + (self.step_max - self.step_min) / 2
                    self.baseline_units = int(
                        np.random.triangular(self.step_min, mode, self.step_max)
                    )
        except Exception as e:
            print("An unexpected error occurred in purchase:", e)

    def check_price(self):
        self.current_price = get_current_price(
            self.model.week_number,
            self.config.joint_calendar,
            self.brand_choice,
        )
        if self.current_price < self.config.brand_base_price[self.brand_choice]:
            self.price_change = "price_decrease"
        elif self.current_price > self.config.brand_base_price[self.brand_choice]:
            self.price_change = "price_increase"
        else:
            self.price_change = "no_price_change"

    def change_units_to_purchase_based_on_price(self):
        """
        Uses probability of an incrementing or decrementing event
        to change baseline units based on price
        """
        event_probability = get_probability_of_change_in_units_purchased_due_to_price(
            self.config.brand_base_price[self.brand_choice],
            self.current_price,
            sensitivity_increase = self.price_increase_sensitivity, 
            sensitivity_decrease = self.price_decrease_sensitivity, 
            threshold = self.price_threshold, 
        )

        event_branch = np.random.choice(
            [True, False], p=[event_probability, 1 - event_probability]
        )

        if event_branch:
            if self.price_change == "price_decrease":
                max_additional_units = self.step_max - self.baseline_units
                if max_additional_units > 0:
                    self.incremental_promo_units = np.random.randint(
                        1, max_additional_units + 1
                    )
                else:
                    self.incremental_promo_units = 0
            if self.price_change == "price_increase":
                max_decremental_units = self.baseline_units - self.step_min
                if max_decremental_units > 0:
                    self.decremental_units = np.random.randint(
                        1, max_decremental_units + 1
                    )
                else:
                    self.decremental_units = 0
            if self.price_change == "no_price_change":
                return
            
    def change_units_to_purchase_based_on_adstock(self):
        """
        Uses probability of an incrementing event
        to change baseline units based on price
        """
        event_probability = get_probability_of_change_in_units_purchased_due_to_adstock(
            self.adstock[self.brand_choice],
            self.adstock_incremental_sensitivity,
            self.adstock_incremental_midpoint,
        )

        event_branch = np.random.choice(
            [True, False], p=[event_probability, 1 - event_probability]
        )

        if event_branch:
            max_additional_units = self.step_max - self.baseline_units
            if max_additional_units > 0:
                self.incremental_ad_units = np.random.randint(
                    1, max_additional_units + 1
                )
            else:
                self.incremental_ad_units = 0

    # NOTE - change the logic below to make sure the sum doesn't violate max using a try block, if it does, limit to max and flip a coin to subtract difference from ad or promo but mark there was adjustment
    
    def make_purchase(self):
        units_to_purchase = (
            self.baseline_units
            + self.incremental_promo_units
            + self.incremental_ad_units
            - self.decremental_units
        )
        self.purchased_this_step[self.brand_choice] = units_to_purchase
        self.pantry_stock += units_to_purchase

    def update_brand_preference(self):
        if len(self.purchase_history) != self.purchase_history_window_length:
            raise Exception(
                f"Purchase history must have {self.purchase_history_window_length
                                              } elements."
            )
        if len(set(self.purchase_history)) == 1:
            self.brand_preference = self.purchase_history[0]
            # print(f"updating brand pref: {self.brand_preference}")

    def update_purchase_history_and_preference(self):
        self.purchase_history.pop(0)
        # print(f"popped_history: {self.purchase_history}")
        self.purchase_history.append(self.brand_choice)
        # print(f"new_purch_hist: {self.purchase_history}")
        self.update_brand_preference()

    def step(self):
        """Defines the sequence of actions for the agent in each step of the simulation"""
        self.consume()
        if self.model.enable_ads:
            self.ad_exposure()
        if self.model.compare_brand_prices:
            self.compare_brand_prices()
        self.set_brand_choice()
        self.reset_purchased_this_step()
        self.get_step_min_and_max_units()
        self.get_baseline_units_to_purchase()
        self.check_price()
        self.change_units_to_purchase_based_on_price()
        self.make_purchase()
        self.update_purchase_history_and_preference()
