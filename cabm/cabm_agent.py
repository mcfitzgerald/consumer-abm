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

    Built using mesa agent based modeling framework.

    Agent is operated via ConsumerModel class in cabm_model.py

    Attributes
    ----------
    unique_id : int
        Unique identifier for the agent
    model : mesa.Model
        The model instance in which the agent lives
    config : Configuration
        An instance of the Configuration class containing configuration parameters for the agent

        Functional Description
    ----------------------
    The `step` method orchestrates the agent's behavior in each simulation step, performing the following actions:

    0. **Reset Step Values (`reset_step_values` method)**: Resets the values that need to be reinitialized at the beginning of each step.
    1. **Consume (`consume` method)**: Reduces pantry stock based on household size and consumption rate.
    2. **Ad Exposure (`ad_exposure` method)**: If enabled, updates adstock and purchase probabilities based on ads.
    3. **Price Comparison (`compare_brand_prices` method)**: If enabled, adjusts purchase probabilities based on price impact.
    4. **Set Brand Choice (`set_brand_choice` method)**: Updates brand choice based on purchase probabilities.
    5. **Determine Min and Max Units (`get_step_min_and_max_units` method)**: Calculates min and max possible purchases based on pantry stock.
    6. **Baseline Units to Purchase (`get_baseline_units_to_purchase` method)**: Determines baseline purchase units using a triangular distribution.
    7. **Check Price (`check_price` method)**: If price elasticity is enabled, updates current price and price change status.
    8. **Adjust Units Based on Price (`change_units_to_purchase_based_on_price` method)**: Adjusts units to purchase based on price changes.
    9. **Adjust Units Based on Adstock (`change_units_to_purchase_based_on_adstock` method)**: If enabled, adjusts units to purchase based on adstock effects.
    10. **Make Purchase (`make_purchase` method)**: Finalizes purchase units and updates pantry stock.
    11. **Update Purchase History and Preference (`update_purchase_history_and_preference` method)**: Updates purchase history and brand preference.
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
        """
        Initializes the household size and consumption rate for the agent.

        This method sets up two critical attributes for the agent:
        1. `household_size`: Determines the number of individuals in the agent's household.
           This is randomly chosen based on a predefined distribution provided in the configuration.
        2. `consumption_rate`: Defines the rate at which the household consumes products.
           This rate is sampled from a normal distribution with a mean specified in the configuration.
           An optional override can be applied to this rate if specified in the configuration.
        """
        self.household_size = np.random.choice(
            self.config.household_sizes, p=self.config.household_size_distribution
        )
        self.consumption_rate = sample_normal_min(
            self.config.base_consumption_rate,
            override=self.config.consumption_rate_override,
        )

    def initialize_brand_preference(self):
        """
        Initializes the brand preference, loyalty rate, and purchase history for the agent.

        This method sets up several critical attributes for the agent:
        1. `brand_preference`: Determines the agent's preferred brand based on market share.
           This is randomly chosen from the available brands using a predefined market share distribution.
        2. `loyalty_rate`: Defines the agent's loyalty to their preferred brand.
           This rate is sampled from a beta distribution with parameters specified in the configuration.
           An optional override can be applied to this rate if specified in the configuration.
        3. `purchase_probabilities`: A dictionary that maps each brand to the probability of the agent purchasing it.
           The probability is higher for the preferred brand and distributed among other brands based on the loyalty rate.
        4. `brand_choice`: Initially set to the preferred brand but can change over time based on various factors.
        5. `purchase_history_window_length`: Determines the length of the purchase history window, sampled from a uniform distribution.
        6. `purchase_history`: A list that keeps track of the last few brands purchased by the agent, used to reset brand preference if switching is persistent.
        """

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
        self.brand_choice = self.brand_preference

        self.purchase_history_window_length = int(
            np.random.uniform(
                self.config.purchase_history_range_lower,
                self.config.purchase_history_range_upper,
            )
        )
        self.purchase_history = [
            self.brand_choice for i in range(self.purchase_history_window_length)
        ]

    def initialize_ad_preferences(self):
        """
        Initializes the ad preferences for the agent.

        This method sets up various attributes related to the agent's interaction with advertisements.
        It configures whether ads are enabled, the decay factor for ad effectiveness, the agent's
        preference for different ad channels, and the adstock levels for each brand.
        """
        self.enable_ads = self.model.enable_ads
        self.ad_decay_factor = sample_normal_min(
            self.config.ad_decay_factor, override=self.config.ad_decay_override
        )
        self.ad_channel_preference = assign_media_channel_weights(
            list(self.config.channel_priors.keys()),
            list(self.config.channel_priors.values()),
        )
        self.adstock = {i: 0 for i in self.config.brand_list}
        self.adstock_incremental_sensitivity = sample_normal_min(
            self.config.adstock_incremental_sensitivity
        )
        self.adstock_incremental_midpoint = sample_normal_min(
            self.config.adstock_incremental_midpoint,
            std_dev=(self.config.adstock_incremental_midpoint / 10.0),
        )
        self.adstock_incremental_limit = 500

    def initialize_pantry(self):
        """
        Initializes the pantry for the agent.

        This method sets up various attributes related to the agent's pantry management.
        It configures the minimum and maximum pantry stock levels, initializes the pantry
        with a fully stocked state, and sets up various counters for tracking product units.
        """
        self.pantry_min = (
            self.household_size * self.config.pantry_min_percent
        )  # Forces must-buy when stock drops percentage of household size
        self.pantry_max = get_pantry_max(self.household_size, self.pantry_min)
        self.pantry_stock = self.pantry_max  # Start with a fully stocked pantry
        self.step_min = (
            0  # fewest number of products needed to bring stock above pantry min
        )
        self.step_max = 0
        self.baseline_units = 0
        self.incremental_promo_units = 0
        self.incremental_promo_overflow = 0
        self.incremental_ad_units = 0
        self.incremental_ad_overflow = 0
        self.decremental_units = 0
        self.units_to_purchase = 0
        self.purchased_this_step = {brand: 0 for brand in self.config.brand_list}

    def initialize_prices(self):
        """
        Initializes the pricing attributes for the agent.

        This method sets up various attributes related to the agent's price sensitivity and
        current price perception. It configures the initial price based on the agent's brand
        choice, sets up the price change status, and initializes the agent's sensitivity to
        price increases and decreases.
        """
        self.current_price = self.config.brand_base_price[self.brand_choice]
        self.price_change = "no_price_change"
        self.price_increase_sensitivity = sample_normal_min(
            self.config.price_increase_sensitivity
        )
        self.price_decrease_sensitivity = sample_normal_min(
            self.config.price_decrease_sensitivity
        )
        self.price_threshold = self.config.price_threshold

    def _create_joint_calendar_property(self, property_name, brand, attribute):
        """
        Dynamically creates a property for accessing joint calendar attributes.

        This method defines a getter function that retrieves the value of a specified
        attribute for a given brand from the joint calendar at the current week number.
        It then sets this getter function as a property of the class, allowing for
        easy access to joint calendar data through a dynamically named property.

        Parameters:
        - property_name (str): The name of the property to be created.
        - brand (str): The brand for which the attribute is being accessed.
        - attribute (str): The specific attribute of the brand to be accessed.

        Example:
        If `property_name` is "promo_A", `brand` is "A", and `attribute` is "promo",
        this method will create a property `promo_A` that returns the promotion value
        for brand A at the current week.
        """

        def getter(self):
            return self.model.config.joint_calendar.loc[
                self.model.week_number, (brand, attribute)
            ]

        setattr(self.__class__, property_name, property(getter))

    def add_joint_calendar_properties(self):
        """
        Adds dynamic properties to the agent for accessing joint calendar attributes.

        This method iterates over the columns of the joint calendar DataFrame, which
        contains various attributes for different brands over time. For each brand-attribute
        pair, it constructs a property name and uses the `_create_joint_calendar_property`
        method to dynamically create a property on the agent class. This allows the agent
        to access these attributes easily through named properties.

        Example:
        If the joint calendar has a column ('A', 'promo'), this method will create a
        property named `promo_A` that allows the agent to access the promotion value
        for brand A at the current week.
        """
        for brand, attribute in self.model.config.joint_calendar.columns:
            property_name = f"{attribute.lower()}_{brand.upper()}"
            self._create_joint_calendar_property(property_name, brand, attribute)

    def reset_step_values(self):
        """
        Resets the values that need to be reinitialized at the beginning of each step.
        """
        self.baseline_units = 0
        self.incremental_promo_units = 0
        self.incremental_ad_units = 0
        self.decremental_units = 0
        self.units_to_purchase = 0
        self.purchased_this_step = {brand: 0 for brand in self.config.brand_list}
        # Add any other attributes that need to be reset here

        # Debugging: Log a message indicating that values have been reset
        logging.debug(
            f"Agent ID: {self.unique_id}, Step: {self.model.week_number}, Step values have been reset."
        )

    def consume(self):
        """
        Simulates the consumption of products by the agent.

        This method reduces the agent's pantry stock based on the household size and
        the consumption rate. It ensures that the pantry stock is decremented correctly
        to reflect the consumption of products over time.
        """
        try:
            self.pantry_stock -= self.household_size / self.consumption_rate
            if self.pantry_stock < 0:
                self.pantry_stock = 0
        except ZeroDivisionError:
            print("Error: Consumption rate cannot be zero.")
        except Exception as e:
            print("An unexpected error occurred:", e)

    def ad_exposure(self):
        """
        Handles the ad exposure for the consumer agent by decaying the current adstock,
        calculating the weekly adstock, updating the adstock, generating purchase probabilities,
        and updating the preferred brand based on advertising effects.

        Steps:
        1. Decay the current adstock using the decay factor.
        2. Calculate the adstock for the current week based on the joint calendar, brand channel map, and ad channel preference.
        3. Update the adstock with the newly calculated weekly adstock.
        4. Generate purchase probabilities influenced by the updated adstock.
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
        Adjusts the purchase probabilities based on the impact of brand prices.

        This method performs the following steps:
        1. Calculates the price impact probabilities for each brand.
        2. Adjusts the current purchase probabilities by averaging them with the price impact probabilities.
        3. Normalizes the adjusted purchase probabilities so they sum to 1.
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

            # Normalize the adjusted purchase probabilities so they sum to 1
            total_probability = sum(self.purchase_probabilities.values())
            if total_probability > 0:
                self.purchase_probabilities = {
                    brand: prob / total_probability
                    for brand, prob in self.purchase_probabilities.items()
                }

        except ZeroDivisionError:
            print("Error: Division by zero in price_exposure.")
        except KeyError as e:
            print(f"KeyError occurred in price_exposure: {e}")
        except Exception as e:
            print(f"An unexpected error occurred in price_exposure: {e}")

    def set_brand_choice(self):
        """
        Updates the brand choice based on purchase probabilities.

        This method selects a brand for the agent to purchase based on the
        current purchase probabilities. It uses a random choice weighted by
        these probabilities to simulate the decision-making process of the
        consumer.

        Steps:
        1. Extracts the list of brands and their corresponding purchase probabilities.
        2. Uses a weighted random choice to select a brand based on these probabilities.
        3. Updates the `self.brand_choice` attribute with the selected brand.
        """
        brands = list(self.purchase_probabilities.keys())
        probabilities = list(self.purchase_probabilities.values())
        self.brand_choice = np.random.choice(brands, p=probabilities)

    def get_step_min_and_max_units(self):
        """
        Determines the minimum and maximum number of units the agent can purchase
        in the current simulation step based on pantry constraints.

        This method calculates the minimum (`self.step_min`) and maximum (`self.step_max`)
        possible purchases for the current step by considering the agent's pantry stock
        and the defined pantry limits (`self.pantry_min` and `self.pantry_max`).

        Calculation Details:
        - `self.step_min` is calculated as the maximum of 0 and the ceiling value of the difference
          between `self.pantry_min` and `self.pantry_stock`. This ensures that the minimum purchase
          is non-negative and respects the lower pantry limit.
        - `self.step_max` is calculated as the maximum of 0 and the floor value of the difference
          between `self.pantry_max` and `self.pantry_stock`. This ensures that the maximum purchase
          is non-negative and respects the upper pantry limit.
        - If `self.step_min` is greater than `self.step_max`, both are set to 0 to prevent invalid
          purchase ranges.

        Example:
        If `self.pantry_min` is 10, `self.pantry_max` is 50, and `self.pantry_stock` is 30,
        then `self.step_min` will be max(0, ceil(10 - 30)) = 0 and `self.step_max` will be
        max(0, floor(50 - 30)) = 20.
        """
        self.step_min = max(0, math.ceil(self.pantry_min - self.pantry_stock))
        self.step_max = max(0, math.floor(self.pantry_max - self.pantry_stock))

        if self.step_min > self.step_max:
            self.step_min = self.step_max = 0

    def get_baseline_units_to_purchase(self):
        """
        Simulates the purchase behavior of the consumer agent using a triangular distribution
        to determine the number of units to purchase, while respecting the pantry constraints.

        This method calculates the baseline number of units (`self.baseline_units`) that the agent
        will purchase in the current simulation step. The calculation is based on the minimum and
        maximum units (`self.step_min` and `self.step_max`) that the agent can purchase, which are
        derived from the pantry constraints (`self.pantry_min` and `self.pantry_max`).

        Calculation Details:
        - If `self.step_min` is equal to `self.step_max`, the baseline units are set to that value.
        - If `self.step_min` is not equal to `self.step_max`, a triangular distribution is used to randomly
          select a value between `self.step_min` and `self.step_max`, with the mode being the midpoint between
          these two values.
        - The triangular distribution ensures that values closer to the mode are more likely to be chosen,
          simulating a realistic consumer behavior where moderate purchases are more common than extreme values.

        Example:
        If `self.step_min` is 5 and `self.step_max` is 15, the mode will be 10. The method will then use a
        triangular distribution to select a value between 5 and 15, with 10 being the most likely value.
        """
        try:
            # # Reset baseline_units at the beginning of the method
            # self.baseline_units = 0

            if self.step_max > 0:
                # Define a small tolerance value for floating-point comparisons
                # epsilon = 1e-9

                # Initialize mode to None
                mode = None

                # Check that step_min is less than or equal to step_max
                if self.step_min > self.step_max:
                    raise ValueError(
                        f"step_min ({self.step_min}) cannot be greater than step_max ({self.step_max})"
                    )

                if self.step_min == self.step_max:
                    # If min and max are the same, set units to purchase to that value
                    self.baseline_units = self.step_min
                else:
                    # Ensure mode is between step_min and step_max
                    mode = self.step_min + (self.step_max - self.step_min) / 2
                    self.baseline_units = int(
                        np.random.triangular(self.step_min, mode, self.step_max)
                    )
                # Ensure baseline_units does not exceed step_max, considering floating-point precision
                if self.baseline_units > self.step_max:
                    self.baseline_units = self.step_max

                # Debugging: Print the calculated values
                logging.debug(
                    f"Agent ID: {self.unique_id}, Step: {self.model.week_number}, Baseline Units: {self.baseline_units}, Step Min: {self.step_min}, Step Max: {self.step_max}, Mode: {mode}"
                )
        except Exception as e:
            print("An unexpected error occurred in get_baseline_units_to_purchase:", e)

    def check_price(self):
        """
        Determines the current price of the selected brand and compares it to the base price.
        Sets the `price_change` attribute based on whether the current price is lower, higher, or the same as the base price.

        This method updates the following attributes:
        - `self.current_price`: The current price of the brand for the current week.
        - `self.price_change`: Indicates whether the price has decreased, increased, or remained the same compared to the base price.

        Why These Attributes:
        - `self.current_price` is essential for making purchasing decisions based on the latest price information.
        - `self.price_change` is used to adjust the number of units to purchase, reflecting consumer behavior in response to price changes.

        Calculation Details:
        - The current price is fetched using the `get_current_price` function, which takes into account the current week, joint calendar, and brand choice.
        - The `price_change` attribute is set to "price_decrease" if the current price is lower than the base price, "price_increase" if it is higher, and "no_price_change" if it is the same.
        """
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
        Adjusts the number of units to purchase based on the current price change.

        This method uses the probability of an incrementing or decrementing event to modify the baseline units
        based on the current price. The probability is determined by the price sensitivity parameters and the
        threshold value.
        """
        event_probability = get_probability_of_change_in_units_purchased_due_to_price(
            self.config.brand_base_price[self.brand_choice],
            self.current_price,
            sensitivity_increase=self.price_increase_sensitivity,
            sensitivity_decrease=self.price_decrease_sensitivity,
            threshold=self.price_threshold,
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
            elif self.price_change == "price_increase":
                max_decremental_units = self.baseline_units - self.step_min
                if max_decremental_units > 0:
                    self.decremental_units = np.random.randint(
                        1, max_decremental_units + 1
                    )
                else:
                    self.decremental_units = 0
            elif self.price_change == "no_price_change":
                return

    def change_units_to_purchase_based_on_adstock(self):
        """
        Adjusts the number of units to purchase based on adstock effects.

        This method calculates the probability of an incrementing event occurring due to adstock,
        which is a measure of the lingering effect of advertising on consumer behavior. If the event
        occurs, it increases the number of units to purchase.
        """
        event_probability = get_probability_of_change_in_units_purchased_due_to_adstock(
            self.adstock[self.brand_choice],
            self.adstock_incremental_sensitivity,
            self.adstock_incremental_midpoint,
            self.adstock_incremental_limit,
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

    def make_purchase(self):
        """
        Calculate and finalize the number of units to purchase for the current step.

        This method calculates the total units to purchase by summing up the baseline units,
        promotional incremental units, adstock incremental units, and subtracting the decremental units.
        If the total units exceed the maximum allowable units (`step_max`), it caps the total units at `step_max`.
        Additionally, if the total units are less than zero, it sets the total units to zero.
        """
        try:
            # Define a small tolerance value for floating-point comparisons
            # epsilon = 1e-9

            # Debugging: Print the intermediate values
            logging.debug(
                f"Agent ID: {self.unique_id}, Step: {self.model.week_number}, Baseline Units: {self.baseline_units}, Incremental Promo Units: {self.incremental_promo_units}, Incremental Ad Units: {self.incremental_ad_units}, Decremental Units: {self.decremental_units}"
            )

            # Calculate the proposed units to purchase
            self.units_to_purchase = (
                self.baseline_units
                + self.incremental_promo_units
                + self.incremental_ad_units
                - self.decremental_units
            )
            # Debugging: Print the proposed units to purchase
            logging.debug(
                f"Agent ID: {self.unique_id}, Step: {self.model.week_number}, Proposed Units to Purchase: {self.units_to_purchase}, Step Max: {self.step_max}"
            )

            # Ensure units_to_purchase does not exceed step_max, considering floating-point precision
            if self.units_to_purchase > self.step_max:
                self.units_to_purchase = self.step_max

            # Ensure units_to_purchase is not less than zero
            if self.units_to_purchase < 0:
                self.units_to_purchase = 0

            # Debugging: Print the final units to purchase
            logging.debug(
                f"Agent ID: {self.unique_id}, Step: {self.model.week_number}, Final Units to Purchase: {self.units_to_purchase}"
            )
        except Exception as e:
            print("An unexpected error occurred in make_purchase:", e)

        # Update the purchased units for the current step
        self.purchased_this_step[self.brand_choice] = self.units_to_purchase
        self.pantry_stock += self.units_to_purchase

    def update_brand_preference(self):
        """
        Update the brand preference based on the purchase history.

        This method checks if the purchase history has the required number of elements
        as specified by `purchase_history_window_length`. If the purchase history does not
        meet this requirement, an exception is raised. If all elements in the purchase history
        are the same, it updates the `brand_preference` to the consistent brand in the history.
        """
        if len(self.purchase_history) != self.purchase_history_window_length:
            raise Exception(
                f"Purchase history must have {self.purchase_history_window_length} elements."
            )
        if len(set(self.purchase_history)) == 1:
            self.brand_preference = self.purchase_history[0]

    def update_purchase_history_and_preference(self):
        """
        Update the purchase history and brand preference.

        This method performs two main tasks:
        1. Updates the purchase history by removing the oldest entry and appending the current brand choice.
        2. Calls the `update_brand_preference` method to potentially update the brand preference based on the updated purchase history.
        """
        self.purchase_history.pop(0)  # Remove the oldest purchase history entry
        self.purchase_history.append(
            self.brand_choice
        )  # Add the current brand choice to the history
        self.update_brand_preference()  # Update the brand preference based on the new purchase history

    def step(self):
        """
        Defines the sequence of actions for the agent in each step of the simulation.

        This method orchestrates the agent's behavior in a single simulation step by calling a series of methods
        that simulate consumption, exposure to advertisements, price comparison, brand choice, and purchase decisions.
        """
        self.reset_step_values()  # Reset the step values at the beginning of each step.
        self.consume()  # Simulate the agent consuming products.
        if self.model.enable_ads:
            self.ad_exposure()  # Expose the agent to advertisements if enabled.
        if self.model.compare_brand_prices:
            self.compare_brand_prices()  # Compare brand prices if enabled.
        self.set_brand_choice()  # Set the agent's brand choice based on various factors.
        # self.reset_purchased_this_step()  # Reset the flag indicating if a purchase was made this step.
        self.get_step_min_and_max_units()  # Determine the minimum and maximum units the agent can purchase this step.
        self.get_baseline_units_to_purchase()  # Get the baseline number of units to purchase.
        if self.model.enable_elasticity:
            self.check_price()  # Check the current price of the brand.
            self.change_units_to_purchase_based_on_price()  # Adjust units to purchase based on price elasticity.
        if self.model.enable_ad_increment:
            self.change_units_to_purchase_based_on_adstock()  # Adjust units to purchase based on adstock effects.
        self.make_purchase()  # Execute the purchase action.
        self.update_purchase_history_and_preference()  # Update the purchase history and brand preference.
