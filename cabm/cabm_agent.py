# cabm/cabm_agent.py

import math
import mesa
import numpy as np
import logging

from cabm.config_helpers import Configuration
from cabm.agent_functions import (
    sample_normal_min,
    sample_beta_min,
    get_pantry_max,
    assign_media_channel_weights,
)
from cabm.signals import (
    AdEffects,
    PriceEffects,
    LoyaltyEffects,
    InventoryEffects,
    PurchaseProbabilityEngine,
)


class ConsumerAgent(mesa.Agent):
    """
    A single consumer/household in the model. Now refactored to rely
    on distinct "signal" classes + a "PurchaseProbabilityEngine."
    """

    def __init__(self, unique_id: int, model: mesa.Model, config: Configuration):
        super().__init__(unique_id, model)
        self.config = config

        # 1) basic household
        self.household_size = np.random.choice(
            config.household_sizes, p=config.household_size_distribution
        )
        self.consumption_rate = sample_normal_min(
            config.base_consumption_rate,
            override=config.consumption_rate_override,
        )

        # 2) brand preference
        self.brand_preference = np.random.choice(
            list(config.brand_market_share.keys()),
            p=list(config.brand_market_share.values()),
        )
        self.loyalty_rate = sample_beta_min(
            config.loyalty_alpha,
            config.loyalty_beta,
            override=config.loyalty_rate_override,
        )
        # purchase_history
        self.purchase_history_window_length = int(
            np.random.uniform(
                config.purchase_history_range_lower, config.purchase_history_range_upper
            )
        )
        self.purchase_history = [
            self.brand_preference
        ] * self.purchase_history_window_length

        # 3) ad preferences
        self.enable_ads = self.model.enable_ads
        self.ad_decay_factor = sample_normal_min(
            config.ad_decay_factor, override=config.ad_decay_override
        )
        self.ad_channel_preference = assign_media_channel_weights(
            list(config.channel_priors.keys()),
            list(config.channel_priors.values()),
        )
        # we store adstock as a brand->float dictionary
        self.adstock = {b: 1.0 for b in config.brand_list}

        # 4) pantry
        self.pantry_min = self.household_size * config.pantry_min_percent
        self.pantry_max = get_pantry_max(self.household_size, self.pantry_min)
        self.pantry_stock = self.pantry_max

        # 5) purchase toggles/counters
        self.step_min = 0
        self.step_max = 0
        self.baseline_units = 0
        self.incremental_promo_units = 0
        self.incremental_ad_units = 0
        self.decremental_units = 0
        self.units_to_purchase = 0
        self.purchased_this_step = {b: 0 for b in config.brand_list}
        self.current_price = config.brand_base_price[self.brand_preference]
        self.price_change = "no_price_change"

        # 6) price elasticity
        self.price_increase_sensitivity = config.price_increase_sensitivity
        self.price_decrease_sensitivity = config.price_decrease_sensitivity
        self.price_threshold = config.price_threshold

        # 7) create signal objects
        # AdEffects
        self.ad_effects = AdEffects(
            decay_factor=self.ad_decay_factor,
            incremental_sensitivity=config.adstock_incremental_sensitivity,
            incremental_midpoint=config.adstock_incremental_midpoint,
            incremental_limit=500.0,  # or config-based if desired
        )
        # PriceEffects
        self.price_effects = PriceEffects(
            base_prices=config.brand_base_price,
            sensitivity_increase=self.price_increase_sensitivity,
            sensitivity_decrease=self.price_decrease_sensitivity,
            threshold=self.price_threshold,
        )
        # LoyaltyEffects
        self.loyalty_effects = LoyaltyEffects(
            preferred_brand=self.brand_preference,
            loyalty_rate=self.loyalty_rate,
            all_brands=config.brand_list,
        )
        # InventoryEffects
        self.inventory_effects = InventoryEffects(
            pantry_min=self.pantry_min, pantry_max=self.pantry_max, k=3.0
        )
        # ProbabilityEngine with example weights
        self.signal_engine = PurchaseProbabilityEngine(
            signal_weights={
                "ad": 0.4,
                "price": 0.3,
                "loyalty": 0.2,
                "inventory": 0.1,
            }
        )

    def reset_step_values(self):
        """
        Reset counters each step so we don't carry them over.
        """
        self.baseline_units = 0
        self.incremental_promo_units = 0
        self.incremental_ad_units = 0
        self.decremental_units = 0
        self.units_to_purchase = 0
        self.purchased_this_step = {brand: 0 for brand in self.config.brand_list}

    def consume(self):
        """
        Simple consumption approach: reduce pantry by household_size / consumption_rate.
        """
        decrement = self.household_size / self.consumption_rate
        self.pantry_stock -= decrement
        if self.pantry_stock < 0:
            self.pantry_stock = 0

    def step(self):
        """
        The main step method that Mesa calls each "tick" or each "week."
        """
        self.reset_step_values()
        self.consume()

        # 1) Decay existing adstock, then update with this week's new adstock
        if self.model.enable_ads:
            self.adstock = self.ad_effects.decay(self.adstock)
            weekly_ad = self.ad_effects.compute_weekly_adstock(
                self.model.week_number,
                self.config.joint_calendar,
                self.config.brand_channel_map,
                self.ad_channel_preference,
            )
            self.adstock = self.ad_effects.update(self.adstock, weekly_ad)

        # 2) Gather signals from each factor:
        ad_signal = {}
        price_signal = {}
        loyalty_signal = {}
        inventory_signal = {}

        # If we want to incorporate brand-choice from ads:
        if self.model.enable_ads:
            ad_signal = self.ad_effects.compute_signal(
                self.adstock,
                self.loyalty_effects.preferred_brand,
                self.loyalty_effects.loyalty_rate,
            )
        else:
            # fallback: uniform
            ad_signal = {b: 1.0 for b in self.config.brand_list}

        # Price-based brand attractiveness:
        if self.model.compare_brand_prices:
            # gather current prices for each brand
            current_prices = {}
            for b in self.config.brand_list:
                current_prices[b] = self.config.joint_calendar.loc[
                    self.model.week_number, (b, "price")
                ]
            price_signal = self.price_effects.compute_signal(current_prices)
        else:
            # fallback: uniform
            price_signal = {b: 1.0 for b in self.config.brand_list}

        # loyalty
        loyalty_signal = self.loyalty_effects.compute_signal()

        # inventory
        inventory_signal = self.inventory_effects.compute_signal(
            pantry_stock=self.pantry_stock, brand_list=self.config.brand_list
        )

        # 3) Combine signals => final brand probabilities
        signals_map = {
            "ad": ad_signal,
            "price": price_signal,
            "loyalty": loyalty_signal,
            "inventory": inventory_signal,
        }
        self.purchase_probabilities = self.signal_engine.compute_final_probabilities(
            signals_map
        )

        # 4) Draw a brand choice
        brands = list(self.purchase_probabilities.keys())
        probs = list(self.purchase_probabilities.values())
        self.brand_choice = np.random.choice(brands, p=probs)

        # 5) Pantry logic => how many we can buy
        self.get_step_min_and_max_units()
        self.get_baseline_units_to_purchase()

        # 6) Price elasticity => check price
        if self.model.enable_elasticity:
            self.check_price()
            self.change_units_to_purchase_based_on_price()

        # 7) Ad increment => if adstock is high, buy more
        if self.model.enable_ad_increment:
            self.change_units_to_purchase_based_on_adstock()

        # 8) final purchase
        self.make_purchase()

        # 9) update purchase history and brand preference
        self.update_purchase_history_and_preference()

        # 10) update loyalty object with the brand we actually bought
        #     you could also pass some "ad_reinforcement" if you want
        self.loyalty_effects.update_loyalty(self.brand_choice, ad_reinforcement=0.0)

        # also keep the agent's "official" brand_preference/loyalty in sync
        self.brand_preference = self.loyalty_effects.preferred_brand
        self.loyalty_rate = self.loyalty_effects.loyalty_rate

    def get_step_min_and_max_units(self):
        """
        Minimal logic: step_min => how many needed to get back to at least pantry_min
        step_max => how many needed to fill up to pantry_max
        """
        needed_for_min = self.pantry_min - self.pantry_stock
        needed_for_max = self.pantry_max - self.pantry_stock

        self.step_min = max(0, math.ceil(needed_for_min))
        self.step_max = max(0, math.floor(needed_for_max))
        if self.step_min > self.step_max:
            self.step_min = 0
            self.step_max = 0

    def get_baseline_units_to_purchase(self):
        """
        Triangular distribution between [step_min..step_max],
        with mode at midpoint.
        """
        if self.step_max <= 0:
            return
        if self.step_min == self.step_max:
            self.baseline_units = self.step_min
            return

        mode = (self.step_min + self.step_max) / 2
        val = np.random.triangular(self.step_min, mode, self.step_max)
        self.baseline_units = int(round(val))
        if self.baseline_units > self.step_max:
            self.baseline_units = self.step_max
        if self.baseline_units < 0:
            self.baseline_units = 0

    def check_price(self):
        """
        Retrieve the brand_choice's current price from the joint calendar
        and label whether it's an increase or decrease relative to base.
        """
        b = self.brand_choice
        self.current_price = self.config.joint_calendar.loc[
            self.model.week_number, (b, "price")
        ]
        base_price = self.config.brand_base_price[b]

        if self.current_price < base_price:
            self.price_change = "price_decrease"
        elif self.current_price > base_price:
            self.price_change = "price_increase"
        else:
            self.price_change = "no_price_change"

    def change_units_to_purchase_based_on_price(self):
        """
        If brand price changed, we do a random chance to buy more/fewer.
        """
        prob = self.price_effects.probability_of_quantity_change(
            brand=self.brand_choice, current_price=self.current_price
        )
        # random branch
        event_branch = np.random.choice([True, False], p=[prob, 1 - prob])
        if not event_branch:
            return

        if self.price_change == "price_decrease":
            max_add = self.step_max - self.baseline_units
            if max_add > 0:
                self.incremental_promo_units = np.random.randint(1, max_add + 1)
        elif self.price_change == "price_increase":
            max_dec = self.baseline_units - self.step_min
            if max_dec > 0:
                self.decremental_units = np.random.randint(1, max_dec + 1)
        else:
            pass

    def change_units_to_purchase_based_on_adstock(self):
        """
        Probability that high adstock triggers extra units.
        """
        brand_adstock = self.adstock[self.brand_choice]
        prob = self.ad_effects.probability_of_incremental_purchase(brand_adstock)
        event_branch = np.random.choice([True, False], p=[prob, 1 - prob])
        if event_branch:
            max_add = self.step_max - self.baseline_units - self.incremental_promo_units
            if max_add > 0:
                self.incremental_ad_units = np.random.randint(1, max_add + 1)

    def make_purchase(self):
        """
        Finalize how many units are purchased from brand_choice.
        """
        proposed = (
            self.baseline_units
            + self.incremental_promo_units
            + self.incremental_ad_units
            - self.decremental_units
        )
        # clamp
        if proposed > self.step_max:
            proposed = self.step_max
        if proposed < 0:
            proposed = 0

        self.units_to_purchase = proposed
        self.purchased_this_step[self.brand_choice] = self.units_to_purchase
        self.pantry_stock += self.units_to_purchase

    def update_purchase_history_and_preference(self):
        """
        Maintain a rolling list of what brand we bought. If the entire
        window is a single brand, set brand_preference to that brand.
        """
        # remove oldest
        self.purchase_history.pop(0)
        # add newest
        self.purchase_history.append(self.brand_choice)

        # check if all same
        if len(set(self.purchase_history)) == 1:
            # all same brand
            new_pref = self.purchase_history[0]
            self.brand_preference = new_pref
