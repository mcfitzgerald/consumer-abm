# cabm/signals.py

import numpy as np
import pandas as pd
from typing import Dict, List

# Optional, only if you want to do quick debug prints:
import logging

logging.basicConfig(level=logging.INFO)


def logistic_function(x: float) -> float:
    """
    Simple logistic function to map any real-valued number into the range (0, 1).
    """
    return 1 / (1 + np.exp(-x))


class AdEffects:
    """
    Encapsulates advertising (adstock) effects:
      - Decays existing adstock each step
      - Updates with weekly adstock
      - Converts final adstock levels into a brand-level probability signal
      - Computes incremental purchase probability if adstock is high
    """

    def __init__(
        self,
        decay_factor: float,
        incremental_sensitivity: float,
        incremental_midpoint: float,
        incremental_limit: float,
    ):
        """
        :param decay_factor: Divisor for existing adstock each week. Higher => faster decay.
        :param incremental_sensitivity: Steepness of logistic function used to
                                        compute probability of incremental purchase.
        :param incremental_midpoint: Adstock level at which the probability of
                                     incremental purchase is 0.5.
        :param incremental_limit: If adstock <= this limit, no incremental purchase is triggered.
        """
        self.decay_factor = decay_factor
        self.incremental_sensitivity = incremental_sensitivity
        self.incremental_midpoint = incremental_midpoint
        self.incremental_limit = incremental_limit

    def decay(self, current_adstock: Dict[str, float]) -> Dict[str, float]:
        """
        Apply decay to each brand’s adstock; ensure a floor of 1.0 so it never goes below 1.
        """
        decayed = {}
        for brand, value in current_adstock.items():
            new_val = value / self.decay_factor
            decayed[brand] = new_val if new_val > 1.0 else 1.0
        return decayed

    def compute_weekly_adstock(
        self,
        week: int,
        joint_calendar: pd.DataFrame,
        brand_channel_map: Dict[str, List[str]],
        channel_preference: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Computes the new adstock for the current week, for each brand, by summing
        weighted spend across that brand's channels.
        """
        weekly_adstock = {}
        for brand, channels in brand_channel_map.items():
            total_brand_ad = 0.0
            for ch in channels:
                spend = joint_calendar.loc[week, (brand, ch)]
                weight = channel_preference.get(ch, 1.0)
                total_brand_ad += spend * weight
            weekly_adstock[brand] = total_brand_ad
        return weekly_adstock

    def update(
        self, current_adstock: Dict[str, float], weekly_adstock: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Adds weekly brand adstock to the existing (already decayed) brand adstock.
        """
        updated = current_adstock.copy()
        for brand, weekly_val in weekly_adstock.items():
            updated[brand] = updated.get(brand, 0.0) + weekly_val
        return updated

    def compute_signal(
        self,
        adstock: Dict[str, float],
        preferred_brand: str,
        loyalty_rate: float,
    ) -> Dict[str, float]:
        """
        Converts adstock into a brand-level "attention" signal, then merges it
        with a loyalty-based prior. The final probabilities are normalized.
        """
        if not adstock:
            return {}

        brands = list(adstock.keys())
        ad_values = np.array(list(adstock.values()))

        # 1) magnitude-adjusted softmax transform of ad values
        transformed = self._magnitude_adjusted_softmax(ad_values)

        # 2) base loyalty vector
        #    This gives loyalty_rate probability to the preferred brand,
        #    and divides (1 - loyalty_rate) among others.
        base_prob = np.full(len(brands), (1 - loyalty_rate) / (len(brands) - 1))
        pref_idx = brands.index(preferred_brand)
        base_prob[pref_idx] = loyalty_rate

        # 3) Multiply elementwise => combined, then normalize
        combined = transformed * base_prob
        if combined.sum() <= 0:
            # fallback
            combined = np.full(len(brands), 1.0 / len(brands))
        else:
            combined /= combined.sum()

        return dict(zip(brands, combined))

    def probability_of_incremental_purchase(self, brand_adstock: float) -> float:
        """
        Computes the likelihood that an agent will purchase additional units
        if the brand's adstock is above a certain limit. Uses a log10-based logistic curve.
        """
        if brand_adstock <= self.incremental_limit:
            return 0.0
        # logistic on log10 scale
        return 1 / (
            1
            + np.exp(
                -self.incremental_sensitivity
                * (np.log10(brand_adstock) - np.log10(self.incremental_midpoint))
            )
        )

    def _magnitude_adjusted_softmax(
        self, x: np.ndarray, log_transform: bool = True, inverse: bool = False
    ) -> np.ndarray:
        """
        A magnitude-based softmax:
          - Optionally applies log1p
          - Shift by max to reduce overflow
          - Divide by "temperature" ~ log(max(x)+1)
        """
        if np.all(x <= 0):
            # fallback: uniform distribution
            return np.full(x.shape, 1.0 / len(x))

        # temperature
        max_val = np.max(x)
        temperature = max(1.0, np.log(max_val + 1.0))

        # transform
        if log_transform:
            x = np.log1p(x)
        if inverse:
            x = np.max(x) - x
        else:
            x = x - np.max(x)

        e_x = np.exp(x / temperature)
        return e_x / e_x.sum()


class PriceEffects:
    """
    Computes a "price attractiveness" signal for each brand, relative to
    the brand's base price. Also can produce a probability that purchase
    quantity changes if the brand's price is above/below the base.
    """

    def __init__(
        self,
        base_prices: Dict[str, float],
        sensitivity_increase: float,
        sensitivity_decrease: float,
        threshold: float,
    ):
        """
        :param base_prices: reference (normal) prices for each brand
        :param sensitivity_increase: logistic steepness for price above base
        :param sensitivity_decrease: logistic steepness for price below base
        :param threshold: price difference below which we do not trigger big changes
        """
        self.base_prices = base_prices
        self.sensitivity_increase = sensitivity_increase
        self.sensitivity_decrease = sensitivity_decrease
        self.threshold = threshold

    def compute_signal(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        For each brand, compute a "price signal" for brand-choice probabilities:
         - Price below base => >1.0 signal (stronger attractiveness)
         - Price above base => <1.0 signal (weaker attractiveness)
         - Near base => ~ 1.0
        """
        signals = {}
        for brand, base_price in self.base_prices.items():
            cprice = current_prices.get(brand, base_price)
            if base_price <= 0:
                signals[brand] = 1.0
                continue

            pct_change = (cprice - base_price) / base_price

            if abs(pct_change) < self.threshold:
                signals[brand] = 1.0
            elif pct_change < 0:  # discount
                # logistic function => a factor > 1
                signals[brand] = 1.0 + (
                    1.0 / (1.0 + np.exp(-self.sensitivity_decrease * abs(pct_change)))
                )
            else:  # price increase
                # logistic function => factor < 1
                signals[brand] = 1.0 / (
                    1.0 + np.exp(self.sensitivity_increase * pct_change)
                )

        return signals

    def probability_of_quantity_change(self, brand: str, current_price: float) -> float:
        """
        Probability that the agent changes units purchased (incremental or decremental)
        based on how far the current_price is from brand's base price.
        """
        base_price = self.base_prices.get(brand, current_price)
        if base_price <= 0:
            return 0.0

        pct_change = (current_price - base_price) / base_price

        if abs(pct_change) < self.threshold:
            return 0.0

        # if discount => logistic on negative side => higher prob
        if pct_change < 0:
            return logistic_function(abs(pct_change) * self.sensitivity_decrease)
        else:
            return logistic_function(abs(pct_change) * self.sensitivity_increase)


class LoyaltyEffects:
    """
    Provides a brand-specific loyalty signal. The agent typically
    has a single `preferred_brand` and a `loyalty_rate` in [0,1].
    You can also store logic to update the preference if brand-switching
    persists, or do so externally in the agent.
    """

    def __init__(
        self, preferred_brand: str, loyalty_rate: float, all_brands: List[str]
    ):
        self.preferred_brand = preferred_brand
        self.loyalty_rate = loyalty_rate
        self.all_brands = all_brands

    def compute_signal(self) -> Dict[str, float]:
        """
        Produces a brand-level factor that "boosts" the preferred brand
        by (1 + loyalty_rate). Others get 1.0
        """
        signals = {b: 1.0 for b in self.all_brands}
        if self.preferred_brand in signals:
            signals[self.preferred_brand] += self.loyalty_rate
        return signals

    def update_loyalty(self, chosen_brand: str, ad_reinforcement: float = 0.0):
        """
        Example logic:
         - If the agent purchased the same brand as the preference, loyalty goes up.
         - If the agent purchased a different brand, loyalty goes down. Possibly switch if too low.
        """
        if chosen_brand == self.preferred_brand:
            # reward
            self.loyalty_rate = min(1.0, self.loyalty_rate + 0.1 + ad_reinforcement)
        else:
            # reduce
            self.loyalty_rate = max(0.0, self.loyalty_rate - 0.05)
            if self.loyalty_rate < 0.3:
                self.preferred_brand = chosen_brand
                self.loyalty_rate = 0.5


class InventoryEffects:
    """
    Computes an "urgency" signal based on pantry stock. If the agent
    is near or below the pantry_min, the signal is high. If near pantry_max,
    the signal is low.
    """

    def __init__(self, pantry_min: float, pantry_max: float, k: float = 3.0):
        self.pantry_min = pantry_min
        self.pantry_max = pantry_max
        self.k = k

    def compute_signal(
        self, pantry_stock: float, brand_list: List[str]
    ) -> Dict[str, float]:
        """
        Returns the same urgency factor for all brands, in [0..1].
        One common approach is a logistic transform that yields ~1 at low stock.
        """
        if self.pantry_max <= self.pantry_min:
            # fallback if something is off
            return {b: 0.5 for b in brand_list}

        # normalize the difference
        diff = (pantry_stock - self.pantry_min) / (self.pantry_max - self.pantry_min)
        # logistic => invert so that lower stock => bigger factor
        # we shift by 0.5 so that stock halfway from min->max => ~0.5 signal
        # larger self.k => sharper transition
        urgency = 1.0 / (1.0 + np.exp(self.k * (diff - 0.5)))

        return {b: urgency for b in brand_list}


class PurchaseProbabilityEngine:
    """
    Aggregates signals from multiple factors (ad, price, loyalty, inventory, etc.)
    to produce final purchase probabilities. Each factor is a dict[brand] => float.
    We multiply them with weights or simply multiply them, then softmax.

    This is a flexible approach: you can choose additive or multiplicative.
    Here we do a weighted additive combination, then exponentiate & normalize.
    """

    def __init__(self, signal_weights: Dict[str, float]):
        """
        :param signal_weights: e.g. {"ad": 0.4, "price": 0.3, "loyalty": 0.2, "inventory": 0.1}
        """
        self.signal_weights = signal_weights

    def compute_final_probabilities(
        self, signal_map: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Steps:
          1. For each brand, combine signals as a weighted sum of logs (or a direct product).
          2. Exponentiate to get a positive number.
          3. Normalize across brands => final probabilities.
        """
        if not signal_map:
            return {}

        # All signals should share the same brand keys
        brands = list(next(iter(signal_map.values())).keys())
        brand_scores = {b: 0.0 for b in brands}

        # Weighted "log" approach or direct approach. Let's do a direct multiplicative approach
        #   combined_score(brand) = Π (signal(brand)^weight)
        #   or we take sum of w * log(signal)
        # Then exponentiate. This is effectively a "log-linear" weighting scheme
        for signal_name, brand_dict in signal_map.items():
            weight = self.signal_weights.get(signal_name, 1.0)
            for b in brands:
                val = brand_dict.get(b, 1.0)
                # if val <= 0 => fallback to small positive
                if val <= 0:
                    val = 1e-6
                # sum of logs approach
                brand_scores[b] += weight * np.log(val)

        # Exponentiate
        for b in brands:
            brand_scores[b] = np.exp(brand_scores[b])

        # Normalize
        total = sum(brand_scores.values())
        if total <= 0:
            # fallback
            return {b: 1.0 / len(brands) for b in brands}
        return {b: brand_scores[b] / total for b in brands}
