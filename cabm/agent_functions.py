import math
import warnings
import logging
import random
import numpy as np
import pandas as pd
from typing import List, Dict


# MATH FUNCTIONS


# A normal distribution sampler that avoids sampling below a set minimum
def sample_normal_min(mean, std_dev=1.0, min_value=1.0, override=None):
    """Sample from a normal distribution, rejecting values less than min_value.
    If override is specified, return only that value."""
    if override is not None:
        warnings.warn("Normal Sampler Override is in effect.")
        return override
    sample = np.random.normal(mean, std_dev)
    while sample < min_value:
        sample = np.random.normal(mean, std_dev)
    return sample


# A beta distributino sampler that avoids sampling below a set minimum or defaults always to override value
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


# Customized softmax for ad response and price point (inverse) response
def magnitude_adjusted_softmax(x: np.ndarray, inverse=False) -> np.ndarray:
    """Compute softmax values for each set of scores in x, with adjustments for magnitude."""
    try:
        # Handle the case where x is a list of zeros
        if np.all(x == 0):
            return np.full(x.shape, 1.0 / x.size)

        # Set temperature relative to max value if not overidden
        # Note this is critical to do before overflow prevention step -- need to test if it changes before log
        temperature = max(1, np.floor(np.log(np.max(x))))
        logging.debug(f"temperature = {temperature}")

        # Apply log transformation
        x = np.log1p(x)
        logging.debug(f"log transformed = {x}")

        # Subtract the max value to prevent overflow
        if inverse:
            x = np.max(x) - x
        else:
            x = x - np.max(x)
        logging.debug(f"overflow transform = {x}")

        e_x = np.exp(x / temperature)
        logging.debug(f"e_x = {e_x}")
        return e_x / np.sum(e_x)
    except ZeroDivisionError:
        print("Error: Division by zero in softmax.")
    except TypeError:
        print("Error: Input should be a numpy array.")
    except Exception as e:
        print(f"An unexpected error occurred in softmax: {e}")


# AGENT SETUP FUNCTIONS


def get_pantry_max(household_size, pantry_min):
    """
    Statistical assignment of maximum number of products a given household stocks
    Pantry min must be set before calling (default behavior of agent class)
    """
    try:
        pantry_max = math.ceil(np.random.normal(household_size, 1))
        if pantry_max < pantry_min:
            pantry_max = math.ceil(pantry_min)
        return pantry_max
    except Exception as e:
        print("An unexpected error occurred in get_pantry_max:", e)


def assign_weights(items: List[str], prior_weights: List[float]) -> Dict:
    """
    This function is used to randomize media channel preferences for each agent.

    Parameters:
    items (list): A list of items.
    prior_weights (list): A list of prior weights for the items.

    Returns:
    dict: A dictionary mapping items to their weights.
    """
    try:
        # Generate random fluctuations
        fluctuations = [random.random() for _ in items]

        # Apply fluctuations to prior weights
        weights = [w + f for w, f in zip(prior_weights, fluctuations)]

        # Normalize weights so they sum to 1
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        # Create a dictionary to map items to their weights
        weights_dict = dict(zip(items, weights))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return weights_dict


# ADVERTISING IMPACT FUNCTIONS


def calculate_adstock(
    week: int,
    joint_calendar: pd.DataFrame,
    brand_channel_map: Dict,
    channel_preference: Dict,
) -> Dict:
    """
    This function calculates the adstock for each brand.

    Parameters:
    week (int): The current week.
    joint_calendar (DataFrame): A DataFrame representing the joint calendar.
    brand_channel_map (dict): A dictionary mapping brands to their channels.
    channel_preference (dict): A dictionary mapping channels to their preference weights.

    Returns:
    dict: A dictionary mapping brands to their adstock.
    """
    adstock = {}
    try:
        for brand, channels in brand_channel_map.items():
            for channel in channels:
                spend = joint_calendar.loc[week, (brand, channel)]
                weighted_spend = spend * channel_preference[channel]
                # print(
                #     f"adstock calc for week {week} for brand {brand}, channel {channel}: spend = {spend}, weighted spend = {weighted_spend}"
                # )
                if brand in adstock:
                    adstock[brand] += weighted_spend
                else:
                    adstock[brand] = weighted_spend
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return adstock


def ad_decay(adstock: Dict, factor: float) -> Dict:
    """
    This function applies a decay factor to the adstock of each brand. If the resulting adstock is less than 1,
    it is set to 1.

    Parameters:
    adstock (dict): A dictionary mapping brands to their adstock.
    factor (float): The decay factor.

    Returns:
    dict: A dictionary mapping brands to their decayed adstock.
    """
    try:
        return {
            brand: (value / factor) if (value / factor) > 1 else 1
            for brand, value in adstock.items()
        }
    except RuntimeError:
        print(f"RuntimeError: current adstock: {adstock}, current factor: {factor}")
    except ZeroDivisionError:
        print("Error: Decay factor cannot be zero.")
    except Exception as e:
        print(
            f"An unexpected error occurred: {e}, current adstock: {adstock}, current factor: {factor}"
        )


def update_adstock(adstock1: Dict, adstock2: Dict) -> Dict:
    """
    This function updates the adstock of each brand by adding the values from a second adstock dictionary.

    Parameters:
    adstock1 (dict): The first adstock dictionary.
    adstock2 (dict): The second adstock dictionary.

    Returns:
    dict: A dictionary representing the updated adstock.
    """
    try:
        updated_adstock = adstock1.copy()
        for brand, value in adstock2.items():
            if brand in updated_adstock:
                updated_adstock[brand] += value
            else:
                updated_adstock[brand] = value
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return updated_adstock


def get_ad_impact_on_purchase_probabilities(
    adstock: Dict,
    brand_preference: str,
    loyalty_rate: float,
) -> Dict:
    """
    This function calculates the probability of purchasing each brand.

    Parameters:
    adstock (dict): A dictionary mapping brands to their adstock.
    brand_preference (str): The preferred brand.
    loyalty_rate (float): The loyalty rate.

    Returns:
    dict: A dictionary mapping brands to their purchase probabilities.
    """

    logging.debug(f"Using adstock: {adstock}")

    try:
        brands = list(adstock.keys())
        adstock_values = np.array(list(adstock.values()))

        # Softmax adstock to return normalized probability distribution
        transformed_adstock = magnitude_adjusted_softmax(adstock_values)

        logging.debug(f"magnitude adjusted softmax adstock: {transformed_adstock}")

        # Initialize base probabilities with equal chance for non-preferred brands
        base_probabilities = np.full_like(
            transformed_adstock, (1 - loyalty_rate) / (len(brands) - 1)
        )

        logging.debug(f"first pass base probabilities: {base_probabilities}")

        brand_preference_index = brands.index(brand_preference)
        base_probabilities[brand_preference_index] = loyalty_rate

        logging.debug(f"second pass base probabilities: {base_probabilities}")

        adjusted_probabilities = transformed_adstock * base_probabilities

        logging.debug(f"unnormalized adjusted probabilities: {adjusted_probabilities}")

        # Normalize the adjusted probabilities so they sum to 1
        probabilities = adjusted_probabilities / np.sum(adjusted_probabilities)

        logging.debug(f"normalized probabilities: {probabilities}")

        return dict(zip(brands, probabilities))
    except ZeroDivisionError:
        print("Error: Division by zero.")
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# PRICE IMPACT FUNCTIONS


def get_current_price(week, joint_calendar, brand):
    price = joint_calendar.loc[week, (brand, "price")]
    return price


def get_price_impact_on_purchase_probabilities(
    week_number,
    joint_calendar,
    brand_preference: str,
    loyalty_rate: float,
) -> Dict:
    """
    This function calculates the probability of purchasing each brand.

    Parameters:
    adstock (dict): A dictionary mapping brands to their price.
    brand_preference (str): The preferred brand.
    loyalty_rate (float): The loyalty rate.

    Returns:
    dict: A dictionary mapping brands to their purchase probabilities.
    """
    price_list = {}

    try:
        for brand in joint_calendar.columns:
            price_list[brand] = joint_calendar.loc[week_number, brand]
    except Exception as e:
        print(f"Could not generate price list: {e}")

    logging.debug(f"Using pricelist: {price_list}")

    try:
        brands = list(price_list.keys())
        price_list_values = np.array(list(price_list.values()))

        # Softmax adstock to return normalized probability distribution
        transformed_price_list = magnitude_adjusted_softmax(
            price_list_values, inverse=True
        )

        logging.debug(
            f"Inverse magnitude adjusted softmax price_list: {transformed_price_list}"
        )

        # Initialize base probabilities with equal chance for non-preferred brands
        base_probabilities = np.full_like(
            transformed_price_list, (1 - loyalty_rate) / (len(brands) - 1)
        )

        brand_preference_index = brands.index(brand_preference)
        base_probabilities[brand_preference_index] = loyalty_rate

        logging.debug(f"base pricing probabilities: {base_probabilities}")

        adjusted_probabilities = transformed_price_list * base_probabilities

        # Normalize the adjusted probabilities so they sum to 1
        probabilities = adjusted_probabilities / np.sum(adjusted_probabilities)

        logging.debug(f"normalized pricing probabilities: {probabilities}")

        return dict(zip(brands, probabilities))
    except ZeroDivisionError:
        print("Error: Division by zero.")
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
