import random
from typing import List, Dict
import pandas as pd


def generate_brand_ad_channel_map(brand_list: List[str], config: Dict) -> Dict:
    """
    This function generates a map of brands to their advertising channels.

    Parameters:
    brand_list (list): A list of brands.
    config (dict): A dictionary containing the configuration settings.

    Returns:
    dict: A dictionary mapping brands to their advertising channels.
    """
    brand_ad_channel_map = {}
    try:
        for brand in brand_list:
            brand_ad_channel_map[brand] = config["brands"][brand]["advertising"][
                "channels"
            ]
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return brand_ad_channel_map


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
    This function applies a decay factor to the adstock of each brand.

    Parameters:
    adstock (dict): A dictionary mapping brands to their adstock.
    factor (float): The decay factor.

    Returns:
    dict: A dictionary mapping brands to their decayed adstock.
    """
    try:
        return {brand: value / factor for brand, value in adstock.items()}
    except ZeroDivisionError:
        print("Error: Decay factor cannot be zero.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


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


def get_switch_probability(
    adstock: Dict, preferred_brand: str, default_loyalty_rate: float
) -> Dict:
    """
    This function calculates the probability of switching to each brand.

    Parameters:
    adstock (dict): A dictionary mapping brands to their adstock.
    preferred_brand (str): The preferred brand.
    default_loyalty_rate (float): The default loyalty rate.

    Returns:
    dict: A dictionary mapping brands to their switch probabilities.
    """
    try:
        brands = list(adstock.keys())
        adstock_values = list(adstock.values())

        if adstock[preferred_brand] > max(adstock_values):
            return {brand: 1 if brand == preferred_brand else 0 for brand in brands}

        elif sum(adstock_values) == 0:
            probabilities = {
                brand: default_loyalty_rate
                if brand == preferred_brand
                else (1 - default_loyalty_rate) / (len(brands) - 1)
                for brand in brands
            }
            return probabilities

        else:
            total_adstock = sum(adstock_values)
            probabilities = {
                brand: value / total_adstock for brand, value in adstock.items()
            }
            return probabilities
    except ZeroDivisionError:
        print("Error: Division by zero.")
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
