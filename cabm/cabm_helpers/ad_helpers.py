import random
import logging
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict

logger = logging.getLogger(__name__)


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
    This function applies a decay factor to the adstock of each brand.

    Parameters:
    adstock (dict): A dictionary mapping brands to their adstock.
    factor (float): The decay factor.

    Returns:
    dict: A dictionary mapping brands to their decayed adstock.
    """
    try:
        return {brand: value / factor for brand, value in adstock.items()}
    except RuntimeError:
        logger.error(f"current adstock: {adstock}")
        logger.error(f"current adstock: {factor}")
    except ZeroDivisionError:
        print("Error: Decay factor cannot be zero.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logger.error(f"current adstock: {adstock}")
        logger.error(f"current adstock: {factor}")


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


def get_purchase_probabilities(
    adstock: Dict, preferred_brand: str, loyalty_rate: float, sensitivity: float
) -> Dict:
    """
    This function calculates the probability of purchasing each brand.

    Parameters:
    adstock (dict): A dictionary mapping brands to their adstock.
    preferred_brand (str): The preferred brand.
    loyalty_rate (float): The loyalty rate.
    sensitivity (float): The sensitivity of the probabilities to the adstock values.

    Returns:
    dict: A dictionary mapping brands to their purchase probabilities.
    """
    warnings.warn("WARNING: YOU ARE USING A MODIFIED PURCH PROB GETTER")
    try:
        brands = list(adstock.keys())
        adstock_values = np.array(list(adstock.values()))

        # Transform adstock values using a logarithm to reduce the impact of large differences
        transformed_adstock = adstock_values

        warnings.warn("transformed_adstock not actually transformed")

        # Normalize the transformed adstock values so they sum to 1
        sum_transformed_adstock = np.sum(transformed_adstock)
        if sum_transformed_adstock == 0:
            # If the sum is zero, set normalized_adstock to a default value (identity -- all 1's)
            normalized_adstock = np.ones_like(transformed_adstock)
        else:
            normalized_adstock = transformed_adstock / sum_transformed_adstock

        # print(f"normalized adstock: {normalized_adstock}")

        # Initialize base probabilities with equal chance for non-preferred brands
        base_probabilities = np.full_like(
            normalized_adstock, (1 - loyalty_rate) / (len(brands) - 1)
        )

        # print(f"first pass base probabilities: {base_probabilities}")

        preferred_brand_index = brands.index(preferred_brand)
        base_probabilities[preferred_brand_index] = loyalty_rate

        # print(f"second pass base probabilities: {base_probabilities}")

        # Adjust probabilities based on adstock and sensitivity
        # adjusted_probabilities = base_probabilities * (
        #     1 + sensitivity * normalized_adstock
        # )
        # # print("Updating purch probs by multiplying base prob times adstock probs")
        adjusted_probabilities = normalized_adstock * base_probabilities

        # print(f"unnormalized adjusted probabilities: {adjusted_probabilities}")

        # Normalize the adjusted probabilities so they sum to 1
        probabilities = adjusted_probabilities / np.sum(adjusted_probabilities)

        # print(f"normalized probabilities: {probabilities}")

        return dict(zip(brands, probabilities))
    except ZeroDivisionError:
        print("Error: Division by zero.")
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# def get_purchase_probabilities(
#     adstock: Dict, preferred_brand: str, loyalty_rate: float, sensitivity: float
# ) -> Dict:
#     """
#     This function calculates the probability of purchasing each brand.

#     Parameters:
#     adstock (dict): A dictionary mapping brands to their adstock.
#     preferred_brand (str): The preferred brand.
#     loyalty_rate (float): The loyalty rate.
#     sensitivity (float): The sensitivity of the probabilities to the adstock values.

#     Returns:
#     dict: A dictionary mapping brands to their purchase probabilities.
#     """
#     try:
#         brands = list(adstock.keys())
#         adstock_values = np.array(list(adstock.values()))

#         # Transform adstock values using a logarithm to reduce the impact of large differences
#         # transformed_adstock = np.log1p(adstock_values)
#         warnings.warn("ADSTOCK TRANSFORMATION DISABLED")
#         transformed_adstock = adstock_values

#         # Normalize the transformed adstock values so they sum to 1
#         sum_transformed_adstock = np.sum(transformed_adstock)
#         if sum_transformed_adstock == 0:
#             # If the sum is zero, set normalized_adstock to a default value
#             normalized_adstock = np.zeros_like(transformed_adstock)
#         else:
#             normalized_adstock = transformed_adstock / sum_transformed_adstock

#         # Initialize base probabilities with equal chance for non-preferred brands
#         base_probabilities = np.full_like(
#             normalized_adstock, (1 - loyalty_rate) / (len(brands) - 1)
#         )
#         preferred_brand_index = brands.index(preferred_brand)
#         base_probabilities[preferred_brand_index] = loyalty_rate

#         # Adjust probabilities based on adstock and sensitivity
#         adjusted_probabilities = base_probabilities * (
#             1 + sensitivity * normalized_adstock
#         )

#         # Normalize the adjusted probabilities so they sum to 1
#         probabilities = adjusted_probabilities / np.sum(adjusted_probabilities)

#         return dict(zip(brands, probabilities))
#     except ZeroDivisionError:
#         print("Error: Division by zero.")
#     except KeyError as e:
#         print(f"KeyError: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

#     except ZeroDivisionError:
#         print("Error: Division by zero.")
#     except KeyError as e:
#         print(f"KeyError: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")