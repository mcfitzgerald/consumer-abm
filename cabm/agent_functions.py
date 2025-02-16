import math
import warnings
import logging
import random
import numpy as np
import pandas as pd
from typing import List, Dict


# MATH FUNCTIONS


def sample_normal_min(
    mean: float, std_dev: float = 1.0, min_value: float = 1.0, override: float = 0
) -> float:
    """
    Function to sample from a normal distribution, rejecting values less than min_value.
    If override is specified, return only that value.

    Parameters:
    mean (float): The mean of the normal distribution.
    std_dev (float): The standard deviation of the normal distribution. Default is 1.0.
    min_value (float): The minimum value that can be sampled. Default is 1.0.
    override (float): If specified, this value will be returned instead of a sample. Default is None.

    Returns:
    float: A sample from the normal distribution that is greater than or equal to min_value, or the override value if specified.
    """
    # If override is specified, issue a warning and return the override value
    if override != 0:
        warnings.warn("Normal Sampler Override is in effect.")
        return override

    # Sample from the normal distribution
    sample = np.random.normal(mean, std_dev)

    # If the sample is less than min_value, continue sampling until a valid sample is obtained
    while sample < min_value:
        sample = np.random.normal(mean, std_dev)

    # Return the valid sample
    return sample


def sample_beta_min(
    alpha: float, beta: float, min_value: float = 0.05, override: float = 0
) -> float:
    """
    Function to sample from a beta distribution, rejecting values less than min_value.
    If override is specified, return only that value.

    Parameters:
    alpha (float): The alpha parameter of the beta distribution.
    beta (float): The beta parameter of the beta distribution.
    min_value (float): The minimum value that can be sampled. Default is 0.05.
    override (float): If specified, this value will be returned instead of a sample. Default is None.

    Returns:
    float: A sample from the beta distribution that is greater than or equal to min_value, or the override value if specified.
    """
    # If override is specified, issue a warning and return the override value
    if override != 0:
        warnings.warn("Beta Sampler Override is in effect.")
        return override

    # Sample from the beta distribution
    sample = np.random.beta(alpha, beta)

    # If the absolute value of the sample is less than min_value, continue sampling until a valid sample is obtained
    while abs(sample) < min_value:
        sample = np.random.beta(alpha, beta)

    # Return the valid sample
    return sample


def logistic_function(x: float) -> float:
    """
    Logistic function to map any real-valued number into the range (0, 1).

    Parameters:
    x (float): The input value.

    Returns:
    float: The output of the logistic function.
    """
    return 1 / (1 + math.exp(-x))


def magnitude_adjusted_softmax(
    x: np.ndarray,
    log_transform: bool = True,
    inverse: bool = False,
    magnitude_adjusted_temperature=True,
) -> np.ndarray:
    """
    Compute softmax values for each set of scores in x, with adjustments for magnitude.

    Parameters:
    x (np.ndarray): Input numpy array for which softmax is to be computed.
    log_transform (bool): If True, applies log transformation to the input array. Default is True.
    inverse (bool): If True, subtracts the input array from its max value to prevent overflow. Default is False.

    Returns:
    np.ndarray: A numpy array of the same shape as x, where each element is the softmax of the corresponding element in x.
    """
    try:
        # If all elements in x are zero, return an array of the same shape where each element is 1 divided by the number of elements in x
        if np.all(x == 0):
            return np.full(x.shape, 1.0 / x.size)

        # Set temperature relative to max value in x. This is done before the overflow prevention step
        if magnitude_adjusted_temperature:
            temperature = max(1, np.log(np.max(x)))
            logging.debug(f"Temperature for softmax calculation: {temperature}")
        else:
            temperature = 1

        # If log_transform is True, apply log transformation to x
        if log_transform:
            x = np.log1p(x)
            logging.debug(f"Log transformed input: {x}")
        else:
            logging.debug("No log transformation applied to input.")

        # If inverse is True, subtract x from its max value to prevent overflow
        if inverse:
            x = np.max(x) - x
        else:
            x = x - np.max(x)
        logging.debug(f"Input after overflow prevention: {x}")

        # Compute softmax values
        e_x = np.exp(x / temperature)
        logging.debug(f"Softmax values: {e_x}")

    except ZeroDivisionError:
        print("Error: Division by zero in softmax calculation.")
    except TypeError:
        print("Error: Input should be a numpy array.")
    except Exception as e:
        print(f"An unexpected error occurred in softmax calculation: {e}")
    return e_x / np.sum(e_x)


# AGENT SETUP FUNCTIONS


def get_pantry_max(household_size: int, pantry_min: int) -> int:
    """
    This function statistically assigns the maximum number of products a given household stocks.
    The pantry minimum must be set before calling this function (default behavior of agent class).

    Parameters:
    household_size (int): The size of the household.
    pantry_min (int): The minimum number of products in the pantry.

    Returns:
    int: The maximum number of products a given household stocks.
    """
    try:
        # Generate a random number from a normal distribution with mean equal to household_size and standard deviation 1
        pantry_max = math.ceil(np.random.normal(household_size, 1))

        # If the generated pantry_max is less than pantry_min, set pantry_max to pantry_min
        if pantry_max < pantry_min:
            pantry_max = math.ceil(pantry_min)

    except Exception as e:
        print("An unexpected error occurred in get_pantry_max:", e)
    return pantry_max


def assign_media_channel_weights(
    items: List[str], prior_weights: List[float]
) -> Dict[str, float]:
    """
    For example, if:
    items = ['Web', 'TV']
    prior_weights = [0.7, 0.3]  # The desired 70/30 split
    """

    # Generate fluctuations between -0.1 and 0.1 for each channel
    # (random.random() gives 0-1, so (random.random() - 0.5) gives -0.5 to 0.5
    # multiply by 0.2 to get -0.1 to 0.1)
    fluctuations = [(random.random() - 0.5) * 0.2 for _ in items]
    # Example: fluctuations might be [0.05, -0.03]

    # Apply fluctuations multiplicatively (1 + fluctuation)
    # For Web (0.7): 0.7 * (1 + 0.05) = 0.735
    # For TV (0.3): 0.3 * (1 - 0.03) = 0.291
    weights = [max(0.001, w * (1 + f)) for w, f in zip(prior_weights, fluctuations)]
    # The max() ensures no negative weights

    # Normalize to ensure sum = 1
    # Sum = 0.735 + 0.291 = 1.026
    # Web: 0.735/1.026 = 0.716 (≈70%)
    # TV: 0.291/1.026 = 0.284 (≈30%)
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    return dict(zip(items, weights))


# ADVERTISING IMPACT FUNCTIONS


def calculate_adstock(
    week: int,
    joint_calendar: pd.DataFrame,
    brand_channel_map: Dict[str, List[str]],
    channel_preference: Dict[str, float],
) -> Dict[str, float]:
    """
    This function calculates the adstock for each brand. Adstock is the total weighted spend for each brand.

    Parameters:
    week (int): The current week.
    joint_calendar (pd.DataFrame): A DataFrame representing the joint calendar. The joint calendar contains the spend for each brand and channel for each week.
    brand_channel_map (Dict[str, List[str]]): A dictionary mapping brands to their channels. Each brand can have multiple channels.
    channel_preference (Dict[str, float]): A dictionary mapping channels to their preference weights. The preference weight is used to weight the spend for each channel.

    Returns:
    Dict[str, float]: A dictionary mapping brands to their adstock. The adstock for each brand is the sum of the weighted spend for all its channels.
    """
    adstock: Dict[str, float] = {}
    try:
        # Loop over each brand and its channels
        for brand, channels in brand_channel_map.items():
            # Loop over each channel for the current brand
            for channel in channels:
                # Get the spend for the current brand and channel for the current week
                spend = joint_calendar.loc[week, (brand, channel)]
                # Calculate the weighted spend by multiplying the spend by the preference weight for the current channel
                weighted_spend = spend * channel_preference[channel]
                # If the brand is already in the adstock dictionary, add the weighted spend to its current adstock
                if brand in adstock:
                    adstock[brand] += weighted_spend
                # If the brand is not in the adstock dictionary, add it to the dictionary with the weighted spend as its adstock
                else:
                    adstock[brand] = weighted_spend
    except KeyError as e:
        print(
            f"KeyError: {e}. Check if the brand and channel exist in the joint calendar and the channel exists in the channel preference."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return adstock


def decay_adstock(adstock: Dict[str, float], factor: float) -> Dict[str, float]:
    """
    This function applies a decay factor to the adstock of each brand. If the resulting adstock is less than 1,
    it is set to 1.

    Parameters:
    adstock (Dict[str, float]): A dictionary mapping brands to their adstock.
    factor (float): The decay factor.

    Returns:
    Dict[str, float]: A dictionary mapping brands to their decayed adstock.
    """
    try:
        # Apply decay factor to each brand's adstock
        # If the resulting adstock is less than 1, set it to 1
        decayed_adstock = {
            brand: (value / factor) if (value / factor) > 1.0 else 1.0
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
    return decayed_adstock


def update_adstock(
    adstock1: Dict[str, float], adstock2: Dict[str, float]
) -> Dict[str, float]:
    """
    This function updates the adstock of each brand by adding the values from a second adstock dictionary.
    If a brand is present in both dictionaries, the values are added. If a brand is only present in the second dictionary,
    it is added to the updated adstock dictionary with its value.

    Parameters:
    adstock1 (Dict[str, float]): The first adstock dictionary mapping brands to their adstock.
    adstock2 (Dict[str, float]): The second adstock dictionary mapping brands to their adstock.

    Returns:
    Dict[str, float]: A dictionary representing the updated adstock mapping brands to their updated adstock.
    """
    try:
        # Create a copy of the first adstock dictionary to avoid modifying the original
        updated_adstock = adstock1.copy()
        # Iterate over each brand and its adstock in the second adstock dictionary
        for brand, value in adstock2.items():
            # If the brand is already in the updated adstock dictionary, add the value from the second adstock dictionary
            if brand in updated_adstock:
                updated_adstock[brand] += value
            # If the brand is not in the updated adstock dictionary, add it with its value from the second adstock dictionary
            else:
                updated_adstock[brand] = value
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    # Return the updated adstock dictionary
    return updated_adstock


def get_probability_of_change_in_units_purchased_due_to_adstock(
    adstock: float,
    sensitivity: float = 1,
    midpoint: float = 1000,
    limit: float = 500,
) -> float:
    """
    This function calculates the probability of an additional purchase based on the adstock value.

    Parameters:
    adstock (float): The adstock value.
    sensitivity (float): The sensitivity factor for adstock. Default is 1.
    midpoint (float): The midpoint of the logistic curve. Default is 1000.
    limit (float): The adstock belowwhich the probability is 0. Default is 500.

    Returns:
    float: The probability of an additional purchase.
    """
    # If adstock is less than or equal to the limit, return 0
    if adstock <= limit:
        return 0.0

    # Use the logistic function to model the probability
    probability = 1 / (
        1 + np.exp(-sensitivity * (np.log10(adstock) - np.log10(midpoint)))
    )

    return probability


def get_ad_impact_on_purchase_probabilities(
    adstock: Dict[str, float],
    brand_preference: str,
    loyalty_rate: float,
) -> Dict[str, float]:
    """
    This function calculates the probability of purchasing each brand based on the adstock, brand preference, and loyalty rate.

    Parameters:
    adstock (Dict[str, float]): A dictionary mapping brands to their adstock.
    brand_preference (str): The preferred brand.
    loyalty_rate (float): The loyalty rate.

    Returns:
    Dict[str, float]: A dictionary mapping brands to their purchase probabilities.
    """

    logging.debug(f"Using adstock: {adstock}")

    try:
        # Extract brands and their corresponding adstock values
        brands = list(adstock.keys())
        adstock_values = np.array(list(adstock.values()))

        # Apply softmax transformation to adstock values to get a normalized probability distribution
        transformed_adstock = magnitude_adjusted_softmax(adstock_values)

        logging.debug(
            f"Transformed adstock using magnitude adjusted softmax: {transformed_adstock}"
        )

        # Initialize base probabilities with equal chance for non-preferred brands
        base_probabilities = np.full_like(
            transformed_adstock, (1 - loyalty_rate) / (len(brands) - 1)
        )

        logging.debug(f"Base probabilities after first pass: {base_probabilities}")

        # Update the base probability of the preferred brand with the loyalty rate
        brand_preference_index = brands.index(brand_preference)
        base_probabilities[brand_preference_index] = loyalty_rate

        logging.debug(f"Base probabilities after second pass: {base_probabilities}")

        # Adjust the base probabilities with the transformed adstock values
        adjusted_probabilities = transformed_adstock * base_probabilities

        logging.debug(
            f"Adjusted probabilities before normalization: {adjusted_probabilities}"
        )

        # Normalize the adjusted probabilities so they sum to 1
        probabilities = adjusted_probabilities / np.sum(adjusted_probabilities)

        logging.debug(f"Final normalized probabilities: {probabilities}")

        # Return a dictionary mapping brands to their purchase probabilities

    except ZeroDivisionError:
        print("Error: Division by zero.")
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return dict(zip(brands, probabilities))


# PRICE IMPACT FUNCTIONS


def get_current_price(week: int, joint_calendar: pd.DataFrame, brand: str) -> float:
    """
    This function retrieves the current price of a specific brand in a given week from the joint calendar.

    Parameters:
    week (int): The week number.
    joint_calendar (pd.DataFrame): A DataFrame containing the joint calendar.
    brand (str): The brand for which the price is to be retrieved.

    Returns:
    float: The price of the brand in the given week.
    """
    # Retrieve the price of the brand in the given week from the joint calendar
    price = joint_calendar.loc[week, (brand, "price")]
    return price


def get_percent_change_in_price(reference_price: float, current_price: float) -> float:
    """
    Calculate the percent change between the reference price and the current price.

    Args:
        reference_price (float): The initial price to compare against.
        current_price (float): The current price to compare with the reference price.

    Returns:
        float: The percent change expressed as a decimal.

    Raises:
        ValueError: If the reference price is zero, as percent change cannot be calculated.
    """
    if reference_price == 0:
        raise ValueError("Reference price cannot be zero.")

    difference = current_price - reference_price
    percent_change = difference / reference_price
    return percent_change


def get_probability_of_change_in_units_purchased_due_to_price(
    reference_price: float,
    current_price: float,
    sensitivity_increase: float = 5,
    sensitivity_decrease: float = 10,
    threshold: float = 0.01,
) -> float:
    """
    Calculate the probability of a change in units purchased based on the price change.

    This function models price elasticity, returning the probability that an agent will purchase more or fewer units
    based on the change in price. It uses a logistic function to determine the probability.

    Parameters:
    reference_price (float): The reference price.
    current_price (float): The current price.
    sensitivity_increase (float, optional): The sensitivity factor for price increases. Default is 5.
    sensitivity_decrease (float, optional): The sensitivity factor for price decreases. Default is 10.
    threshold (float, optional): The threshold around zero percent difference where the probability is zero. Default is 0.01.

    Returns:
    float: The probability of purchasing more units (for a price decrease) or fewer units (for a price increase).
    """
    percent_change = get_percent_change_in_price(reference_price, current_price)

    # Handle the threshold around zero percent difference
    if abs(percent_change) < threshold:
        return 0.0

    # Use the logistic function to model the probability
    if percent_change < 0:
        # For a price decrease, the probability of purchasing more units increases
        probability = logistic_function(
            abs(percent_change) * sensitivity_decrease
        )  # Adjust the sensitivity with the scaling factor
    else:
        # For a price increase, the probability of purchasing fewer units increases
        probability = logistic_function(
            abs(percent_change) * sensitivity_increase
        )  # Adjust the sensitivity with the scaling factor

    return probability


def get_price_impact_on_brand_choice_probabilities(
    week_number: int,
    brand_list: List[str],
    joint_calendar: pd.DataFrame,
    brand_preference: str,
    loyalty_rate: float,
) -> Dict[str, float]:
    """
    This function calculates the probability of purchasing each brand.
    Note that there is separate logic for how much of a chosen brand to purchase,
    this differs by setting probability of switching brands based on price.

    Parameters:
    week_number (int): The week number.
    brand_list (List[str]): A list of all available brands.
    joint_calendar (pd.DataFrame): A DataFrame containing the joint calendar.
    brand_preference (str): The preferred brand.
    loyalty_rate (float): The loyalty rate.

    Returns:
    dict: A dictionary mapping brands to their purchase probabilities.
    """
    logging.debug("Entered Price Impact Block")
    price_list = {}

    # Generate price list for all brands
    try:
        for brand in brand_list:
            price_list[brand] = joint_calendar.loc[week_number, (brand, "price")]
    except Exception as e:
        print(f"Could not generate price list: {e}")

    logging.debug(f"Using pricelist: {price_list}")

    try:
        brands = list(price_list.keys())
        price_list_values = np.array(list(price_list.values()))

        # Apply inverse softmax transformation to price list to get normalized probability distribution
        transformed_price_list = magnitude_adjusted_softmax(
            price_list_values, log_transform=False, inverse=True
        )

        logging.debug(
            f"Inverse magnitude adjusted softmax price_list: {transformed_price_list}"
        )

        # Initialize base probabilities with equal chance for non-preferred brands
        base_probabilities = np.full_like(
            transformed_price_list, (1 - loyalty_rate) / (len(brands) - 1)
        )

        # Update the base probability of the preferred brand with the loyalty rate
        brand_preference_index = brands.index(brand_preference)
        base_probabilities[brand_preference_index] = loyalty_rate

        logging.debug(f"Base pricing probabilities: {base_probabilities}")

        # Adjust the base probabilities with the transformed price list values
        adjusted_probabilities = transformed_price_list * base_probabilities

        # Normalize the adjusted probabilities so they sum to 1
        probabilities = adjusted_probabilities / np.sum(adjusted_probabilities)

        logging.debug(f"Normalized pricing probabilities: {probabilities}")

        # Return a dictionary mapping brands to their purchase probabilities

    except ZeroDivisionError:
        print("Error: Division by zero.")
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return dict(zip(brands, probabilities))
