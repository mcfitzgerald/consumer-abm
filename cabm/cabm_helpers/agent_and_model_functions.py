import math
import warnings
import numpy as np


# General Helper Functions


# A BETA SAMPLER THAT AVOIDS SMALL VALUES
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


# Household setup functions


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


# Consumer choice functions


def get_current_price(week, joint_calendar, brand):
    price = joint_calendar.loc[week, (brand, "price")]
    return price


# Model reporting functions


def compute_total_purchases(model):
    """Model-level KPI: sum of total purchases across agents each step for each brand"""
    try:
        purchases = {brand: 0 for brand in model.brand_list}
        for agent in model.schedule.agents:
            for brand in model.brand_list:
                purchases[brand] += agent.purchased_this_step[brand]
        return purchases
    except Exception as e:
        print("An unexpected error occurred in compute_total_purchases:", e)


def compute_average_price(model):
    """Model-level KPI: average product price each step"""
    try:
        prices = [agent.current_price for agent in model.schedule.agents]
        return np.mean(prices)
    except Exception as e:
        print("An unexpected error occurred in compute_average_price:", e)
