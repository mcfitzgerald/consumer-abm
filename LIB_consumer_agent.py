import math
import numpy as np


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


def get_current_price(base_product_price, promo_depths, promo_frequencies):
    """
    base_price: unitless number ("1" could be 1 dollar, 1 euro, etc.)
    promo_depths: list of percentage discounts to be take off base
    promo_frequencies: list of probabilities reflecting percentage of occasions depth will be applied

    Example: get_current_price(4.99, promo_depths=[1, 0.75, 0.5], promo_frequencies=[0.5,0.25,0.25])

    Above example will return a price that is 4.99 50% of the time, 3.74 and 2.50 25% of the time
    """
    try:
        promo_depth = np.random.choice(promo_depths, p=promo_frequencies)
        current_price = base_product_price * promo_depth
        return current_price
    except Exception as e:
        print("An unexpected error occurred in get_current_price:", e)


def compute_total_purchases(model):
    """Model-level KPI: sum of total purchases across agents each step"""
    try:
        purchases = [agent.purchased_this_step for agent in model.schedule.agents]
        return sum(purchases)
    except Exception as e:
        print("An unexpected error occurred in compute_total_purchases:", e)


def compute_average_price(model):
    """Model-level KPI: average product price each step"""
    try:
        prices = [agent.current_price for agent in model.schedule.agents]
        return np.mean(prices)
    except Exception as e:
        print("An unexpected error occurred in compute_average_price:", e)
