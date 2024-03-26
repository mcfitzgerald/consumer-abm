import numpy as np


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
