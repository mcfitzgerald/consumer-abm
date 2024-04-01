import numpy as np
from typing import Dict
from mesa import Model


def compute_total_purchases(model: Model) -> Dict[str, int]:
    """
    Compute the total purchases for each brand in the model.

    Args:
        model (Model): The model instance.

    Returns:
        Dict[str, int]: A dictionary where keys are brand names and values are total purchases.

    Raises:
        Exception: If an unexpected error occurs.
    """
    try:
        # Initialize a dictionary to store total purchases for each brand
        purchases = {brand: 0 for brand in model.brand_list}

        # Iterate over all agents in the model
        for agent in model.schedule.agents:
            # For each agent, iterate over all brands
            for brand in model.brand_list:
                # Add the number of purchases of the current brand by the current agent to the total
                purchases[brand] += agent.purchased_this_step[brand]

        return purchases
    except Exception as e:
        print("An unexpected error occurred in compute_total_purchases:", e)


def compute_average_price(model: Model) -> float:
    """
    Compute the average product price in the model.

    Args:
        model (Model): The model instance.

    Returns:
        float: The average product price.

    Raises:
        Exception: If an unexpected error occurs.
    """
    try:
        # Get the current price for each agent in the model
        prices = [agent.current_price for agent in model.schedule.agents]

        # Return the mean of the prices
        return np.mean(prices)
    except Exception as e:
        print("An unexpected error occurred in compute_average_price:", e)
