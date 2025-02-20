# cabm/model_functions.py

import numpy as np
from typing import Dict
from mesa import Model


def compute_total_purchases(model: Model) -> Dict[str, int]:
    """
    Summation across all agents for each brand.
    """
    purchases = {b: 0 for b in model.brand_list}
    for agent in model.schedule.agents:
        for b in model.brand_list:
            purchases[b] += agent.purchased_this_step[b]
    return purchases


def compute_average_price(model: Model) -> float:
    """
    Average of agent.current_price
    """
    prices = [agent.current_price for agent in model.schedule.agents]
    return np.mean(prices) if prices else 0.0


def compute_average_purchase_probability(model: Model) -> Dict[str, float]:
    """
    For each brand, average the agent's purchase_probabilities[brand].
    """
    brand_probs = {b: [] for b in model.brand_list}
    for agent in model.schedule.agents:
        for b in model.brand_list:
            brand_probs[b].append(agent.purchase_probabilities.get(b, 0.0))
    return {b: np.mean(brand_probs[b]) for b in model.brand_list}
