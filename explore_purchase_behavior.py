import pandas as pd
from unittest.mock import patch
from cabm_agents import ConsumerModel


def ConsumerAgent_init():
    model = ConsumerModel(1)
    agent = model.schedule.agents[0]
    return agent


def ConsumerAgent_set_purchase_behavior(agent):
    # Initialize a DataFrame to store the values
    data = []

    # Test when pantry_stock is less than pantry_min
    agent.pantry_stock = agent.pantry_min - 1
    agent.last_product_price = 10

    # Mock the get_current_price function to return 9
    with patch("cabm_agents.get_current_price", return_value=9):
        agent.set_purchase_behavior()
        data.append(
            [
                "pantry_stock < pantry_min, price=9",
                agent.pantry_stock,
                agent.last_product_price,
                agent.purchase_behavior,
            ]
        )

    # Mock the get_current_price function to return 11
    with patch("cabm_agents.get_current_price", return_value=11):
        agent.set_purchase_behavior()
        data.append(
            [
                "pantry_stock < pantry_min, price=11",
                agent.pantry_stock,
                agent.last_product_price,
                agent.purchase_behavior,
            ]
        )

    # Test when pantry_stock is between pantry_min and pantry_max
    agent.pantry_stock = (agent.pantry_min + agent.pantry_max) / 2

    # Mock the get_current_price function to return 9
    with patch("cabm_agents.get_current_price", return_value=9):
        agent.set_purchase_behavior()
        data.append(
            [
                "pantry_stock between min and max, price=9",
                agent.pantry_stock,
                agent.last_product_price,
                agent.purchase_behavior,
            ]
        )

    # Mock the get_current_price function to return 11
    with patch("cabm_agents.get_current_price", return_value=11):
        agent.set_purchase_behavior()
        data.append(
            [
                "pantry_stock between min and max, price=11",
                agent.pantry_stock,
                agent.last_product_price,
                agent.purchase_behavior,
            ]
        )

    # Create a DataFrame from the data
    df = pd.DataFrame(
        data,
        columns=[
            "Test Block",
            "Pantry Stock",
            "Last Product Price",
            "Purchase Behavior",
        ],
    )
    return df
