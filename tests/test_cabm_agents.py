from unittest.mock import patch
from cabm_agents import ConsumerModel


def test_ConsumerAgent_init():
    model = ConsumerModel(1)
    agent = model.schedule.agents[0]
    assert agent.unique_id == 0
    assert agent.model == model
    assert agent.household_size in [1, 2, 3, 4, 5, 6, 7]
    assert agent.consumption_rate >= 0
    assert agent.pantry_min >= 0
    assert agent.pantry_max >= agent.pantry_min
    assert agent.pantry_stock == agent.pantry_max
    assert agent.purchased_this_step == 0
    assert agent.current_price == 5
    assert agent.last_product_price == 5
    assert agent.purchase_behavior == "buy_minimum"
    assert agent.step_min == 0
    assert agent.step_max == 0


def test_ConsumerAgent_consume():
    model = ConsumerModel(1)
    agent = model.schedule.agents[0]
    initial_pantry_stock = agent.pantry_stock
    agent.consume()
    assert agent.pantry_stock <= initial_pantry_stock


def test_ConsumerAgent_set_purchase_behavior():
    model = ConsumerModel(1)
    agent = model.schedule.agents[0]

    # Test when pantry_stock is less than pantry_min
    agent.pantry_stock = agent.pantry_min - 1
    agent.last_product_price = 10

    # Mock the get_current_price function to return 9
    with patch("cabm_agents.get_current_price", return_value=9):
        agent.set_purchase_behavior()
        assert agent.purchase_behavior == "buy_maximum"

    # Mock the get_current_price function to return 11
    with patch("cabm_agents.get_current_price", return_value=11):
        agent.set_purchase_behavior()
        assert agent.purchase_behavior == "buy_minimum"

    # Test when pantry_stock is between pantry_min and pantry_max
    agent.pantry_stock = (agent.pantry_min + agent.pantry_max) / 2

    # Mock the get_current_price function to return 9
    with patch("cabm_agents.get_current_price", return_value=9):
        agent.set_purchase_behavior()
        assert agent.purchase_behavior == "buy_maximum"

    # Mock the get_current_price function to return 11
    with patch("cabm_agents.get_current_price", return_value=11):
        agent.set_purchase_behavior()
        assert agent.purchase_behavior == "buy_some_or_none"

    # Test when pantry_stock is greater than pantry_max
    agent.pantry_stock = agent.pantry_max + 1

    # Mock the get_current_price function to return any value
    with patch("cabm_agents.get_current_price", return_value=10):
        agent.set_purchase_behavior()
        assert agent.purchase_behavior == "buy_none"


def test_ConsumerAgent_purchase():
    model = ConsumerModel(1)
    agent = model.schedule.agents[0]
    initial_pantry_stock = agent.pantry_stock
    agent.purchase()
    assert agent.pantry_stock >= initial_pantry_stock


def test_ConsumerModel_init():
    model = ConsumerModel(10)
    assert model.num_agents == 10
    assert len(model.schedule.agents) == 10


def test_ConsumerModel_step():
    model = ConsumerModel(10)
    model.step()
    assert model.schedule.steps == 1
