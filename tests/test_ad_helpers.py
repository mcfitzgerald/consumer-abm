import pandas as pd
import pytest
from cabm_helpers.ad_helpers import *


def test_generate_brand_ad_channel_map():
    brand_list = ["brand1", "brand2"]
    config = {
        "brands": {
            "brand1": {"advertising": {"channels": ["channel1", "channel2"]}},
            "brand2": {"advertising": {"channels": ["channel3", "channel4"]}},
        }
    }
    expected_output = {
        "brand1": ["channel1", "channel2"],
        "brand2": ["channel3", "channel4"],
    }
    assert generate_brand_ad_channel_map(brand_list, config) == expected_output


def test_assign_weights():
    items = ["item1", "item2", "item3"]
    prior_weights = [0.1, 0.2, 0.7]
    weights_dict = assign_weights(items, prior_weights)
    assert isinstance(weights_dict, dict)
    assert len(weights_dict) == len(items)
    assert sum(weights_dict.values()) == pytest.approx(1.0)


def test_calculate_adstock():
    # Mock week
    week = 1

    # Mock joint_calendar DataFrame
    joint_calendar = pd.DataFrame(
        {
            ("brand1", "channel1"): [100, 200],
            ("brand1", "channel2"): [150, 250],
            ("brand2", "channel1"): [50, 100],
            ("brand2", "channel2"): [75, 125],
        }
    )

    # Mock brand_channel_map
    brand_channel_map = {
        "brand1": ["channel1", "channel2"],
        "brand2": ["channel1", "channel2"],
    }

    # Mock channel_preference
    channel_preference = {"channel1": 0.6, "channel2": 0.4}

    # Expected output
    expected_output = {"brand1": 220, "brand2": 110}

    # Call the function with the mocks
    result = calculate_adstock(
        week, joint_calendar, brand_channel_map, channel_preference
    )

    # Assert the result is as expected
    assert result == expected_output


def test_ad_decay():
    adstock = {"brand1": 100, "brand2": 200}
    factor = 2
    expected_output = {"brand1": 50, "brand2": 100}
    assert ad_decay(adstock, factor) == expected_output


def test_update_adstock():
    adstock1 = {"brand1": 100, "brand2": 200}
    adstock2 = {"brand1": 50, "brand2": 50, "brand3": 150}
    expected_output = {"brand1": 150, "brand2": 250, "brand3": 150}
    assert update_adstock(adstock1, adstock2) == expected_output


# def test_get_switch_probability():
#     adstock = {"brand1": 100, "brand2": 200, "brand3": 300}
#     preferred_brand = "brand2"
#     default_loyalty_rate = 0.1
#     probabilities = get_switch_probability(
#         adstock, preferred_brand, default_loyalty_rate
#     )
#     assert isinstance(probabilities, dict)
#     assert len(probabilities) == len(adstock)
#     assert sum(probabilities.values()) == pytest.approx(1.0)


def test_get_purchase_probabilities():
    adstock = {"brand1": 1265, "brand2": 0, "brand3": 245}
    preferred_brand = "brand1"
    loyalty_rate = 0.6
    sensitivity = 0.5

    probabilities = get_purchase_probabilities(
        adstock, preferred_brand, loyalty_rate, sensitivity
    )

    # Check if the function returns a dictionary
    assert isinstance(probabilities, dict)

    # Print probabilities
    print(probabilities)

    # Check if the dictionary has the correct keys
    assert set(probabilities.keys()) == set(adstock.keys())

    # Check if the probabilities sum to 1
    assert sum(probabilities.values()) == pytest.approx(1.0)

    # Check if no brand gets a purchase probability of zero
    for prob in probabilities.values():
        assert prob > 0
