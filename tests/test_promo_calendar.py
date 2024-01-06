import pytest
from promo_calendar import prepare_promo_schedule_variables


def test_prepare_promo_schedule_variables():
    # Define a mock configuration
    config = {
        "brands": {
            "brand1": {
                "base_product_price": 100.0,
                "promotions": {
                    "promo_depths": {"promo1": 0.1},
                    "promo_frequencies": {"promo1": 0.5},
                    "promo_calendar": ["campaign1"],
                },
            }
        },
        "ad_campaigns": {"campaign1": [1, 2, 3]},
    }

    # Test with valid brand and config
    result = prepare_promo_schedule_variables("brand1", config)
    assert result == (100.0, {"promo1": 0.1}, {"promo1": 0.5}, [1, 2, 3])

    # Test with non-existent brand
    with pytest.raises(KeyError):
        prepare_promo_schedule_variables("non_existent_brand", config)

    # Test with non-existent campaign
    config["brands"]["brand1"]["promotions"]["promo_calendar"] = [
        "non_existent_campaign"
    ]
    with pytest.raises(KeyError):
        prepare_promo_schedule_variables("brand1", config)
