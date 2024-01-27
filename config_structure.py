{
    "title": "Consumer Model",
    "household": {
        "household_sizes": [1, 2, 3, 4, 5, 6, 7],
        "household_size_distribution": [0.28, 0.36, 0.15, 0.12, 0.06, 0.02, 0.01],
        "base_consumption_rate": 3,
        "pantry_min_percent": 0.1,
        "base_channel_preferences": {"TV": 0.4, "Web": 0.6},
        "ad_decay_factor": 2.0,
        "loyalty_alpha": 100,
        "loyalty_beta": 10,
        "sensitivity_alpha": 2,
        "sensitivity_beta": 5,
    },
    "brands": {
        "A": {
            "name": "Brand A",
            "current_market_share": 0.8,
            "base_product_price": 5.0,
            "promotions": {
                "promo_depths": [0.0, 0.25, 0.5],
                "promo_frequencies": [0.5, 0.25, 0.25],
                "promo_calendar": ["first_three_weeks_per_quarter", "holiday_season"],
            },
            "advertising": {
                "budget": 52000.0,
                "channels": ["TV", "Web"],
                "priority": {"TV": 40, "Web": 60},
                "ad_campaigns": {
                    "TV": "first_three_weeks_per_quarter",
                    "Web": "back_to_school",
                },
            },
        },
        "B": {
            "name": "Brand B",
            "current_market_share": 0.2,
            "base_product_price": 5.0,
            "promotions": {
                "promo_depths": [0.0, 0.25, 0.5],
                "promo_frequencies": [0.5, 0.25, 0.25],
                "promo_calendar": ["first_three_weeks_per_quarter", "holiday_season"],
            },
            "advertising": {
                "budget": 48000.0,
                "channels": ["TV", "Web"],
                "priority": {"TV": 30, "Web": 70},
                "ad_campaigns": {
                    "TV": "first_three_weeks_per_quarter",
                    "Web": "back_to_school",
                },
            },
        },
    },
    "ad_campaigns": {
        "first_three_weeks_per_quarter": [1, 2, 3, 14, 15, 16, 27, 28, 29, 40, 41, 42],
        "holiday_season": [48, 49, 50, 51, 52],
        "back_to_school": [33, 34, 35],
        "summer_sale": [26, 27, 28, 29, 30],
        "black_friday": [47],
        "spring_sale": [14, 15, 16, 17, 18],
        "valentines_day": [6, 7],
        "easter_holiday": [13, 14],
        "halloween": [44],
        "cyber_monday": [48],
    },
}
