from ad_calendar import generate_brand_ad_schedule


def test_generate_brand_ad_schedule():
    budget = 1000
    media_channels = ["channel1", "channel2"]
    priority = {"channel1": 60, "channel2": 40}
    schedule = {"channel1": [1, 2, 3], "channel2": [2, 3, 4]}
    spending_strategy = lambda num_weeks: [1 / num_weeks] * num_weeks

    df = generate_brand_ad_schedule(
        budget, media_channels, priority, schedule, spending_strategy
    )

    # Check that the DataFrame has the correct shape
    assert df.shape == (52, len(media_channels))

    # Check that the total spend is equal to the budget
    assert df.sum().sum() == budget

    # Check that the spend is distributed correctly according to the priority
    assert df["channel1"].sum() == budget * (priority["channel1"] / 100)
    assert df["channel2"].sum() == budget * (priority["channel2"] / 100)
