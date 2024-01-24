from typing import List


# callable functions to be passed to generate ad calendar for custom spending strategy
def declining_spending_strategy(num_weeks: int) -> List[float]:
    if num_weeks == 1:
        return [1]
    first_week = 0.5
    remaining = (1 - first_week) / (num_weeks - 1)
    return [first_week] + [remaining] * (num_weeks - 1)


# Existing Campaigns
first_three_weeks_per_quarter = [1, 2, 3, 14, 15, 16, 27, 28, 29, 40, 41, 42]
holiday_season = [48, 49, 50, 51, 52]
back_to_school = [33, 34, 35]
summer_sale = [26, 27, 28, 29, 30]
black_friday = [47]
spring_sale = [14, 15, 16, 17, 18]  # Mid-April to Mid-May
valentines_day = [6, 7]  # Early February
easter_holiday = [13, 14]  # Early to Mid-April
halloween = [44]  # End of October
cyber_monday = [48]  # End of November
