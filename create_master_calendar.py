import pandas as pd
from typing import Dict, List
from ad_calendar import prepare_ad_schedule_variables, generate_brand_ad_schedule
from promo_calendar import (
    prepare_promo_schedule_variables,
    generate_brand_promo_schedule,
)


def generate_joint_ad_promo_schedule(brands: List[str], config: Dict) -> pd.DataFrame:
    """
    This function generates a joint ad and promo schedule for a list of brands.

    Parameters:
    brands (List[str]): A list of brand names.
    config (Dict): A dictionary containing configuration parameters.

    Returns:
    pd.DataFrame: A DataFrame containing the joint ad and promo schedule for the brands.
    """
    joint_schedule = pd.DataFrame()

    for brand in brands:
        try:
            # Prepare variables for promo schedule
            (
                base_product_price,
                promo_depths,
                promo_frequencies,
                promo_weeks,
            ) = prepare_promo_schedule_variables(brand, config)

            # Generate promo schedule
            promo_schedule = generate_brand_promo_schedule(
                base_product_price, promo_depths, promo_frequencies, promo_weeks
            )

            # Prepare variables for ad schedule
            (
                budget,
                media_channels,
                priority,
                schedule,
            ) = prepare_ad_schedule_variables(brand, config)

            # Generate ad schedule
            ad_schedule = generate_brand_ad_schedule(
                budget, media_channels, priority, schedule
            )

            # Merge promo and ad schedules
            brand_schedule = pd.concat([promo_schedule, ad_schedule], axis=1)

            # Create a MultiIndex for the columns
            brand_schedule.columns = pd.MultiIndex.from_product(
                [[brand], brand_schedule.columns]
            )

            # Concatenate the joint_schedule DataFrame with the new DataFrame for the brand
            joint_schedule = pd.concat([joint_schedule, brand_schedule], axis=1)
        except Exception as e:
            print(f"An error occurred while generating schedule for {brand}: {e}")

    return joint_schedule
