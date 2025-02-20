# cabm/promo_calendar.py

import pandas as pd
from typing import Dict, List, Any, Tuple


def prepare_promo_schedule_variables(
    brand: str, config: Dict[str, Any]
) -> Tuple[float, Dict[int, float]]:
    """
    Prepare base price + dictionary of {week -> discounted_price}
    using the config's "promotions" data.
    """
    try:
        brand_info = config["brands"][brand]
        promo_info = brand_info["promotions"]
        base_product_price = brand_info["base_product_price"]
        promo_calendar_config = promo_info["promo_calendar"]
        promo_calendar: Dict[int, float] = {}

        for campaign, depth in promo_calendar_config.items():
            if campaign not in config["campaign_library"]:
                raise KeyError(f"{campaign} not found in config['campaign_library'].")
            for week in config["campaign_library"][campaign]:
                discounted_price = base_product_price * (1 if depth == 0 else depth)
                # store the best discount if multiple overlap
                if (week not in promo_calendar) or (
                    discounted_price < promo_calendar[week]
                ):
                    promo_calendar[week] = discounted_price

        return base_product_price, promo_calendar
    except KeyError as e:
        print(f"KeyError: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def generate_brand_promo_schedule(
    base_product_price: float,
    promo_calendar: Dict[int, float],
) -> pd.DataFrame:
    """
    Returns a df(weeks=1..52, col='price') with either base or discounted price.
    """
    df = pd.DataFrame(index=pd.RangeIndex(1, 53), columns=["price"])
    for week in df.index:
        if week in promo_calendar:
            df.loc[week, "price"] = promo_calendar[week]
        else:
            df.loc[week, "price"] = base_product_price
    return df
