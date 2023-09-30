import mesa
import numpy as np
import toml
from joint_calendar import generate_joint_ad_promo_schedule
from ad_helpers import generate_brand_ad_channel_map, assign_weights, calculate_adstock
from NEW_LIB_consumer_agent import (
    get_pantry_max,
    get_current_price,
    compute_total_purchases,
    compute_average_price,
)


config = toml.load("config.toml")

# Set up household parameters
household_sizes = config["household"]["household_sizes"]
household_size_distribution = config["household"]["household_size_distribution"]
base_consumption_rate = config["household"]["base_consumption_rate"]
pantry_min_percent = config["household"]["pantry_min_percent"]

# Set up retail environment
brand_list = list(config["brands"].keys())
brand_market_share = [
    config["brands"][brand]["current_market_share"] for brand in brand_list
]
try:
    assert round(sum(brand_market_share), 2) == 1.0
except AssertionError:
    print("Error: Brand market shares do not sum to 1.")


# Set up advertising and promotion
joint_calendar = generate_joint_ad_promo_schedule(brand_list, config)
brand_channel_map = generate_brand_ad_channel_map(brand_list, config)
channel_set = set(
    channel for channels in brand_channel_map.values() for channel in channels
)
channel_priors = [
    config["household"]["base_channel_preferences"][channel] for channel in channel_set
]


class ConsumerAgent(mesa.Agent):
    """Consumer of products"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.household_size = np.random.choice(
            household_sizes, p=household_size_distribution
        )
        self.consumption_rate = abs(
            np.random.normal(base_consumption_rate, 1)
        )  # Applied at household level
        self.brand_preference = np.random.choice(brand_list, p=brand_market_share)
        self.ad_channel_preference = assign_weights(channel_set, channel_priors)
        self.adstock = {i: 0 for i in brand_list}
        self.pantry_min = (
            self.household_size * pantry_min_percent
        )  # Forces must-buy when stock drops percentage of household size
        self.pantry_max = get_pantry_max(self.household_size, self.pantry_min)
        self.pantry_stock = self.pantry_max  # Start with a fully stocked pantry
        self.purchased_this_step = 0
        self.current_price = config["brands"][self.brand_preference][
            "base_product_price"
        ]
        self.last_product_price = config["brands"][self.brand_preference][
            "base_product_price"
        ]
        self.purchase_behavior = "buy_minimum"
        self.step_min = (
            0  # fewest number of products needed to bring stock above pantry minimum
        )
        self.step_max = 0

    def consume(self):
        try:
            self.pantry_stock = self.pantry_stock - (
                self.household_size / self.consumption_rate
            )
        except ZeroDivisionError:
            print("Error: Consumption rate cannot be zero.")
        except Exception as e:
            print("An unexpected error occurred:", e)

    # def ad_exposure(self):

    def set_purchase_behavior(self):
        try:
            print(self.model.week_number)
            self.current_price = get_current_price(
                self.model.week_number, joint_calendar, self.brand_preference
            )
            price_dropped = self.current_price < self.last_product_price
            if self.pantry_stock <= self.pantry_min:
                self.purchase_behavior = (
                    "buy_maximum" if price_dropped else "buy_minimum"
                )
            elif self.pantry_min < self.pantry_stock < self.pantry_max:
                self.purchase_behavior = (
                    "buy_maximum" if price_dropped else "buy_some_or_none"
                )
            elif self.pantry_stock >= self.pantry_max:
                self.purchase_behavior = "buy_none"
        except Exception as e:
            print("An unexpected error occurred in set_purchase_behavior:", e)

    def purchase(self):
        try:
            purchase_behaviors = {
                "buy_minimum": self.step_min,
                "buy_maximum": self.step_max,
                "buy_some_or_none": np.random.randint(self.step_min, self.step_max + 1),
                "buy_none": 0,
            }
            self.purchased_this_step = purchase_behaviors.get(self.purchase_behavior, 0)
            self.pantry_stock += self.purchased_this_step
        except Exception as e:
            print("An unexpected error occurred in purchase:", e)

    def step(self):
        self.consume()
        self.set_purchase_behavior()
        self.purchase()


class ConsumerModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N):
        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)
        self.week_number = 1  # Add week_number attribute

        # Create agents
        for i in range(self.num_agents):
            a = ConsumerAgent(i, self)
            self.schedule.add(a)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Total_Purchases": compute_total_purchases,
                "Average_Product_Price": compute_average_price,
                "Week_Number": "week_number",
            },
            agent_reporters={
                "Household_Size": "household_size",
                "Purchased_This_Step": "purchased_this_step",
                "Pantry_Stock": "pantry_stock",
                "Pantry_Max": "pantry_max",
                "Pantry_Min": "pantry_min",
                "Purchase_Behavior": "purchase_behavior",
                "Minimum_Purchase_Needed": "step_min",
                "Current_Product_Price": "current_price",
                "Last_Product_Price": "last_product_price",
            },
        )

    def step(self):
        self.datacollector.collect(self)
        """Advance the model by one step and collect data"""
        self.schedule.step()
        self.week_number += 1  # Increment week_number each step
        if self.week_number == 53:  # Reset week_number to 1 after 52 weeks
            self.week_number = 1

    # def get_week_number(self):
    #     return self.week_number
