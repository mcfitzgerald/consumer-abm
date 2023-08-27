import mesa
import numpy as np
import toml

# Import library functions for ConsumerAgent
from LIB_consumer_agent import get_pantry_max, get_current_price

# Import config file
config = toml.load("config.toml")

houseshold_sizes = config["houseshold_sizes"]
houseshold_size_distribution = config["houseshold_size_distribution"]
base_consumption_rate = config["base_consumption_rate"]
pantry_min_percent = config["pantry_min_percent"]
base_product_price = config["base_product_price"]
promo_depths = config["promo_depths"]
promo_frequencies = config["promo_frequencies"]


class ConsumerAgent(mesa.Agent):
    """Consumer of products"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.household_size = np.random.choice(
            houseshold_sizes, p=houseshold_size_distribution
        )
        self.consumption_rate = abs(
            np.random.normal(base_consumption_rate, 1)
        )  # Applied at household level
        self.pantry_min = (
            self.household_size * pantry_min_percent
        )  # Forces must-buy when stock drops percentage of household size
        self.pantry_max = get_pantry_max(self.household_size, self.pantry_min)
        self.pantry_stock = self.pantry_max  # Start with a fully stocked pantry
        self.purchased_this_step = 0
        self.current_price = base_product_price
        self.last_product_price = base_product_price
        self.purchase_behavior = "buy_minimum"
        self.step_min = (
            0  # fewest number of products needed to bring stock above pantry minimum
        )
        self.step_max = 0

    def consume(self):
        self.pantry_stock = self.pantry_stock - (
            self.household_size / self.consumption_rate
        )

    def set_purchase_behavior(self):
        self.current_price = get_current_price(
            base_product_price, promo_depths, promo_frequencies
        )
        price_dropped = self.current_price < self.last_product_price
        if self.pantry_stock <= self.pantry_min:
            self.purchase_behavior = "buy_maximum" if price_dropped else "buy_minimum"
        elif self.pantry_min < self.pantry_stock < self.pantry_max:
            self.purchase_behavior = (
                "buy_maximum" if price_dropped else "buy_some_or_none"
            )
        elif self.pantry_stock >= self.pantry_max:
            self.purchase_behavior = "buy_none"

    # def set_purchase_behavior(self):
    #     self.current_price = get_current_price(
    #         base_product_price, promo_depths, promo_frequencies
    #     )
    #     if self.pantry_stock <= self.pantry_min:
    #         if self.current_price >= base_product_price:
    #             self.purchase_behavior = "buy_minimum"
    #         elif self.current_price < self.last_product_price:
    #             self.purchase_behavior = "buy_maximum"
    #         else:
    #             self.purchase_behavior = "buy_minimum"
    #     elif self.pantry_min < self.pantry_stock < self.pantry_max:
    #         if self.current_price >= base_product_price:
    #             self.purchase_behavior = "buy_some_or_none"
    #         elif self.current_price < self.last_product_price:
    #             self.purchase_behavior = "buy_maximum"
    #         else:
    #             self.purchase_behavior = "buy_some_or_none"
    #     elif self.pantry_stock >= self.pantry_max:
    #         self.purchase_behavior = "buy_none"

    def purchase(self):
        if self.purchase_behavior == "buy_minimum":
            self.purchased_this_step = self.step_min
        elif self.purchase_behavior == "buy_maximum":
            self.purchased_this_step = self.step_max
        elif self.purchase_behavior == "buy_some_or_none":
            self.purchased_this_step = np.random.randint(
                self.step_min, self.step_max + 1
            )
        else:
            self.purchased_this_step = 0
        self.pantry_stock += self.purchased_this_step

    def step(self):
        self.consume()
        self.set_purchase_behavior()
        self.purchase()
