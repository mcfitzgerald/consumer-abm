# %%
import mesa
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
# %%
# Set plot aesthetics

custom_settings = {
    "lines.linewidth": 0.8,
}

sns.set_theme(style='ticks', palette=["black"], rc=custom_settings)

# %%
# /*** CONFIG SECTION ***/

# Number of persons in household - approx percentage for 1,2...7 people in USA - from Statista
houseshold_sizes = [1,2,3,4,5,6,7]
houseshold_size_distribution = [0.28,0.36,0.15,0.12,0.06,0.02,0.01]

# Consumption rate: number of steps per person neded to consume 1 product
# E.g. a consumption rate of 3 means that a product is consumed by one person in 3 steps
# Concretely, that could mean a tube of toothpaste takes 3 weeks to use per person
# Consumption rates are defined statistically via absolute value of a normal distribution
base_consumption_rate = 3

# Pantry minimum percent (percent of household size)
pantry_min_percent = 0.1 # when pantry drops below 20% of household size, consumer must buy to replenish

# Base product price
base_product_price = 5

# Promo depths and frequencies
promo_depths=[1, 0.75, 0.5]
promo_frequencies=[0.5, 0.25, 0.25]


# Buying behaviors
# Just reference right now - not used in code yet, probably should make a dispatch table
buying_behaviors = ["buy_minimum", "buy_maximum", "buy_some_or_none", "buy_none"]

def get_pantry_max(household_size, pantry_min):
    '''
    Statistical assignment of maximum number of products a given household stocks
    Pantry min must be set before calling (default behavior of agent class)
    '''
    pantry_max = math.ceil(np.random.normal(household_size,1))
    if pantry_max < pantry_min:
        pantry_max = math.ceil(pantry_min)
    return pantry_max

def get_current_price(base_price, promo_depths=promo_depths, promo_frequencies=promo_frequencies):
    '''
    base_price: unitless number ("1" could be 1 dollar, 1 euro, etc.) 
    promo_depths: list of percentage discounts to be take off base 
    promo_frequencies: list of probabilities reflecting percentage of occasions depth will be applied
    
    Example: get_current_price(4.99, promo_depths=[1, 0.75, 0.5], promo_frequencies=[0.5,0.25,0.25])
    
    Above example will return a price that is 4.99 50% of the time, 3.74 and 2.50 25% of the time
    '''
    
    promo_depth = np.random.choice(promo_depths, p=promo_frequencies)
    current_price = base_price * promo_depth
    return current_price

def compute_total_purchases(model):
    '''Model-level KPI: sum of total purchases across agents each step'''
    purchases = [agent.purchased_this_step for agent in model.schedule.agents]
    return sum(purchases)

def compute_average_price(model):
    '''Model-level KPI: average product price each step'''
    prices = [agent.current_price for agent in model.schedule.agents]
    return np.mean(prices)

class ConsumerAgent(mesa.Agent):
    """Consumer of products"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.household_size = np.random.choice(houseshold_sizes, p=houseshold_size_distribution)
        self.consumption_rate = abs(np.random.normal(base_consumption_rate,1)) # Applied at household level
        self.pantry_min = (self.household_size * pantry_min_percent) # Forces must-buy when stock drops percentage of household size
        self.pantry_max = get_pantry_max(self.household_size, self.pantry_min)
        self.pantry_stock = self.pantry_max # Start with a fully stocked pantry
        self.purchased_this_step = 0
        self.current_price = base_product_price
        self.last_product_price = base_product_price
        self.purchase_behavior = "buy_minimum"
        self.step_min = 0 # fewest number of products needed to bring stock above pantry minimum
        self.step_max = 0 

    def consume(self):
        self.pantry_stock = (self.pantry_stock - (self.household_size/self.consumption_rate))
        
    # Need to define a parametric model for purchase behavior and price sensitivity
    # Could pobably make this more readable in a dispatch table or other form
    def set_purchase_behavior(self):
        self.current_price = get_current_price(base_product_price)
        # Must buy something to restock if at or below min pantry
        if self.pantry_stock <= self.pantry_min:
            if self.current_price >= base_product_price:
                # Only buy min if at the bast product (e.g. ticket/msrp...) price
                self.purchase_behavior = "buy_minimum"
            elif self.current_price < self.last_product_price: # not great, should allow for more nuanced spending in this regime
                # Cheaper than last time? Great! Fully restock pantry
                self.purchase_behavior = "buy_maximum"
            else:
                # Not comparatively cheaper from last trip? Fine. Buy minimum and wait for next shopping trip.
                self.purchase_behvaior = "buy_minimum" # redundant but specific for all cases
        elif self.pantry_min < self.pantry_stock < self.pantry_max:
            if self.current_price >= base_product_price:
                # Not desperate (pantry not at min), roll the dice and maybe buy / mabye not
                self.purchase_behavior = "buy_some_or_none"
            elif self.current_price < self.last_product_price:
                # Ooo! Sale! Max out the pantry, whatever that may be
                self.purchase_behavior = "buy_maximum"
            else:
                # Not comparatively cheaper from last trip? Roll the dice and maybe buy / mabye not
                self.purchase_behavior = "buy_some_or_none"
        elif self.pantry_stock >= self.pantry_max:
            # Pantry is full, can't buy
            self.purchase_behavior = "buy_none"
           
    
    def purchase(self):
        self.purchased_this_step = 0 # Reset purchase count
        # Determine purchase needed this step to maintain pantry_min or above
        if self.pantry_stock <= self.pantry_min:
            self.step_min = math.ceil(self.pantry_min - self.pantry_stock)
        else:
            self.step_min = 0
        self.step_max = math.floor(self.pantry_max - self.pantry_stock) # set max possible purchase for step
        if self.purchase_behavior == "buy_minimum":
            self.purchased_this_step += self.step_min
        elif self.purchase_behavior == "buy_maximum":
            self.purchased_this_step += self.step_max
        elif self.purchase_behavior == "buy_some_or_none": # include 0 as a possible purchase even if pantry not full
            self.purchased_this_step += np.random.choice(list(range(0, (self.step_max + 1))))
        elif self.purchase_behavior == "buy_none":
            self.purchased_this_step += 0 # redundant but specific
        self.pantry_stock += self.purchased_this_step
                        
    
    def step(self):
        self.last_product_price = self.current_price
        self.consume()
        self.set_purchase_behavior()
        self.purchase()
            
class ConsumerModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N):
        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)
        
        # Create agents
        for i in range(self.num_agents):
            a = ConsumerAgent(i, self)
            self.schedule.add(a)
        
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Total_Purchases": compute_total_purchases,
                "Average_Product_Price": compute_average_price
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
                "Last_Product_Price": "last_product_price"
                
            }
        )

    def step(self):
        self.datacollector.collect(self)
        """Advance the model by one step and collect data"""
        self.schedule.step()

# %%
model = ConsumerModel(1000)

for i in range(100):
    model.step()

# %%
model_summary_df = model.datacollector.get_model_vars_dataframe()

# %%
model_summary_df

# %%
agent_summary_df = model.datacollector.get_agent_vars_dataframe()

# %%
agent_summary_df

# %%
agent_4 = agent_summary_df.xs(4,level="AgentID")

# %%
sns.scatterplot(data=model_summary_df[10:],x="Average_Product_Price", y="Total_Purchases", marker="o", facecolor="none", edgecolor="black")

# %%
sns.stripplot(data=agent_4,x='Current_Product_Price',y='Purchased_This_Step', color="none", edgecolor="black", linewidth=0.8)

# %%
mod = smf.ols(formula='Total_Purchases ~ Average_Product_Price', data=model_summary_df[10:])

# %%
res = mod.fit()

# %%
print(res.summary())

# %%
consumer_mod = smf.ols(formula='Purchased_This_Step ~ Current_Product_Price', data=agent_summary_df.xs(4,level="AgentID")[10:])

# %%
consumer_res = consumer_mod.fit()

# %%
print(consumer_res.summary())

# %%
res.predict()

# %%
pred_ols = res.get_prediction()
iv_l = pred_ols.summary_frame()["obs_ci_lower"]
iv_u = pred_ols.summary_frame()["obs_ci_upper"]