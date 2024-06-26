class ConsumerModel(mesa.Model):
    def __init__(self, N, config_file, enable_ads=True, enable_pricepoint=False, enable_adstock_influence=False):
        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)
        self.config = Configuration(config_file)
        self.enable_ads = enable_ads
        self.enable_pricepoint = enable_pricepoint
        self.enable_adstock_influence = enable_adstock_influence

        # Create agents
        for i in range(self.num_agents):
            agent = ConsumerAgent(i, self)
            self.schedule.add(agent)

    def step(self):
        self.schedule.step()

#for agent


def purchase(self):
    """
    This method simulates the purchase behavior of the consumer agent.
    It first resets the purchase count for the current step.
    Then, it determines the minimum and maximum possible purchases for the step based on the current pantry stock.
    Depending on the purchase behavior, it updates the purchase count and the pantry stock.
    """
    try:
        self.purchased_this_step = {
            brand: 0 for brand in self.config.brand_list
        }  # Reset purchase count
        # Determine purchase needed this step to maintain pantry_min or above
        if self.pantry_stock <= self.pantry_min:
            self.step_min = math.ceil(self.pantry_min - self.pantry_stock)
        else:
            self.step_min = 0
        # Set max possible purchase for step
        self.step_max = math.floor(self.pantry_max - self.pantry_stock)
        
        # Update purchase count based on purchase behavior
        if self.purchase_behavior == "buy_minimum":
            self.purchased_this_step[self.brand_choice] += self.step_min
        elif self.purchase_behavior == "buy_maximum":
            self.purchased_this_step[self.brand_choice] += self.step_max
        elif self.purchase_behavior == "buy_some_or_none":
            adstock_value = self.adstock[self.brand_choice]
            if self.model.enable_adstock_influence:
                if adstock_value > 1:
                    lower_bound = min(int(math.log10(adstock_value)), self.step_max)
                    self.purchased_this_step[self.brand_choice] += np.random.choice(
                        list(range(lower_bound, self.step_max + 1))
                    )
                else:
                    self.purchased_this_step[self.brand_choice] += np.random.choice(
                        list(range(0, self.step_max + 1))
                    )
            else:
                self.purchased_this_step[self.brand_choice] += np.random.choice(
                    list(range(0, self.step_max + 1))
                )
        elif self.purchase_behavior == "buy_none":
            self.purchased_this_step[self.brand_choice] += 0  # No purchase
        # Update pantry stock
        self.pantry_stock += sum(self.purchased_this_step.values())
    except Exception as e:
        print("An unexpected error occurred in purchase:", e)

    self.purchase_history.pop(0)
    self.purchase_history.append(self.brand_choice)
    self.update_brand_preference()

agent IndentationError

class ConsumerAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.config = model.config
        self.adstock = {brand: 0 for brand in self.config.brand_list}
        self.pantry_stock = self.config.pantry_max
        self.purchase_behavior = "buy_minimum"
        self.step_min = 0
        self.step_max = 0
        self.brand_choice = None
        self.purchase_history = []

    # Other methods...