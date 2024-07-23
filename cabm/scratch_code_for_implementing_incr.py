class ConsumerAgent(mesa.Agent):
    # ... existing code ...

    def get_incremental_units_to_purchase_based_on_price(self):
        """
        Adjusts self.units_to_purchase based on price changes using the probability of change.
        """
        try:
            # Get the current price of the brand choice
            current_price = get_current_price(
                self.model.week_number, self.config.joint_calendar, self.brand_choice
            )

            # Calculate the probability of change in units purchased due to price
            probability_of_change = (
                get_probability_of_change_in_units_purchased_due_to_price(
                    self.last_product_price, current_price
                )
            )

            # Determine if the price increased or decreased
            if current_price > self.last_product_price:
                # Price increased, decrement units to purchase
                incremental_units = -int(probability_of_change * self.units_to_purchase)
            else:
                # Price decreased, increment units to purchase
                incremental_units = int(probability_of_change * self.units_to_purchase)

            # Adjust units to purchase while respecting step_min and step_max
            new_units_to_purchase = self.units_to_purchase + incremental_units
            self.units_to_purchase = max(
                self.step_min, min(new_units_to_purchase, self.step_max)
            )

            # Update the last product price to the current price
            self.last_product_price = current_price

        except Exception as e:
            print(
                "An unexpected error occurred in get_incremental_units_to_purchase_based_on_price:",
                e,
            )

    # ... existing code ...
