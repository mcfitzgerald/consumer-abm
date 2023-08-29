# Consumer Agent Based Model (CABM)

This project is a simulation of consumer behavior using agent-based modeling. The model simulates the purchasing behavior of consumers based on the current price of a product and the stock in their pantry.

## Files in this project

- `cabm_agents.py`: This is the main file where the `ConsumerAgent` and `ConsumerModel` classes are defined. The `ConsumerAgent` class represents a consumer who purchases and consumes a product. The `ConsumerModel` class represents the model that contains multiple consumer agents.

- `config.toml`: This file contains the configuration for the model. It includes parameters such as household sizes and their distribution, base consumption rate, pantry minimum percent, base product price, promo depths, and promo frequencies.

- `LIB_consumer_agent.py`: This file contains helper functions used by the `ConsumerAgent` class. These functions include `get_pantry_max`, `get_current_price`, `compute_total_purchases`, and `compute_average_price`.

- `test_cabm_agents.py`: This file contains unit tests for the `ConsumerAgent` and `ConsumerModel` classes.

## How to run the model

To run the model, you need to execute the `cabm_agents.py` file. Before running the model, make sure to set the desired parameters in the `config.toml` file.

## Testing

To run the tests, execute the `test_cabm_agents.py` file.

## Consumer Logic

### Purchasing Behavior
```
Start
  |
  |--- Try
  |     |
  |     |--- Set self.current_price
  |     |
  |     |--- Check if price dropped (self.current_price < self.last_product_price)
  |     |
  |     |--- If self.pantry_stock <= self.pantry_min
  |     |     |
  |     |     |--- If price dropped, set self.purchase_behavior to "buy_maximum"
  |     |     |--- Else, set self.purchase_behavior to "buy_minimum"
  |     |
  |     |--- Else If self.pantry_min < self.pantry_stock < self.pantry_max
  |     |     |
  |     |     |--- If price dropped, set self.purchase_behavior to "buy_maximum"
  |     |     |--- Else, set self.purchase_behavior to "buy_some_or_none"
  |     |
  |     |--- Else If self.pantry_stock >= self.pantry_max
  |     |     |
  |     |     |--- Set self.purchase_behavior to "buy_none"
  |
  |--- Except Exception
        |
        |--- Print error message
End
```

## Contributing

Contributions are welcome. Please submit a pull request if you have something to add or fix.
## Exploring Purchase Behavior
