from cabm import cabm_agent
from cabm.cabm_helpers.config_helpers import Configuration
from cabm.cabm_helpers.ad_helpers import (
    assign_weights,
    calculate_adstock,
    ad_decay,
    update_adstock,
    get_purchase_probabilities,
)
import toml

# Load the configuration from the TOML file
config = toml.load("config.toml")

# Create a Configuration object
configuration = Configuration(config)

# Set up variables for testing
week = 1
joint_calendar = configuration.joint_calendar
brand_channel_map = configuration.brand_channel_map
channel_preference = configuration.channel_priors

# Test assign_weights function
items = list(channel_preference.keys())
prior_weights = list(channel_preference.values())
weights_dict = assign_weights(items, prior_weights)
print(f"weights_dict: {weights_dict}")

# Test calculate_adstock function
adstock = calculate_adstock(week, joint_calendar, brand_channel_map, channel_preference)
print(f"adstock: {adstock}")

# Test ad_decay function
factor = configuration.ad_decay_factor
decayed_adstock = ad_decay(adstock, factor)
print(f"decayed_adstock: {decayed_adstock}")

# Test update_adstock function
updated_adstock = update_adstock(adstock, decayed_adstock)
print(f"updated_adstock: {updated_adstock}")

# Test get_purchase_probabilities function
preferred_brand = "Brand A"
loyalty_rate = 0.5
sensitivity = 0.5
purchase_probabilities = get_purchase_probabilities(
    updated_adstock, preferred_brand, loyalty_rate, sensitivity
)
print(f"purchase_probabilities: {purchase_probabilities}")
