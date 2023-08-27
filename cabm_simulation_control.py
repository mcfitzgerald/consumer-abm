import toml

# Import config file
config = toml.load("config.toml")

houseshold_sizes = config["houseshold_sizes"]
houseshold_size_distribution = config["houseshold_size_distribution"]
base_consumption_rate = config["base_consumption_rate"]
pantry_min_percent = config["pantry_min_percent"]
base_product_price = config["base_product_price"]
promo_depths = config["promo_depths"]
promo_frequencies = config["promo_frequencies"]
