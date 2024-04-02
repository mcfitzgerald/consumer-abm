import toml
from cabm import config_helpers

config = toml.load("config.toml")

Config = config_helpers.Configuration(config)
