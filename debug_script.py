import inspect
import toml
from cabm import cabm_agent
from cabm.cabm_helpers import config_helpers




config = toml.load("config.toml")

C = config_helpers.Configuration(config)

M = cabm_agent.ConsumerModel(1, "config.toml")
