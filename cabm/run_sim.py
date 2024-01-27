import toml
from tqdm import tqdm
from .cabm_agent import ConsumerModel, Configuration


with open("config.toml", "r") as config_file:
    model = ConsumerModel(N, config_file, enable_ads=True)
