import toml
from tqdm import tqdm
from cabm.cabm_agent import ConsumerModel, Configuration


def run_simulation(config_file, num_agents, num_steps, enable_ads=True):
    # Load configuration
    config = toml.load(config_file)
    config = Configuration(config)

    # Create model
    model = ConsumerModel(num_agents, enable_ads, config)

    # Run the model with a progress bar
    for i in tqdm(range(num_steps)):  # tqdm adds the progress bar
        model.step()

    # Collect and return the data
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()
    return model_data, agent_data


if __name__ == "__main__":
    model_data, agent_data = run_simulation("config.toml", 1000, 110)
    print("Model Data:\n", model_data)
    print("Agent Data:\n", agent_data)
