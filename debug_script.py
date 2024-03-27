from cabm import cabm_model

model = cabm_model.ConsumerModel(
    1, "config.toml", enable_ads=True, enable_pricepoint=True
)
