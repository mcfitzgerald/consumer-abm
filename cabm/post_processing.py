# cabm/post_processing.py

import pandas as pd


def add_date_column(
    agent_df: pd.DataFrame, start_date: str = "2021-01-03", freq: str = "W"
) -> pd.DataFrame:
    """
    Adds a 'Date' column to the agent level output DataFrame.
    """
    agent_df_reset = agent_df.reset_index()
    dates = pd.date_range(
        start=start_date, periods=agent_df_reset["Step"].nunique(), freq=freq
    )
    step_to_date = pd.DataFrame({"Step": range(len(dates)), "Date": dates})
    agent_df_reset = agent_df_reset.merge(step_to_date, on="Step")
    agent_df = agent_df_reset.set_index(["Step", "AgentID"])
    return agent_df


def calculate_total_purchases(
    agent_df: pd.DataFrame, dynamic_attributes: list, purchase_columns: list
) -> pd.DataFrame:
    """
    Summation of purchases from agent-level data, plus the mean of dynamic attributes.
    """
    # Initialize accumulators
    purchases_data = {col: [] for col in purchase_columns}
    dates = []
    steps = []
    dynamic_data = {attr: [] for attr in dynamic_attributes}

    for (idx), row in agent_df.iterrows():
        for col in purchase_columns:
            brand = col.split("_")[-1]
            purchases_data[col].append(row["Purchased_This_Step"][brand])
        dates.append(row["Date"])
        steps.append(idx[0])
        for attr in dynamic_attributes:
            dynamic_data[attr].append(row[attr])

    total_purchases_df = pd.DataFrame({**purchases_data, "Date": dates, "Step": steps})
    for attr in dynamic_attributes:
        total_purchases_df[attr] = dynamic_data[attr]

    # group by step
    agg_dict = {attr: "mean" for attr in dynamic_attributes}
    for col in purchase_columns:
        agg_dict[col] = "sum"

    total_purchases_df = (
        total_purchases_df.groupby(["Date", "Step"]).agg(agg_dict).reset_index()
    )
    return total_purchases_df


def add_total_sales_columns(
    purchases_df: pd.DataFrame, purchase_columns: list, price_columns: list
) -> pd.DataFrame:
    """
    For each brand, multiply total units by average price => total sales column.
    """
    for purchase_col in purchase_columns:
        brand = purchase_col.split("_")[-1]
        price_col = f"price_{brand}"
        if price_col in price_columns:
            sales_col = f"Total_Sales_{brand}"
            purchases_df[sales_col] = (
                purchases_df[purchase_col] * purchases_df[price_col]
            )
    return purchases_df


def calculate_average_adstock(agent_df):
    """
    Example aggregator if you want brand-level adstock means. Just a placeholder.
    """
    adstock_cols = [c for c in agent_df.columns if c.startswith("Adstock_")]
    data = {c: [] for c in adstock_cols}
    dates = []
    steps = []

    for (idx), row in agent_df.iterrows():
        for col in adstock_cols:
            data[col].append(row[col])
        dates.append(row["Date"])
        steps.append(idx[0])

    df = pd.DataFrame({**data, "Date": dates})
    df = df.groupby(["Date"]).mean().reset_index()
    return df


def ensure_float_columns(df, exclude_columns=["Date", "Step"]):
    """
    Ensures that all columns except those excluded are float type.
    """
    for c in df.columns:
        if c not in exclude_columns:
            df[c] = df[c].astype(float)
    return df
