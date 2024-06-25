import pandas as pd


# Time series

def add_date_column(agent_df: pd.DataFrame, start_date: str = '2021-01-03', freq: str = 'W') -> pd.DataFrame:
    '''
    Adds a 'Date' column to the agent level output DataFrame.

    Parameters:
    agent_df (pd.DataFrame): The agent level data with 'Step' and 'AgentID' columns.
    start_date (str): The start date for the date range. Default is '2021-01-03'.
    freq (str): The frequency of the date range. Default is 'W' (weekly).

    Returns:
    pd.DataFrame: The modified DataFrame with an added 'Date' column.
    '''
    # Reset index and move it to columns
    agent_df_reset: pd.DataFrame = agent_df.reset_index()

    # Create a date range starting from start_date with a frequency of freq
    # The number of periods is the number of unique 'Step' values in agent_df
    dates: pd.DatetimeIndex = pd.date_range(start=start_date, periods=agent_df_reset['Step'].nunique(), freq=freq)

    # Create a DataFrame mapping 'Step' to 'Date'
    step_to_date: pd.DataFrame = pd.DataFrame({'Step': range(0, len(dates)), 'Date': dates})

    # Merge this DataFrame with agent_df to add the 'Date' column
    agent_df_reset: pd.DataFrame = agent_df_reset.merge(step_to_date, on='Step')

    # Set 'Step' and 'AgentID' back as indices
    agent_df: pd.DataFrame = agent_df_reset.set_index(['Step', 'AgentID'])

    return agent_df







# Summary functions

def calculate_total_purchases(agent_df: pd.DataFrame, dynamic_attributes: list) -> pd.DataFrame:
    '''
    Computes total purchases per brand per step from agent level output
    and includes specified dynamic attributes (e.g. price and adspend from the joint calendar)
    
    Note: This function uses the mean of the dynamic attributes (e.g. price and adspend from the joint calendar)
    
    Parameters:
    agent_df (pd.DataFrame): The agent level data with 'Purchased_This_Step' and 'Date' columns.
    dynamic_attributes (list): List of dynamic attributes to be included in the output DataFrame.
    
    Returns:
    total_purchases_df (pd.DataFrame): A DataFrame with total purchases for each brand per date. This is no longer agent level data.
    '''
    # Initialize empty lists to store the results
    total_purchases_A = []
    total_purchases_B = []
    dates = []
    steps = []
    dynamic_data = {attr: [] for attr in dynamic_attributes}

    # Iterate over the 'Purchased_This_Step' column
    for index, row in agent_df.iterrows():
        # Append the number of purchases for each brand to the respective list
        total_purchases_A.append(row['Purchased_This_Step']['A'])
        total_purchases_B.append(row['Purchased_This_Step']['B'])
        dates.append(row['Date'])
        steps.append(index[0])  # Assuming 'Step' is the first level of the index

        # Append dynamic attributes
        for attr in dynamic_attributes:
            dynamic_data[attr].append(row[attr])

    # Create a new DataFrame with the total purchases for each brand
    total_purchases_df = pd.DataFrame({
        'Total_Purchases_A': total_purchases_A,
        'Total_Purchases_B': total_purchases_B,
        'Date': dates,
        'Step': steps
    })

    # Add dynamic attributes to the DataFrame
    for attr in dynamic_attributes:
        total_purchases_df[attr] = dynamic_data[attr]

    # Group by 'Date' and 'Step' and sum the purchases, but take the mean of dynamic attributes
    total_purchases_df = total_purchases_df.groupby(['Date', 'Step']).agg(
        {**{attr: 'mean' for attr in dynamic_attributes},
         'Total_Purchases_A': 'sum',
         'Total_Purchases_B': 'sum'}).reset_index()

    return total_purchases_df

def add_total_sales_columns(purchases_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Adds columns for total sales based on the brand price data and total purchases for each brand.
    The function dynamically identifies the columns for total purchases and prices based on standard naming convention
    used in calculate_total_purchases function.

    Parameters:
    purchases_df (pd.DataFrame): The DataFrame returned from calculate_total_purchases.

    Returns:
    pd.DataFrame: The modified DataFrame with added total sales columns for each brand.
    '''
    # Identify columns for total purchases and prices
    purchase_columns = [col for col in purchases_df.columns if col.startswith('Total_Purchases_')]
    price_columns = [col for col in purchases_df.columns if col.startswith('price_')]

    # Calculate total sales for each brand
    for purchase_col in purchase_columns:
        brand = purchase_col.split('_')[-1]
        price_col = f'price_{brand}'
        if price_col in price_columns:
            sales_col = f'Total_Sales_{brand}'
            purchases_df[sales_col] = purchases_df[purchase_col] * purchases_df[price_col]

    return purchases_df

def calculate_average_adstock(agent_df):
    '''
    Computes average adstock per brand per step from agent level output
    '''
    # Identify columns for adstock
    adstock_columns = [col for col in agent_df.columns if col.startswith('Adstock_')]

    # Initialize a dictionary to store the results
    average_adstock = {col: [] for col in adstock_columns}
    dates = []
    steps = []

    # Iterate over the 'Ad_Stock' column
    for index, row in agent_df.iterrows():
        # Append the adstock for each brand to the respective list
        for col in adstock_columns:
            average_adstock[col].append(row[col])
        dates.append(row['Date'])
        steps.append(index[0])  # Assuming 'Step' is the first level of the index

    # Create a new DataFrame with the average adstock for each brand
    average_adstock_df = pd.DataFrame({
        **average_adstock,
        'Date': dates,
    })

    # Group by 'Date' and 'Step' and calculate the average adstock
    average_adstock_df = average_adstock_df.groupby(['Date']).mean().reset_index()

    return average_adstock_df

# Validation

def ensure_float_columns(df, exclude_columns=['Date','Step']):
    '''
    Ensures that all columns in the DataFrame are of type float, except for the specified columns.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to be checked and modified.
    exclude_columns (list): List of column names to be excluded from conversion.
    
    Returns:
    pandas.DataFrame: The modified DataFrame with specified columns as floats.
    '''
    for col in df.columns:
        if col not in exclude_columns:
            df[col] = df[col].astype(float)
    return df