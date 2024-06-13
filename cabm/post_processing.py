
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
def calculate_total_purchases(agent_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Computes total purchases per brand per step from agent level output and returns a DataFrame aggregated by date.
    
    Parameters:
    agent_df (pd.DataFrame): The agent level data with 'Purchased_This_Step' and 'Date' columns.
    
    Returns:
    total_purchases_df (pd.DataFrame): A DataFrame with total purchases for each brand per date. This is no longer agent level data.
    '''
    # Initialize empty lists to store the results
    total_purchases_A = []
    total_purchases_B = []
    dates = []

    # Iterate over the 'Purchased_This_Step' column
    for index, row in agent_df.iterrows():
        # Append the number of purchases for each brand to the respective list
        total_purchases_A.append(row['Purchased_This_Step']['A'])
        total_purchases_B.append(row['Purchased_This_Step']['B'])
        dates.append(row['Date'])

    # Create a new DataFrame with the total purchases for each brand
    total_purchases_df = pd.DataFrame({
        'Total_Purchases_A': total_purchases_A,
        'Total_Purchases_B': total_purchases_B,
        'Date': dates,
    })

    # Group by 'Date' and sum the purchases
    total_purchases_df = total_purchases_df.groupby(['Date']).sum().reset_index()

    return total_purchases_df