def get_switch_probability(
    adstock: Dict, preferred_brand: str, default_loyalty_rate: float
) -> Dict:
    """
    This function calculates the probability of switching to each brand.

    Parameters:
    adstock (dict): A dictionary mapping brands to their adstock.
    preferred_brand (str): The preferred brand.
    default_loyalty_rate (float): The default loyalty rate.

    Returns:
    dict: A dictionary mapping brands to their switch probabilities.
    """
    try:
        brands = list(adstock.keys())
        adstock_values = list(adstock.values())

        # DEBUG
        print(f"adstock used in getprob function: {adstock}")

        if adstock[preferred_brand] > max(adstock_values):
            # DEBUG
            print("gretprob used first branch")
            return {brand: 1 if brand == preferred_brand else 0 for brand in brands}

        elif sum(adstock_values) == 0:
            probabilities = {
                brand: default_loyalty_rate
                if brand == preferred_brand
                else (1 - default_loyalty_rate) / (len(brands) - 1)
                for brand in brands
            }
            # DEBUG
            print("gretprob used second branch")
            return probabilities

        else:
            total_adstock = sum(adstock_values)
            probabilities = {
                brand: value / total_adstock for brand, value in adstock.items()
            }
            # DEBUG
            print("gretprob used third branch")
            return probabilities
    except ZeroDivisionError:
        print("Error: Division by zero.")
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
