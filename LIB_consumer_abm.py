import numpy as np

# Pricing functions

# Get current promo/no-promo price from base-price, depth, frequency

def get_current_price(base_price, promo_depths=[1, 0.75, 0.5], promo_frequencies=[0.7, 0.2, 0.1]):
    '''
    base_price: unitless number ("1" could be 1 dollar, 1 euro, etc.)
    promo_depths: list of percentage discounts to be take off base
    promo_frequencies: list of probabilities reflecting percentage of 
                       of occasions depth will be applied
    
    Example: get_current_price(4.99, promo_depths=[1, 0.75, 0.5], promo_frequencies=[0.5,0.25,0.25])
    
    Above example will return a price that is 4.99 50% of the time, 3.74 and 2.50 25% of the time)
    '''
    
    promo_depth = np.random.choice(promo_depths, p=promo_frequencies)

    current_price = base_price * promo_depth

    return current_price
