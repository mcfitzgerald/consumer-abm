import numpy as np


def magnitude_adjusted_softmax(x: np.ndarray) -> np.ndarray:
    """Compute magnitude_adjusted_softmax values for each sets of scores in x."""
    try:
        # Handle the case where x is a list of zeros
        if np.all(x == 0):
            return np.full(x.shape, 1.0 / x.size)

        # Set temperature relative to max value if not overidden
        # Note this is critical to do before overflow prevention step -- need to test if it changes before log
        temperature = np.floor(np.log(np.max(x)))
        print(f"temperature = {temperature}")

        # Apply log transformation
        x = np.log1p(x)
        print(f"log transformed = {x}")

        # Subtract the max value to prevent overflow
        x = x - np.max(x)
        print(f"overflow transform = {x}")

        e_x = np.exp(x / temperature)
        print(f"e_x = {e_x}")
        return e_x / np.sum(e_x)
    except ZeroDivisionError:
        print("Error: Division by zero in magnitude_adjusted_softmax.")
    except TypeError:
        print("Error: Input should be a numpy array.")
    except Exception as e:
        print(f"An unexpected error occurred in magnitude_adjusted_softmax: {e}")


# # Testing the magnitude_adjusted_softmax function with different inputs
# print("Testing magnitude_adjusted_softmax function...")

# Test 8: Adstock Value
print("Test 8:")
x = np.array([45372, 9754, 367, 1.0])
print(f"Input: {x}")
print(f"Output: {magnitude_adjusted_softmax(x)}")

# Test 9: Adstock Value
print("Test 9:")
x = np.array([45372, 0])
print(f"Input: {x}")
print(f"Output: {magnitude_adjusted_softmax(x)}")

# Test 10: Adstock Value
print("Test 9:")
x = np.array([0, 0])
print(f"Input: {x}")
print(f"Output: {magnitude_adjusted_softmax(x)}")


print("Testing completed.")
