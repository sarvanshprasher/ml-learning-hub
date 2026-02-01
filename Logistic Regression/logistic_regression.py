import sys
import os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import lab_utils_common
sys.path.append(str(Path(__file__).resolve().parent.parent))

STYLE_PATH = Path(__file__).resolve().parent.parent / 'deeplearning.mplstyle'
plt.style.use(str(STYLE_PATH))
from lab_utils_common_for_logistic_regression import draw_vthresh

input_array = np.array([1,2,3])
exp_array = np.exp(input_array)

print(f"input_array: {input_array}")
print(f"exp_array: {exp_array}")

input_val = 1  
exp_val = np.exp(input_val)


def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
      z (scalar or ndarray): input value or array

    Returns:
      sigmoid_z (scalar or ndarray): sigmoid of input
    """
    sigmoid_z = 1 / (1 + np.exp(-z))
    return sigmoid_z

# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)

# Use the function implemented above to get the sigmoid values
y = sigmoid(z_tmp)

# # Code for pretty printing the two arrays next to each other
# np.set_printoptions(precision=3) 
# print("Input (z), Output (sigmoid(z))")
# print(np.c_[z_tmp, y])

# Plot z vs sigmoid(z)
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
plt.show()
