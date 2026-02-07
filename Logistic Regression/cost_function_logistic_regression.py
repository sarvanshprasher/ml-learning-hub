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
from lab_utils_common_for_logistic_regression import  plot_data, sigmoid, dlc

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]]) 
y_train = np.array([0, 0, 0, 1, 1, 1])

# fig,ax = plt.subplots(1,1,figsize=(4,4))
# plot_data(X_train, y_train, ax)

# ax.axis([0, 4, 0, 3.5])
# ax.set_ylabel('$x_1$', fontsize=12)
# ax.set_xlabel('$x_0$', fontsize=12)
# plt.show()

def compute_cost_logistic(x,y,w,b):
    m = x.shape[0]
    total_cost = 0.0

    for i in range(m):
        z_i = np.dot(w, x[i]) + b
        f_wb_i = sigmoid(z_i)
        cost_i = -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
        total_cost += cost_i

    total_cost /= m
    return total_cost

w_array1 = np.array([1,1])
b_1 = -3
w_array2 = np.array([1,1])
b_2 = -4

print("Cost for b = -3 : ", compute_cost_logistic(X_train, y_train, w_array1, b_1))
print("Cost for b = -4 : ", compute_cost_logistic(X_train, y_train, w_array2, b_2))