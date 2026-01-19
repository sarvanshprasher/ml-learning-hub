import math,copy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

STYLE_PATH = Path(__file__).resolve().parent / 'deeplearning.mplstyle'
plt.style.use(str(STYLE_PATH))

from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_cost(x, y, w, b):
    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw,dj_db

# plt_gradients(x_train,y_train,compute_cost,compute_gradient)
# plt.show()

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, compute_cost, compute_gradient):
    """
    Performs gradient descent to learn w and b. Updates w and b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w_in,b_in (scalar): initial values of model parameters
      alpha (float): learning rate
      num_iters (int): number of iterations to run gradient descent
      compute_cost (function): function to compute cost
      compute_gradient (function): function to compute gradient
    Returns:
      w (scalar): updated value of parameter after gradient descent
      b (scalar): updated value of parameter after gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
    """
    w = w_in
    b = b_in
    J_history = []
    p_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i<100000:     
            J_history.append(compute_cost(x, y, w , b))
            p_history.append([w,b])
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history #return w and J,w history for graphing

w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train,
                                                    w_init, b_init,
                                                    tmp_alpha,iterations,compute_cost,compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

# # plot cost versus iteration  
# fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
# ax1.plot(J_hist[:100])
# ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
# ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
# ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
# ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
# plt.show()

print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")

fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)
plt.show()