import numpy as np
import time
import copy,math
from pathlib import Path
import matplotlib.pyplot as plt

STYLE_PATH = Path(__file__).resolve().parent / 'deeplearning.mplstyle'
plt.style.use(str(STYLE_PATH))

x_train = np.array([[2104, 5, 1	,45], [1416, 3, 2, 40], [852, 2 ,1 , 35]])
y_train = np.array([460, 232, 178])


# print(f"x_train = {x_train} , shape = {x_train.shape} , x_train.type = {x_train.dtype}")

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def predict(x, w, b):
    """
    Predict the output for a single data point x using the linear model with parameters w and b.

    Args:
      x (ndarray (n,)): Input data point with n features
          Returns:
      p (scalar):  prediction
    """
    p = np.dot(w, x) + b
    return p

# make a prediction
x_vec = x_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")


def compute_cost(x, y, w, b):
    """
    Compute the cost function for all the training data

    Args:
      x (ndarray (m,n)): Input data, m examples with n features
      y (ndarray (m,)):  true "label" values
      w (ndarray (n,)):  model parameters
      b (scalar):        model parameter

    Returns:
      total_cost (scalar): The total cost of using w,b as the parameters for linear regression to fit the data points in x and y
    """
    m = x.shape[0]  # number of training examples
    total_cost = 0.0

    for i in range(m):
        f_wb_i = np.dot(w, x[i]) + b
        cost_i = (f_wb_i - y[i]) ** 2
        total_cost += cost_i

    total_cost /= (2 * m)

    return total_cost

cost = compute_cost(x_train, y_train, w_init, b_init)
print(f"cost : {cost}")

def compute_gradient(x, y, w, b):
    """
    Compute the gradient for linear regression

    Args:
      x (ndarray (m,n)): Input data, m examples with n features
      y (ndarray (m,)):  true "label" values
      w (ndarray (n,)):  model parameters
      b (scalar):        model parameter

    Returns:
        dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
        dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
      """
    m,n = x.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(x[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * x[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

tmp_dj_db, tmp_dj_dw = compute_gradient(x_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing


# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(x_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = x_train.shape
for i in range(m):
    print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")


# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()

# a = np.arange(10)

# # print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")
# # print(f"a[-1]: {a[-1]}")

# # try:
# #     c = a[10]
# # except Exception as e:
# #     print(f"IndexError: {e}")

# def my_dot(x, y):
#     """
#     Compute the dot product of two vectors

#     Args:
#       x (ndarray (n,)): first vector
#       y (ndarray (n,)): second vector
#     Returns:
#       dot_product (scalar): dot product of x and y
#     """
#     dot_product = 0
#     n = x.shape[0]

#     for i in range(n):
#         dot_product += x[i] * y[i]

#     return dot_product

# a = np.array([1, 2, 3, 4])
# b = np.array([-1, 4, 3, 2])
# c = my_dot(a, b)
# c = np.dot(a, b)
# print(f"{c}")

# print(f"my_dot(a, b) = {my_dot(a, b)}")

# a= np.array([[1,2,3],[4,5,6]])
# print(f"a.shape: {a.shape} a = \n{a}")

# a = np.zeros((1,5))
# print(f"a.shape: {a.shape} a = \n{a}")

# a = np.random.random_sample((3,3))
# print(f"a.shape: {a.shape} a = \n{a}")


