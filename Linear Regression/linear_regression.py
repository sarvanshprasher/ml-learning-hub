import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')

# %%
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# %%
print(f"x_train_shape is {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is : {m}")

# %%
m = len(x_train)
print(f"m : {m}")

# %%
i = 0

x_i = x_train[i]
y_i = y_train[i]

print(f"(x^{i}, y^{i})) = ({x_i} , {y_i})")

# %%
plt.scatter(x_train, y_train, marker='x', color='r')
plt.title("Housing prices")
plt.ylabel("Prices in (1000s of dollars)")
plt.xlabel("Size in sq ft")
plt.show()

# %%
w = 100
b = 100
print(f"w : {w}")
print(f"b : {b}")


# %%
def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


# %%
tmp_f_wb = compute_model_output(x_train, w, b, )

plt.plot(x_train, tmp_f_wb, c='b', label="Our prediction")

plt.scatter(x_train, y_train, marker='x', c='red', label='Actual Values')

plt.title("Housing prices")
plt.ylabel("Prices in (1000s of dollars)")
plt.xlabel("Size in sq ft")
plt.legend()
plt.show()

# %%
x_i = 1.2
cost_1200_sqft = w * x_i + b

print(f"Cost of 1200 square feet home is : {cost_1200_sqft}")


