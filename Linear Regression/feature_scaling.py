import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import load_house_data,run_gradient_descent
from lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc
from pathlib import Path

STYLE_PATH = Path(__file__).resolve().parent / 'deeplearning.mplstyle'
plt.style.use(str(STYLE_PATH))

x_train, y_train = load_house_data()
x_features = ['size(sqft)', 'bedrooms', 'floors', 'age(years)']

# fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
# for i in range(len(ax)):
#     ax[i].scatter(x_train[:,i],y_train)
#     ax[i].set_xlabel(x_features[i])
# ax[0].set_ylabel("Price (1000's)")
# plt.show()

_, _, hist = run_gradient_descent(x_train, y_train, 10, alpha = 1e-7)

plot_cost_i_w(x_train, y_train, hist)

def zscore_normalize_features(X):
    """
    Z-score normalize each feature in X

    Args:
      X (ndarray (m,n)): Data, m examples with n features
    Returns:
      X_norm (ndarray (m,n)): normalized data
      mu (ndarray (n,)): feature means
      sigma (ndarray (n,)): feature standard deviations
    """

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# normalize the original features
X_norm, X_mu, X_sigma = zscore_normalize_features(x_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(x_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

fig,ax=plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i],x_train[:,i],)
    ax[i].set_xlabel(x_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features before normalization")
plt.show()
fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],X_norm[:,i],)
    ax[i].set_xlabel(x_features[i])
ax[0].set_ylabel("count"); 
fig.suptitle("distribution of features after normalization")

plt.show()

w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )

#predict target using normalized features
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

    # plot predictions and targets versus original features    
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(x_features[i])
    ax[i].scatter(x_train[:,i],yp,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

# First, normalize out example.
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")