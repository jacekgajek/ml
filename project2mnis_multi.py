from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

mnist: Bunch = fetch_openml('mnist_784', as_frame=False)

#%%
X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#%%
