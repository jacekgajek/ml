from math import sqrt
from time import sleep

import sklearn
from joblib import Parallel, delayed
from skimage import transform
from skimage.transform import EuclideanTransform
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
import matplotlib.pyplot as plt
import numpy as np

sklearn.set_config(display="diagram")
sklearn.set_config(transform_output="pandas")
# %%
mnist: Bunch = fetch_openml('mnist_784', as_frame=False)

# %%
X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

X_test: np.ndarray
X_train: np.ndarray
y_train: np.ndarray
y_test: np.ndarray

some_digit = X_train[0]

X_train

# %%
image_size = 28


def plot_image(pixels: np.ndarray):
    image = pixels.reshape(image_size, image_size)
    plt.imshow(image, cmap="binary")
    plt.axis("off")


def plot_images(samples: np.ndarray):
    plt.figure(figsize=(10, 10))
    for idx, pixels in enumerate(samples[:100]):
        plt.subplot(10, 10, idx + 1)
        plot_image(pixels)
    plt.ion()
    plt.show()


plot_image(X[2])
plt.show()

# %%
X_train[0]


# %%
def add_noise(x: np.ndarray) -> np.ndarray:
    np.random.seed(42)
    noise = np.random.randint(0, 100, (len(x), image_size ** 2))
    return x + noise

X_train_noised = add_noise(X_train)
X_test_noised = add_noise(X_test)

# %%

plot_image(X_train_noised[0])
# %%
plot_image(X_train[0])

# %%
knn_clf = KNeighborsClassifier(n_jobs=-1)
knn_clf.fit(X_train_noised, X_train)

#%%
clean_digit = knn_clf.predict([X_test_noised[0]])
plot_image(clean_digit)

#%%
plot_image(X_test_noised[0])

#%%
plot_image(X_test[0])
