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
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
import matplotlib.pyplot as plt
import numpy as np

sklearn.set_config(display="diagram")
sklearn.set_config(transform_output="pandas")
#%%
mnist: Bunch = fetch_openml('mnist_784', as_frame=False)

#%%
X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

X_test: np.ndarray
X_train: np.ndarray
y_train: np.ndarray
y_test: np.ndarray

some_digit = X_train[0]

X_train

#%%
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
def add_shifted_samples(X_original: np.ndarray) -> np.ndarray:
    bitmaps = X_original.reshape(len(X_original), image_size, image_size)
    def random_translation():
        return (np.random.randint(-2, 3), np.random.randint(-2, 3))
    def random_rotation():
        return np.pi * (np.random.rand() - 0.5) / 5

    transforms = [ EuclideanTransform(translation=random_translation(), rotation=random_rotation()) for _ in range(5) ]
    def doWarp(bitmap: np.ndarray, t) -> np.ndarray:
        # print(f"warping bitmap with ${t}")
        return transform.warp(bitmap, t.inverse)

    parallel = Parallel(n_jobs= 24)
    tt = np.array(parallel(delayed(doWarp)(bitmap, t) for bitmap in bitmaps for t in transforms ))
    transformed = np.array([ transform.warp(bitmap, t.inverse) for t in transforms for bitmap in bitmaps])
    return tt
    # return np.array(transformed).reshape(len(transformed), image_size * image_size)

X_augmented = add_shifted_samples(X_train[:1])
# %%
X_augmented
# %%
def fff(x):
    x / 2

Parallel(n_jobs=2)(delayed(fff)(i ** 2) for i in range(10))

# %%
plot_images(X_augmented[60000:])
# %%
plot_images(X_augmented)
# %%
print(X_augmented.shape)
print(X_train.shape)
print(y_train.shape)
# %%
# plot_image(X_augmented[2])
X_augmented.shape
# %%
np.random.seed(42)


noise = np.random.randint(-100, 100, (len(X_augmented), image_size * image_size))

# X_train_noised = X_train + noise
X_train_noised = np.clip(X_augmented + noise, 0, 255)

plot_image(X_train_noised[0])
# %%
plot_images(X_train_noised)

#%%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_noised.astype('float64'))
#%%
X_train_scaled

#%%
sgd_clf = SGDClassifier(random_state=42, n_jobs=-1)
sgd_clf.fit(X_train_scaled, y_train)
#%%
score = cross_val_score(sgd_clf, X_train, y_train, cv=3, n_jobs=-1, scoring='accuracy')
score

#%%
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3, n_jobs=-1)
y_train_pred
#%%
sample_weight = y_train_pred != y_train
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, sample_weight=sample_weight,
                                        values_format=".0%", cmap="cubehelix", normalize='true')

#%%