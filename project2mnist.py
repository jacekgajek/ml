# %%
from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch


mnist: Bunch = fetch_openml('mnist_784', as_frame=False)

# %%

mnist.DESCR
# %%

import matplotlib.pyplot as plt
import numpy as np

X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


def plot_image(pixels: np.ndarray):
    image_size = 28
    image = pixels.reshape(image_size, image_size)
    plt.imshow(image, cmap="binary")
    plt.axis("off")


plot_image(X[2])
plt.show()

# %%
plt.figure(figsize=(20, 20))
for idx, pixels in enumerate(X[:400]):
    plt.subplot(20, 20, idx + 1)
    plot_image(pixels)
plt.ion()
plt.show()

# %%
from sklearn.linear_model import SGDClassifier

# Stochastic Gradient Descent

y_train_5 = (y_train == '5')
sgd_classifier = SGDClassifier(random_state=42)
sgd_classifier.fit(X_train, y_train_5)

sgd_classifier.predict([X[1]])

# %%
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier

cross_val_score(sgd_classifier, X_train, y_train_5, cv=3, scoring='accuracy')

# %%
dummy_classifier = DummyClassifier()
dummy_classifier.fit(X_train, y_train_5)
print(any(dummy_classifier.predict(X_train)))

cross_val_score(dummy_classifier, X_train, y_train_5, cv=3, scoring='accuracy')
del dummy_classifier

# %%
from sklearn.model_selection import cross_val_predict

y_train_5_pred = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3)
y_train_5_pred

# %%
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


def print_scores(y_actual, y_prediction):
    print(confusion_matrix(y_actual, y_prediction))
    print(f"precision = TP/(TP + FP) = ${precision_score(y_actual, y_prediction)}")
    print(f"recall = TP/(TP + FN) = ${recall_score(y_actual, y_prediction)}")
    print(f"score = harmonic mean of precision and recall = ${f1_score(y_actual, y_prediction)}")


print_scores(y_train_5, y_train_5_pred)

# %%
plt.subplots(figsize=(20, 20))
for i, x in enumerate(X[0:100]):
    x: np.ndarray
    image_size = 28
    image = x.reshape(image_size, image_size)
    plt.subplot(10, 10, i + 1)
    plt.imshow(image, cmap='binary')
    plt.title(round(sgd_classifier.decision_function([x])[0], 2))
    plt.axis('off')
plt.show()

# %%
from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3, method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
precisions: np.ndarray
recalls: np.ndarray
thresholds: np.ndarray

# %%
plt.figure(figsize=(15, 4))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(-2379.064500932274, 0, 1.0, "k", "dotted", label="threshold")
plt.grid()
plt.xlabel("Threshold")
plt.axis([-50000, 50000, -0, 1])
plt.legend(loc="center right")
plt.show()

#%%
# for i in range(precisions)
# np.array([precisions == recalls]).
optimal_threshold = np.array(np.abs(precisions - recalls)).argmin()

print(precisions[optimal_threshold])
print(recalls[optimal_threshold])

thresholds[optimal_threshold]
#%%
plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
plt.grid(True)
plt.axis([0, 1, 0, 1])
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.legend()
#%%
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

plt.plot(fpr, tpr)
plt.grid(True)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


#%%

from sklearn.ensemble import RandomForestClassifier

forest_classifier = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_classifier, X_train, y_train_5, cv=3, method='predict_proba')
#%%

y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)

plt.figure(figsize=(15, 4))
plt.plot(thresholds_forest, precisions_forest[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds_forest, recalls_forest[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(0, 0, 1.0, "k", "dotted", label="threshold")
plt.grid()
plt.xlabel("Threshold")
plt.legend(loc="center right")
plt.show()

#%%
plt.plot(recalls_forest, precisions_forest)
plt.grid()
#%%

forest_classifier.fit(X_train, y_train_5)
#%%
from sklearn.metrics import roc_auc_score
y_pred_forest = y_probas_forest[:, 1] >= 0.5
print(f1_score(y_train_5, y_pred_forest))
print(roc_auc_score(y_train_5, y_scores_forest))
