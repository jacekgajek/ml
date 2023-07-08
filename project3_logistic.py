# %%
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# %%
iris = load_iris(as_frame=True)
iris_df: DataFrame = iris.data

np.max(iris_df['petal width (cm)']), np.min(iris_df['petal width (cm)'])


#%%

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

plt.figure(figsize=(10, 4), dpi=150)
plt.grid()
plt.plot(np.linspace(-8, 8, 1000), sigmoid(np.linspace(-8, 8, 1000)), label="$\sigma(t)=\\frac{1}{1+e^{-t}}$")

plt.axvline([0], color="k")
plt.axhline([0.5], color="k")
plt.legend()
plt.show()

#%%
X = iris_df[['petal width (cm)', 'petal length (cm)']].to_numpy()
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(n_jobs=-1)

log_reg.fit(X_train, y_train)

#%%
X_petal_space = np.linspace([0, 0], [3, 7], 1000)
X_petal_space
#%%
y_petal_proba: np.ndarray = log_reg.predict_proba(X_petal_space)
y_predictions: np.ndarray = log_reg.predict(X_train)

# def proba(petal_width):
#     return y_petal_proba[]

w = log_reg.coef_
b = log_reg.intercept_

# w[0] *x0 + w[1]*x[1] + b

decision_boundary = 1./w[0] * (-w[1] * X_petal_space[:, 0] - b)
decision_boundary

#%%
plt.figure(figsize=(10, 4), dpi=150)
# plt.plot(X_petal_width_space, y_petal_width_proba[:, 0], "b--", label="Not virginica")
# plt.plot(X_petal_width_space, y_petal_width_proba[:, 1], label="Virginica")
# plt.plot([decision_boundary, decision_boundary], [0, 1], "k:", linewidth = 2, label="Decision boundary")
plt.plot(X_train[y_train == 0, 0], X_train[y_train == 0, 1], "k.") #[y_train == 0], y_train[y_train == 0], "rs")
plt.plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], "r^") #[y_train == 0], y_train[y_train == 0], "rs")

plt.plot(X_petal_space[:, 0], y_petal_proba[:, 1], "b,") #[y_train == 0], y_train[y_train == 0], "rs")
# plt.plot(X_train[y_train == 1], y_train[y_train == 1], "g^")
# plt.xticks(np.concatenate([np.linspace(0, 3., 7), [np.round(decision_boundary, 2)]]))
plt.legend()
plt.xlabel("Petal width")
plt.ylabel("Probability")
plt.grid()
plt.show()

#%%
