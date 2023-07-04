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

iris.target_names


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
X = iris_df[['petal width (cm)']].to_numpy()
y = iris.target_names[iris.target] == 'virginica'

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(n_jobs=-1)

log_reg.fit(X_train, y_train)

#%%
