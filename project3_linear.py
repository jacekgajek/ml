# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import get_scorer_names
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# %%

def f(x):
    return x**3 - 2* x**2 + 2 * x + 3


np.random.seed(45)
m = 300
X: np.ndarray = np.random.rand(m, 1) * 10 - 5
y: np.ndarray = f(X) + np.random.randn(m, 1) * 4

np.min(X), np.max(X)

# %%

rank = 10

pipeline = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=rank),
    # LinearRegression(n_jobs=-1),
    ElasticNet(max_iter=100000, alpha=0.1, l1_ratio=0.5)
)
pipeline.fit(X, y.ravel())

plt.figure(figsize=(10, 10))
plt.plot(X, y, 'r.', scalex=True)
X_pol = np.linspace(-5.0, 5.0, 10000).reshape(-1, 1)
Y_pol = pipeline.predict(X_pol)
plt.plot(X_pol, Y_pol)
plt.ylim(-100, 20)
plt.show()

np.mean(cross_val_score(pipeline, X, y, n_jobs=-1, cv=5))
# %%

train_sizes, train_scores, valid_scores = learning_curve(pipeline, X, y, train_sizes=np.linspace(0.01, 1.0, 5), cv=5,
                                                         n_jobs=-1, scoring='neg_root_mean_squared_error')
train_sizes: np.ndarray
train_scores: np.ndarray
valid_scores: np.ndarray

train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label='train')
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label='valid')
plt.ylim(0, 50)
plt.grid()
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('RMSE')

plt.show()
