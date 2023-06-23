# %%
from pathlib import Path
from pandas import DataFrame
import pandas as pd
import numpy as np
import sklearn

sklearn.set_config(display="diagram")
sklearn.set_config(transform_output="pandas")


def load_movie_data():
    return pd.read_csv(Path("data/movie_dataset.csv"))


def prepare_movie_data(data: DataFrame):
    filtered: DataFrame = data.dropna().loc[~(data == 0).any(axis=1)]
    filtered.set_index('index')
    return filtered


movie_df = prepare_movie_data(load_movie_data())

movie_df

# %%

from sklearn.model_selection import train_test_split


def add_budget_cat_column():
    bincount = 5
    bins = np.append(np.linspace(0., 1e8, bincount), [np.inf])
    labels = np.arange(1, bincount + 1)
    movie_df['budget_cat'] = pd.cut(movie_df['budget'], bins=bins, labels=labels)


def drop_budget_cat_column(dfs: list[DataFrame]):
    for train_set in dfs:
        train_set.drop(columns=['budget_cat'], inplace=True)


movie_df.hist(figsize=(10, 10))

add_budget_cat_column()

training_df: DataFrame
testing_df: DataFrame
training_df, testing_df = train_test_split(movie_df, test_size=0.2, stratify=movie_df['budget_cat'], random_state=42)

drop_budget_cat_column([training_df, testing_df])


def get_train_and_label(data: DataFrame) -> tuple[DataFrame, DataFrame]:
    feature_attributes = ['budget', 'genres', 'runtime', 'vote_average', 'popularity', 'vote_count']
    # feature_attributes = ['budget', 'runtime', 'vote_average', 'popularity', 'vote_count']
    label_attributes = ['revenue']
    return data[feature_attributes].copy(), data[label_attributes].copy()


x_train, y_train = get_train_and_label(training_df)

x_train
# %%
from sklearn.base import TransformerMixin, BaseEstimator
from pandas import Series


class NaFiller(BaseEstimator, TransformerMixin):
    def __init__(self, replacement):
        self.replacement = replacement

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: DataFrame):
        return X.fillna(self.replacement)


class SeparatorSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, sep=' '):
        self.sep = sep

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, XX: DataFrame):
        X = XX.copy()
        for column in X.columns:
            X[column] = [v.split(self.sep) for v in X[column]]
        return X


class ListExpander(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.classes = []

    def fit(self, X: DataFrame, y=None, **fit_params):
        self.classes = []
        for column in X.columns:
            values: Series = X[column]
            feat_classes = set()
            for x in values:
                feat_classes = feat_classes | set(x)
            self.classes.append((column, list(feat_classes)))
        return self

    def transform(self, XX: DataFrame):
        X = XX.copy()
        # all_classes = [cl for _, cl in self.classes][0]
        for (column, classes) in self.classes:
            rows = []
            for array in X[column]:
                single_feature_row = []
                for c in classes:
                    if c in array:
                        single_feature_row.append(1)
                    else:
                        single_feature_row.append(0)
                rows.append(single_feature_row)
            for idx, c in enumerate(classes):
                X[c] = np.transpose(rows)[idx]
            X.drop(columns=[column], inplace=True)
        return X


# %%

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline


def make_preprocessing():
    num_attribs = ['budget', 'runtime', 'vote_average', 'popularity', 'vote_count']
    cat_attribs = ['genres']

    genre_pipeline = make_pipeline(
        NaFiller(''),
        SeparatorSplitter(),
        ListExpander()
    )

    num_pipeline = make_pipeline(
        FunctionTransformer(np.log, np.exp)
    )

    return ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('genre', genre_pipeline, cat_attribs)
    ])


preprocessing = make_preprocessing()

feature_prepared = preprocessing.fit_transform(x_train, y_train)

feature_prepared.hist(figsize=(10, 10))
feature_prepared

# %%

from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(x_train, y_train)

forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_reg.fit(x_train, y_train.values.ravel())

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(x_train, y_train)

# %%

from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg, x_train, y_train, scoring='neg_root_mean_squared_error', cv=10)

pd.Series(tree_rmses).describe()

# %%

from sklearn.model_selection import cross_val_score

forest_rmses: np.ndarray = -cross_val_score(forest_reg, x_train, y_train.values.ravel(),
                                            scoring='neg_root_mean_squared_error', cv=10)

pd.Series(forest_rmses).describe()

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def get_grid_search():
    full_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state=42)),
        # ("random_forest", HistGradientBoostingRegressor(random_state=42))
    ])
    param_grid = [
        # {'random_forest__learning_rate': np.arange(1, 10) / 10.0},
        {'random_forest__max_features': np.arange(1, 10)},
    ]
    return GridSearchCV(full_pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=20)


grid_search = get_grid_search()
grid_search.fit(x_train, y_train.values.ravel())

grid_search.best_params_

# %%
from sklearn.metrics import mean_squared_error

estimator: Pipeline = grid_search.best_estimator_

x_test, y_test = get_train_and_label(testing_df)

test_rmse = mean_squared_error(y_test, estimator.predict(x_test), squared=False)
train_rmse = mean_squared_error(y_train, estimator.predict(x_train), squared=False)
print(train_rmse)
print(test_rmse)

# %%
xx = x_test.copy()
xx['revenue'] = y_test
xx['estimated_rev'] = estimator.predict(x_test).astype(int)
xx['err'] = np.abs((xx['estimated_rev'] - xx['revenue']) / xx['revenue'])
xx
# %%
xx = x_train.copy()
xx['revenue'] = y_train
xx['estimated_rev'] = estimator.predict(x_train).astype(int)
xx['err'] = np.abs((xx['estimated_rev'] - xx['revenue']) / xx['revenue'])
xx
