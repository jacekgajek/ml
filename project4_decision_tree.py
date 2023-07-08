# %%
from typing import Callable, List

import numpy as np
import pandas as pd
import matplotlib as plt
from pandas import DataFrame
from pandas.plotting import scatter_matrix
from pandas.util._str_methods import removesuffix
from sklearn import set_config
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

set_config(display="diagram")
set_config(transform_output="pandas")


# %%

def load_auto_data():
    return pd.read_csv("data/poland_used_cars.csv")


auto_datasource = load_auto_data()

auto_datasource.describe()

# %%
voivodeship_capital_coords = {
    'Zachodniopomorskie': (53.26, 14.32),
    'Pomorskie': (54.20, 18.38),
    'Warmińsko': (53.47, 20.30),
    'Podlaskie': (53.08, 23.08),
    'Mazowieckie': (52.13, 21),
    'Wielkopolskie': (52.24, 16.56),
    'Lubuskie': (52.43, 15.14),
    'Dolnośląskie': (51.06, 17.01),
    'Opolskie': (50.39, 17.55),
    'Łódzkie': (51.46, 19.27),
    'Śląskie': (50.15, 19.01),
    'Świętokrzyskie': (50.53, 20.37),
    'Małopolskie': (50.03, 19.56),
    'Lubelskie': (51.14, 22.34),
    'Podkarpackie': (50.02, 22),
    'Warmińsko-mazurskie': (53.46, 20.28),
}


# %%
def clean_data(data: DataFrame) -> DataFrame:
    clean1 = data[data.year.str.match('[0-9][0-9][0-9][0-9]')]
    clean2 = clean1[clean1.fuel_type.str.match('[a-zA-Z+]+')]
    clean3 = clean2[clean2.mileage.str.match('[0-9 ]*km')]
    return clean3[clean3.voivodeship.isin(voivodeship_capital_coords.keys())]


auto = clean_data(auto_datasource)


def get_train_and_label(data: DataFrame):
    feature_attributes = ['brand', 'mileage', 'gearbox', 'engine_capacity', 'fuel_type', 'voivodeship', 'year']
    # feature_attributes = ['budget', 'runtime', 'vote_average', 'popularity', 'vote_count']
    label_attributes = ['price_in_pln']
    return data[feature_attributes].copy(), data[label_attributes].copy()


train_df, test_df = train_test_split(auto)

auto_train, y_train = get_train_and_label(train_df)
auto_test, y_test = get_train_and_label(test_df)


# %%
class PerItemFunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, function):
        self.function = function

    def fit(self, X: DataFrame, y=None, **fit_params):
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X: DataFrame):
        result = X.applymap(self.function)
        return result


# %%
class SplittingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str, tuple_extractor, suffixes: List[str]):
        self.tuple_extractor = tuple_extractor
        self.column_name = column_name
        self.suffixes = suffixes

    def fit(self, X, y=None):
        self.feature_names_in_ = [
            self.column_name + '_' + self.suffixes[0],
            self.column_name + '_' + self.suffixes[1]
        ]
        return self

    def transform(self, X: DataFrame):
        extracted = X.applymap(self.tuple_extractor)
        new_X = X.copy()
        list = extracted[self.column_name].tolist()
        x1 = [tuple[0] for tuple in list]
        x2 = [tuple[1] for tuple in list]
        name1 = self.suffixes[0]
        name2 = self.suffixes[1]
        new_X[name1] = x1
        new_X[name2] = x2
        new_X.drop(columns=[self.column_name], inplace=True)
        return new_X

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: DataFrame):
        return X.drop(columns=self.columns)

# %%

def mileage_to_number(mileage: str) -> int:
    return int(removesuffix(mileage, " km").replace(" ", ""))


def engine_capacity_to_number(engine_capacity: str) -> int:
    return int(removesuffix(engine_capacity, " cm3").replace(" ", ""))


def year_to_number(year: str) -> int:
    return int(removesuffix(year, " cm3").replace(" ", ""))


def voivodeship_to_spatial(voivodeship: str) -> (float, float):
    if voivodeship in voivodeship_capital_coords:
        return voivodeship_capital_coords[voivodeship]
    else:
        raise ValueError(f"Voivodeship unknown: {voivodeship}")


# %%

year_pipeline = make_pipeline(
    PerItemFunctionTransformer(year_to_number),
    FunctionTransformer(lambda y: 2023 - y),
    StandardScaler(),
)

engine_capacity_pipeline = make_pipeline(
    PerItemFunctionTransformer(engine_capacity_to_number),
    # FunctionTransformer(np.log),
    StandardScaler()
)

mileage_pipeline = make_pipeline(
    PerItemFunctionTransformer(mileage_to_number),
    # FunctionTransformer(np.log),
    StandardScaler()
)

# cat_attributes = ['gearbox', 'fuel_type', 'brand']
cat_attributes = ['gearbox']
cat_pipeline = make_pipeline(
    OneHotEncoder(handle_unknown='ignore', sparse_output=False)
)

preprocessing = make_pipeline(ColumnTransformer([
    ('mileage', mileage_pipeline, ['mileage']),
    ('engine_capacity', engine_capacity_pipeline, ['engine_capacity']),
    ('year', year_pipeline, ['year']),
    ('voivodeship', SplittingTransformer('voivodeship', voivodeship_to_spatial, ['lat', 'long']), ['voivodeship']),
    ('cat', cat_pipeline, cat_attributes),
]),
    DropColumnsTransformer(['cat__gearbox_manual', 'voivodeship__lat']),
ColumnTransformer([('long', StandardScaler(), ['voivodeship__long'])], remainder='passthrough')
)


auto_train_processed: DataFrame = preprocessing.fit_transform(auto_train, y_train['price_in_pln'])

auto_train_processed
# %%
# %%
train_and_y = auto_train_processed.copy()
train_and_y['price'] = y_train['price_in_pln']
# train_and_y['mil_to_year'] = np.log( np.maximum(1e-10, auto_train_processed['year__year'] / auto_train_processed['mileage__mileage'] ))
train_and_y
# %%
scatter_matrix(train_and_y[:5000], figsize=(20, 15))
# %%
auto_train_processed.describe()

# %%

np.random.seed(42)
tree_reg = RandomForestRegressor(random_state=42, n_jobs=-1)

tree_reg.fit(auto_train_processed, y_train.to_numpy().ravel())

tree_reg



#%%
score = cross_val_score(tree_reg, auto_train_processed, y_train.to_numpy().ravel(), n_jobs=-1)
score
#%%
