# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp

# %%
immo = pd.read_csv('data/data-final-clean3.csv').iloc[:, 1:]

# %%
immo.info()


# %% GROUP EDUCATION VARIABLES
immo = (
    immo
    .assign(
        ecoles=lambda _: _.autres + _.elementaire + _.lycee + _.maternelle + _.college,
        gare_proche=lambda _: _[['dist_ratp', 'dist_sncf']].min(axis=1),
        exp_gare_proche=lambda _: np.exp(_.gare_proche),
        b_n1=lambda _: np.where(_.n1_ratp > 0, 1, np.where(_.n1_sncf > 0, 1, 0)),
        code_departement=lambda _: pd.Categorical(_.code_departement),
        year=lambda _: pd.Categorical(_.year)
        )
)

# %% Create categorical income columns for stratified spliting
immo['revenu_cat'] = pd.cut(
    immo.med_revenu,
    bins=[0., 20000., 25000., 30000., 35000., np.inf],
    labels=[1, 2, 3, 4, 5])

# %% TRAIN TEST SPLIT
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=7)
for train_index, test_index in split.split(immo, immo.revenu_cat):
    strat_train_set = immo.loc[train_index]
    strat_test_set = immo.loc[test_index]
# %% Remove categorical income columns
for set_ in (strat_train_set, strat_test_set):
    set_.drop('revenu_cat', axis=1, inplace=True)
# %%
X_train = strat_train_set.drop('prix_mcarre', axis=1)
y_train = strat_train_set.loc[:, 'prix_mcarre'].to_numpy()

# %%
X_train.columns

# %%
num_cols = [
    'population',
    'med_revenu', 
    'velib_stations',
    #'n1_ratp', 'n2_ratp', 'n3_ratp',
    #'n3_ratp',
    'dist_ratp', 
    #'n1_sncf', 'n2_sncf', 'n3_sncf',
    #'n3_sncf',
    'dist_sncf',
    #'gare_proche',
    #'college', 'elementaire', 'lycee', 'maternelle',
    'ecoles',
    'count_area_vert'
]

cat_cols = ['year', 'type_local', 'code_departement']

# %%
X_train

# %%
relevant_cols = [
    y for x in [x for x in 
        [num_cols, cat_cols, ['nombre_pieces_principales', 'b_naturalia', 'b_n1']]
        ] for y in x
]
X_train = X_train.loc[:, relevant_cols]
# %%
X_train
# %%
# PIPELINE

# 1. Convert `nombre_pieces_principales` and `surface` to categorical
# 2. Convert `code_departement` and `code_insee` to categorical
# 3. One-Hot econde categorical variables
# 4. Normalize numerical variables

# %% 1. Categorize Surface and number of rooms
from sklearn.base import BaseEstimator, TransformerMixin

class CategorizePieces(BaseEstimator, TransformerMixin):
    def __init__(self, pieces=1, max_pieces=10):
        self.pieces = pieces
        self.max_pieces = max_pieces

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()

        bins = list(map(
            float, 
            [x for y in [[-1], range(1, self.max_pieces, self.pieces), [np.inf]] for x in y]
            )
        )

        cats = ['pieces_upto' + str(x) for x in bins if x > 0.1]
        
        X_['nombre_pieces_principales'] = pd.cut(
            X_.nombre_pieces_principales,
            bins,
            labels=cats
        )

        return X_

# %% 2. Categorize numerical vars
#class CategorizeDep(BaseEstimator, TransformerMixin):
#    def __init__(self):
#        self
#
#    def fit(self, X, y=None):
#        return self
#
#    def transform(self, X):
#        X_ = X.copy()
#        X_['code_departement'] = pd.Categorical(
#            X_['code_departement']
#        )
#
#        return X_



# %% APPLYING ALL TOGETHER
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

cat_cols = [
    'type_local', 'code_departement', 
    'nombre_pieces_principales', 'year']

cat_pipeline = Pipeline([
    ('pieces', CategorizePieces(max_pieces=5)),
    #('cat_dep', CategorizeDep()),
    ('one_hot', OneHotEncoder())
])

full_pipeline = ColumnTransformer([
    ('cat', cat_pipeline, cat_cols),
    ('num', StandardScaler(), num_cols)
], remainder='passthrough')

# %%
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import RidgeCV

ridge_pipe = Pipeline(steps=[
    ('processor', full_pipeline),
    ('regressor', RidgeCV(alphas=np.logspace(-10, 10, 21)))
])

ridge_model = TransformedTargetRegressor(
        regressor=ridge_pipe,
        func=np.log10,
        inverse_func=sp.special.exp10
    )
# %%
ridge_model.fit(X_train, y_train)

# %%
from sklearn.metrics import mean_squared_error, r2_score

y_pred = ridge_model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_pred)

print(f'MSE: {rmse}')
print(f'R2: {r2}')


# %%
cat_hot_cols = list(
    ridge_model
    .regressor_[0]
    .transformers_[0][1]
    .named_steps['one_hot']
    .get_feature_names(cat_cols)
    
)

feature_names = [x for y in [w for w in [cat_hot_cols, num_cols, ['b_naturalia', 'b_n1']]] for x in y]

# %%
feature_names

# %%
X_prep = full_pipeline.fit_transform(X_train)


# %%
X_prep

# %% PLOT HEATMAP
ridge_cols = pd.DataFrame(
    X_prep,
    columns=feature_names
)

corr_matrix = ridge_cols.corr()

sns.heatmap(corr_matrix, xticklabels=True, yticklabels=True)
# %% PLOT FEATURE IMPORTANCE
coefs = list(
    ridge_model
    .regressor_[1]
    .coef_
)

df_coefs = pd.DataFrame({
    'name':feature_names,
    'coef':coefs
}).sort_values('coef', ascending=False)

a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(data=df_coefs, x='coef', y='name')

# %%

# %%

# %%
