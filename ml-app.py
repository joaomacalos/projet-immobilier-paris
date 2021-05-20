# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp

# %%
immo = pd.read_csv('data/data-final-clean5.csv')

# %%
immo.info()


# %% GROUP EDUCATION VARIABLES
immo = (
    immo
    .assign(
        ecoles=lambda _: _.autres + _.elementaire + _.lycee + _.maternelle + _.college,
        gare_proche=lambda _: _[['dist_ratp', 'dist_sncf']].min(axis=1),
        gare_proche_sq=lambda _: _.gare_proche ** 2,
        b_n1=lambda _: np.where(_.n1_ratp > 0, 1, np.where(_.n1_sncf > 0, 1, 0)),
        b_nat=lambda _: np.where(_.n3_nat > 0, 1, 0),
        b_new=lambda _: np.where(_.n1_new > 0, 1, 0),
        code_departement=lambda _: _.code_departement.astype('str'),
        year=lambda _: _.year.astype('str')
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
X_train = immo.drop('prix_mcarre', axis=1)
y_train = immo.loc[:, 'prix_mcarre'].to_numpy()

# %%
# X_train.columns

# %%
num_cols = [
    'gare_proche',
    'gare_proche_sq',
    'n3_sncf',
    'dist_new',
    'n3_velib',
    ]

cat_cols = ['year', 'type_local', 'code_insee', 'code_departement']

# %%
X_train.columns

# %%
relevant_cols = [
    y for x in [x for x in 
        [num_cols, cat_cols]
        ] for y in x
]
relevant_cols
# %%
X_train = X_train.loc[:, relevant_cols]
# %%
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
    #'nombre_pieces_principales', 
    'year', 'code_insee']

cat_pipeline = Pipeline([
    #('pieces', CategorizePieces(max_pieces=5)),
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
from sklearn.linear_model import LinearRegression

linear_pipe = Pipeline(steps=[
    ('processor', full_pipeline),
    ('regressor', LinearRegression())
])

linear_model = TransformedTargetRegressor(
        regressor=linear_pipe,
        func=np.log10,
        inverse_func=sp.special.exp10
    )
# %%
#ridge_model.fit(X_train, y_train)
linear_model.fit(X_train, y_train)

# %%
from sklearn.metrics import mean_squared_error, r2_score

y_pred = ridge_model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_pred)

print(f'MSE: {rmse}')
print(f'R2: {r2}')

# %%
y_pred = linear_model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_pred)

print(f'MSE: {rmse}')
print(f'R2: {r2}')

# %%
max(y_pred)


# %%
cat_hot_cols = list(
    linear_model
    .regressor_[0]
    .transformers_[0][1]
    .named_steps['one_hot']
    .get_feature_names(cat_cols)
    
)

#feature_names = [x for y in [w for w in [cat_hot_cols, num_cols, ['b1_n1', 'b_nat', 'b_new']]] for x in y]
feature_names = [x for y in [w for w in [cat_hot_cols, num_cols]] for x in y]



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
    linear_model
    .regressor_[1]
    .coef_
)

df_coefs = pd.DataFrame({
    'name':feature_names,
    'coef':coefs
}).sort_values('coef', ascending=False)

# %%
import re
df_coefs_insee = (
    df_coefs
    .query("name.str.contains('code_insee')", engine='python')
    .assign(group='commune')
)

df_coefs_departement = (
    df_coefs
    .query("name.str.contains('code_departement')", engine='python')
    .assign(group='departement')
    .assign(name=['Paris', 'Hauts-de-Seine', 'Val-de-Marne', 'Saint-Denis'])
)

df_coefs_year = (
    df_coefs
    .query("name.str.contains('year')", engine='python')
    .assign(group='year')
    .assign(name=lambda _: _.name.str.replace(r'(.+?)_', ''))
)

df_coefs_else = (
    df_coefs
    .query("~name.str.contains('year|insee|departement')", engine='python')
    .assign(
        group='Other',
        name=[
            'Maison', 'Velib', 'Distance to Gare',
            'Distance to new stations', '# SNCF Gares (< 3km)',
            'Distance to Gare Squared', 'Appartement']
    )
)

# %%
df_coefs_insee = (
    pd.merge(
    df_coefs_insee
    .assign(insee=lambda _: _.name.str.replace('code_insee_', '')),
    pd.read_csv('data/cities-clean.csv').assign(code_insee=lambda x: x.code_insee.astype('str')),
    how='left', left_on='insee', right_on='code_insee')
    .assign(name=lambda _: _.commune)
    .loc[[0, 1, 2, 3, 4, 138, 139, 140, 141, 142], ['name', 'coef', 'group']]
)

# %%
# %%
df_coefs_clean = (
    pd.concat([
        df_coefs_else, df_coefs_insee,
        df_coefs_departement, df_coefs_year
        ], 0)
    .assign(name=lambda _: _.name.str.replace(r'(.+?)_', ''))

)

# %%
fig, axes = plt.subplots(2, 2, figsize=(17, 10))

g0=sns.barplot(ax=axes[0, 0], data=df_coefs_insee, x='coef', y='name', palette='Spectral')
g0.set(ylabel=None)

g1=sns.barplot(ax=axes[1, 0], data=df_coefs_else, x='coef', y='name', palette='Spectral')
g1.set(ylabel=None)

g2=sns.barplot(ax=axes[0, 1], data=df_coefs_departement, x='coef', y='name', palette='Spectral')
g2.set(ylabel=None)

g3=sns.barplot(ax=axes[1, 1], data=df_coefs_year, x='coef', y='name', palette='Spectral')
g3.set(ylabel=None)

# %%
df1 = df_coefs.query("~name.str.contains('code')", engine='python')
# %%
g = sns.FacetGrid(df_coefs_clean, col='group')
g.map(sns.barplot, x="coef", y='name')

# %%

a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(data=df1, x='coef', y='name')

# %% SAVE MODEL
import joblib
filename = 'finalized_model.sav'
joblib.dump(linear_model, filename)


# %%

# %%
