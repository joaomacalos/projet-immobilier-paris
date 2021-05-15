# PROJET IMMOBILIER PARIS

# The objective of this code is to download and clean all the data that we
# are going to use in our final project.

# %%
import pandas as pd
import numpy as np
from janitor import clean_names
import seaborn as sns

# DVF data:
# Source: Metadata
# Already cleaned

df_main = pd.read_csv('data/dvf-clean-work.csv').iloc[:, 2:]

# %% View columns
df_main.columns

# %% Keep only a subset of columns:
columns_to_keep = [
    'date_mutation', 'nature_mutation', 
    'no_voie', 'type_de_voie', 'voie', 'code_postal', 
    'code_departement', 'code_commune', 'valeur_fonciere', 'section', 
    'nombre_de_lots', 'nombre_pieces_principales', 'commune', 
    'surface_batiment', 'surface', 'prix_mcarre', 'prix', 'date', 'year'
    ]

df_main = df_main.loc[:, columns_to_keep]

# %% Get CODE INSEE from code departement and code commune
df_main = (
    df_main
        .assign(
            code_commune=lambda x: x.code_commune.astype('str'),
            code_departement=lambda x: x.code_departement.astype('str')
        )
        .assign(
            code_commune=lambda x: np.where(
                x.code_commune.str.len() == 1,
                "0" + x.code_commune,
                x.code_commune
            ),           
            code_insee=lambda x: np.where(
                x.code_commune.str.len() == 3,
                x.code_departement + x.code_commune,
                x.code_departement + '0' + x.code_commune),
            no_voie=lambda x: x.no_voie.astype('int').astype('str'),
            adresse=lambda x: (x.no_voie.astype('str') + 
                                " " + x.type_de_voie + " " + x.voie)
        )
        .assign(adresse=lambda x: x.adresse.str.strip())
)

df_main.head()

# %%
df_main.query("code_insee.str.len() == 4", engine='python')
# %% Save CSV with code_insee and adresse to collect geo-coordinates
(df_main
    .loc[:, ['code_insee', 'adresse']]
    .drop_duplicates()
    .to_csv('dvf-adresses.csv', index=False)
)

# %%
df_geo = pd.read_csv('dvf-geocoords.csv')

# %% All adresses are in the same insee_code, great!
df_geo.query("code_insee != result_citycode")

# %% Function to efficiently calculate the distance between two points
# Source: https://stackoverflow.com/a/19414306/7705000
def spherical_dist(pos1, pos2, r=6731):
    pos1 = pos1 * np.pi / 180
    pos2 = pos2 * np.pi / 180
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))


# %%
def retrieve_info(array):
    a3 = list(filter(lambda x: x <= 3, array))
    a2 = list(filter(lambda x: x <= 2, a3))
    a1 = list(filter(lambda x: x <= 1, a2))

    n3 = len(a3)
    n2 = len(a2)
    n1 = len(a1)

    dmin = min(array)

    return n1, n2, n3, dmin

# %%
lat = df_geo.latitude.to_numpy()
lon = df_geo.longitude.to_numpy()

coords = np.array([[lat[i], lon[i]] for i in range(len(lat))])

# %%
sncf = pd.read_csv('data/sncf-geocoordinates.csv').iloc[:, 1:]
sncf = sncf.query("latitude.notna()", engine="python")

# %%
sncf_lat = sncf.latitude.to_numpy()
sncf_lon = sncf.longitude.to_numpy()

sncf_coords = np.array([[sncf_lat[i], sncf_lon[i]] for i in range(len(sncf_lat))])

# %%
ratp = pd.read_csv("data/ratp-geocoordinates.csv").iloc[:, 1:]
ratp = ratp.query("stop_lat.notna()", engine="python")

ratp_lat = ratp.stop_lat.to_numpy()
ratp_lon = ratp.stop_lon.to_numpy()

ratp_coords = np.array([[ratp_lat[i], ratp_lon[i]] for i in range(len(ratp_lat))])

# %%
import time
start_time = time.time()
dist_to_sncf = list(map(
    lambda x: retrieve_info(list(map(
        lambda y: spherical_dist(x, y), sncf_coords)
        )), coords)
    )

print("Process finished --- %s seconds ---" % (time.time() - start_time))
# %%
import time
start_time = time.time()
dist_to_ratp = list(map(
    lambda x: retrieve_info(list(map(
        lambda y: spherical_dist(x, y), ratp_coords)
        )), coords)
    )

print("Process finished --- %s seconds ---" % (time.time() - start_time))

# %%
dist_ratp = pd.DataFrame(data = np.asarray(dist_to_ratp),
             columns=['n1_ratp', 'n2_ratp', 'n3_ratp', 'dist_ratp']
)

dist_sncf = pd.DataFrame(data = np.asarray(dist_to_sncf),
             columns=['n1_sncf', 'n2_sncf', 'n3_sncf', 'dist_sncf']
)

# %%
df_geo = pd.concat([df_geo, dist_ratp, dist_sncf], axis=1)

# %%
df_main = pd.merge(
    df_main,
    df_geo.assign(code_insee=lambda x: x.code_insee.astype('str')),
    left_on=['code_insee', 'adresse'],
    right_on=['code_insee', 'adresse']
)

# %%
df_main.info()
# %%

df_main.to_csv('dvf-main.csv')

# %% Create a DF with the code_insee to filter out info in the following tables
code_insee = df_main[['code_insee']].drop_duplicates()
code_insee.head(2)

# %% POPULATION
# Source: Metadata
population = pd.read_csv('tmp-data/code-postal-code-insee-2015.csv', sep=';').clean_names()

# %%
population.query("insee_com == '92023'")

# %%
population = (
    population
    .clean_names()
    .loc[:, ['insee_com', 'population']]
)

# %%
population = (pd.merge(code_insee, population, left_on='code_insee', right_on='insee_com')
 .drop_duplicates()
)

population.to_csv("data/insee-population2015.csv")

# %% ECOLES
# Source: Metadata
ecoles = pd.read_csv("tmp-data/les_etablissements_d_enseignement_des_1er_et_2d_degres_en_idf.csv", sep=';').clean_names()

# %%
ecoles = (
    ecoles
    .assign(code_commune=lambda x: x.code_commune.astype('str'))
    .loc[:, ['secteur_public_prive', 'code_commune',
     'departement', 'denomination_principale', 'nature']]
)

# %%
ecoles = pd.merge(code_insee, ecoles, left_on='code_insee', right_on='code_commune')

# %% Create groups
ecoles = (
    ecoles
    .assign(nature=lambda x: np.where(
        x.nature.str.contains('LYCEE'),
        'lycee',
        np.where(
            x.nature.str.contains('ELEMENTAIRE'),
            'elementaire',
            np.where(
                x.nature.str.contains('MATERNELLE'),
                'maternelle',
                np.where(
                    x.nature.str.contains('COLLEGE'),
                    'college',
                    'autres'
                )
                )
            )
        )
    )
)
    
# %%
ecoles = (
    ecoles
    .groupby(['code_insee', 'nature'], as_index=False)
    .agg(n=('code_commune', 'count'))
    .pivot(index='code_insee', columns='nature', values='n')
    .reset_index()
)

# %%

ecoles.to_csv('data/idf-ecoles.csv', index=False)

# %% AREA VERTS
area_vert = pd.read_csv('tmp-data/espaces-verts-et-boises-surfaciques-ouverts-ou-en-projets-douverture-au-public.csv', sep=';')

# %%
area_vert = area_vert.clean_names()
# %%
area_vert = (
    area_vert
    .assign(insee=lambda x: x.insee.astype('str'))
    .loc[:, ['insee', 'surftotha', 'st_areasha']]
    .query("surftotha > 1")
)

# %%
area_vert = pd.merge(code_insee, area_vert, 
            left_on='code_insee', right_on='insee', how='left')

# %%
area_vert = (
    area_vert
    .groupby('code_insee', as_index=False)
    .agg(
        count_area_vert=('surftotha', 'count'),
        avg_area_vert=('surftotha', 'mean')
    )
    .fillna(0)
)

# %%
area_vert.to_csv("data/idf-area-verts.csv")

# %%
revenus = pd.read_csv('tmp-data/BASE_TD_FILO_DISP_IRIS_2018.csv', sep=';').clean_names()
# %%
import re
revenus = (
    revenus
    .dropna(subset=['iris'])
    .assign(
        iris=lambda x: x.iris.astype('str'),
        insee=lambda x: x.iris.apply(lambda y: re.search('\d{5}', y).group(0))
    )
    .loc[:, ['insee', 'disp_med18']]
)

# %%
revenus = pd.merge(code_insee, revenus, left_on='code_insee', right_on='insee', how='left')
# %%
revenus = (
    revenus
    .groupby('insee', as_index=False)
    .agg(
        med_revenu=('disp_med18', 'mean')
    )
)

# %%
revenus = pd.merge(code_insee, revenus, left_on='code_insee', right_on='insee', how='left')

# %%
revenus.to_csv('data/filosofi-revenus2018.csv')
# %% VELIB
df_velib_cp = pd.read_csv('velib-codepostal.csv')
# %%
velib = (
    df_velib_cp
    .assign(insee=lambda x: x.result_citycode.astype('str'))
    .loc[:, ['insee', 'lat']]
    .groupby('insee', as_index=False)
    .agg(velib_stations=('lat', 'count'))
)

velib = pd.merge(code_insee, velib, left_on='code_insee', right_on='insee', how='left')

# %%
velib.to_csv('velib.csv')
# %% MERGE everything together

df = pd.merge(df_main, population, left_on='code_insee', right_on='code_insee')
# %%
df = pd.merge(df, ecoles, left_on='code_insee', right_on='code_insee')
len(df)
# %%
df = pd.merge(df, revenus, left_on='code_insee', right_on='code_insee')
len(df)
# %%
df = pd.merge(df, velib, left_on='code_insee', right_on='code_insee')
len(df)
# %%
df = pd.merge(df, area_vert, left_on='code_insee', right_on='code_insee')
len(df)
# %%
df.info()
# %% DEAL WITH NAS
df = (
    df
    .assign(
        rev_med_dptm=lambda x: (x.groupby('code_departement')
                                .med_revenu.transform('mean')),
        med_revenu=lambda x: np.where(
            x.med_revenu.isna(),
            x.rev_med_dptm,
            x.med_revenu
        )
    )
    .fillna(value={
        'autres':0,
        'college':0,
        'lycee':0,
        'maternelle':0,
        'velib_stations':0
    })
    .drop(['insee_x', 'insee_y'], 1)
)

# %%
df.info()
# %%
df.to_csv('data/data-final-clean.csv')
# %%
sncf
# %% Get code insee from government API and count gare per insee regions
sncf_ = pd.read_csv('data/sncf-geocoordinates.csv')
sncf_ = sncf_.loc[:, ['latitude', 'longitude']]
sncf_.to_csv('tmp-data/sncf-geo.csv', index=False)
# %%
sncf_insee = pd.read_csv('tmp-data/sncf-insee.csv')
sncf_insee.head()
# %%
sncf_insee = (
    sncf_insee
    .dropna(subset=['result_citycode'])
    .assign(insee=lambda x: x.result_citycode.astype('int').astype('str'))
    .loc[:, ['latitude', 'insee']]
    .groupby('insee', as_index=False)
    .agg(
        sncf_stations=('latitude', 'count')
    )
)

sncf_insee
# %%
sncf_insee.to_csv('data/sncf-gare-count.csv')
# %%
ratp_ = pd.read_csv('data/ratp-geocoordinates.csv')
ratp_ = ratp_.loc[:, ['stop_lat', 'stop_lon']]
ratp_.columns = ['lat', 'lon']
ratp_.to_csv('tmp-data/ratp-geo.csv', index=False)
# %%
ratp_insee = pd.read_csv('tmp-data/ratp-insee.csv')
# %%
ratp_insee = (
    ratp_insee
    .dropna(subset=['result_citycode'])
    .assign(insee=lambda x: x.result_citycode.astype('int').astype('str'))
    .loc[:, ['lat', 'insee']]
    .groupby('insee', as_index=False)
    .agg(
        ratp_stations=('lat', 'count')
    )
)

ratp_insee
# %%
ratp_insee.to_csv('data/ratp-gare-count.csv', index=False)
# %%
