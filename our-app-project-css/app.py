from flask import Flask, render_template, request, url_for, flash, redirect, Response
import pandas as pd
import numpy as np
import scipy as sp
import joblib
import requests


app = Flask(__name__)


@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method == "POST":
        adresse = request.form["adresse"]

        ville = request.form["ville"]

        local = request.form["local"]

        try:
            X = get_all_infos(adresse, ville, local)

            y_pred = loaded_model.predict(X)

            y_pred = int(y_pred[0])

            message = 'Estimated price per squared meter: ' + str(y_pred) + ' Euros'

        except:
            return render_template('error.html')

        return render_template(
            'arrive.html', 
            message=message
            )
    
    #if not adresse:
    #    flash('Adresse is required!')
    #else:
    #    return render_template('arrive.html', message=adresse)

    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

'''
def index():
    if request.method == "POST":
        title = request.form['title']

        if not title:
            flash('URL is required!')

        else:
            try:
                message, df = find_wiki_way(title)
                plot_wiki(df)
            except:
                return render_template('error.html')
            
            return render_template('arrive.html', message=message)

    return render_template('index.html')
'''
    
'''
@def.route('/dash/') 
    def dashboard():
    return render_template("dashboard.html")

@app.route('/404')
def page_non_trouvee():
    return "Cette page devrait vous avoir renvoyé une erreur 404"
'''


if __name__ == '__main__':
    app.run(debug=True)


## PYTHON FUNCTIONS

cities = pd.read_csv('static/cities-clean.csv')

def get_citycode(city):
    df = cities[cities.commune == city]
    return df.reset_index().code_insee[0]

def get_geocords(address, insee):
    address = str(address)
    insee = str(insee)
    
    params = (
        ('q', address),
        ('citycode', insee)
    )

    response = requests.get('https://api-adresse.data.gouv.fr/search/', params=params)

    response_json = response.json()['features'][0]

    if response_json['properties']['score'] < 0.5:
        raise ValueError('Invalid address...')
    else:
        longitude, latitude = response_json['geometry']['coordinates']

    return longitude, latitude

def extract_query(address, city):

    insee = get_citycode(city)
    longitude, latitude = get_geocords(address, insee)

    return insee, longitude, latitude

# %%
#insee, lon, lat = extract_query('5 rue roger poncelet', 'ASNIERES-SUR-SEINE')

# %%
#insee
# %% GET LOCATION FUNCTIONS

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
sncf = pd.read_csv('static/sncf-geocoordinates.csv')
sncf_lat = sncf.latitude.to_numpy()
sncf_lon = sncf.longitude.to_numpy()

sncf_coords = np.array([[sncf_lat[i], sncf_lon[i]] for i in range(len(sncf_lat))])

# %%
ratp = pd.read_csv('static/ratp-geocoordinates.csv')
ratp_lat = ratp.stop_lat.to_numpy()
ratp_lon = ratp.stop_lon.to_numpy()

ratp_coords = np.array([[ratp_lat[i], ratp_lon[i]] for i in range(len(ratp_lat))])

# %%
velib_geo = pd.read_csv("static/velib-geo.csv")
velib_lat = velib_geo.lat.to_numpy()
velib_lon = velib_geo.lon.to_numpy()

velib_coords = np.array([[velib_lat[i], velib_lon[i]] for i in range(len(velib_lat))])
# %%
new_stations = pd.read_csv('static/new-stations.csv')
new_lat = new_stations.latitude.to_numpy()
new_lon = new_stations.longitude.to_numpy()

new_coords = np.array([[new_lat[i], new_lon[i]] for i in range(len(new_lat))])

# %%
def get_dist_infos(lat, lon):
    sncf_array = list(
        map(lambda y: spherical_dist(np.array((lat, lon)), y), sncf_coords)
    )

    n1_sncf, n2_sncf, n3_sncf, dist_sncf = retrieve_info(sncf_array)

    ratp_array = list(
        map(lambda y: spherical_dist(np.array((lat, lon)), y), ratp_coords)
    )

    n1_ratp, n2_ratp, n3_ratp, dist_ratp = retrieve_info(ratp_array)

    new_array = list(
        map(lambda y: spherical_dist(np.array((lat, lon)), y), new_coords)
    )

    n1_new, n2_new, n3_new, dist_new = retrieve_info(new_array)

    velib_array = list(
        map(lambda y: spherical_dist(np.array((lat, lon)), y), velib_coords)
    )

    n1_velib, n2_velib, n3_velib, dist_velib = retrieve_info(velib_array)

    gare_proche = min(dist_sncf, dist_ratp)
    gare_proche_sq = gare_proche ** 2

    return gare_proche, gare_proche_sq, n3_sncf, dist_new, n3_velib
# %%
#gare_proche, gare_proche_sq, n3_sncf, dist_new, n3_velib = get_dist_infos(lat, lon)

# %%
num_cols = [
    'gare_proche',
    'gare_proche_sq',
    'n3_sncf',
    'dist_new',
    'n3_velib',
    ]

cat_cols = ['year', 'type_local', 'code_insee', 'code_departement']

def get_all_infos(adresse, ville, local):
    
    insee, lon, lat = extract_query(adresse, ville)

    gare_proche, gare_proche_sq, n3_sncf, dist_new, n3_velib = get_dist_infos(lat, lon)

    year = '2020'

    code_departement = str(insee)[:2]

    df = pd.DataFrame({
        'gare_proche':gare_proche,
        'gare_proche_sq':gare_proche_sq,
        'n3_sncf':n3_sncf,
        'dist_new':dist_new,
        'n3_velib':n3_velib,
        'year':year,
        'code_insee':insee,
        'type_local':local,
        'code_departement':code_departement
    }, index=[0])


    num_cols = [
    'gare_proche',
    'gare_proche_sq',
    'n3_sncf',
    'dist_new',
    'n3_velib',
    ]

    cat_cols = ['year', 'type_local', 'code_insee', 'code_departement']

    relevant_cols = [
    y for x in [x for x in 
        [num_cols, cat_cols]#, ['nombre_pieces_principales']]
        ] for y in x
    ]

    df = df.loc[:, relevant_cols]

    return df

# %%
#X = get_all_infos('5 rue roger poncelet', 'ASNIERES-SUR-SEINE', 'Appartement')

# %%
loaded_model = joblib.load('static/finalized_model.sav')