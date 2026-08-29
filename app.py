"""
Flask web app integrating BOTH prediction systems into one site.
Theme: beautiful golden / white.
"""
import os
from flask import Flask, render_template, request, jsonify

import ml_pipeline as ml

app = Flask(__name__)

# Cache trained models so we only train once per process
CACHE = {}


def get_house_model():
    if 'house' not in CACHE:
        model, scaler, encoders, features, metrics, feat_imp = ml.train_house()
        CACHE['house'] = dict(model=model, scaler=scaler, encoders=encoders,
                              features=features, metrics=metrics, feat_imp=feat_imp)
    return CACHE['house']


def get_spotify_model():
    if 'spotify' not in CACHE:
        model, scaler, encoders, features, metrics, feat_imp = ml.train_spotify()
        CACHE['spotify'] = dict(model=model, scaler=scaler, encoders=encoders,
                                features=features, metrics=metrics, feat_imp=feat_imp)
    return CACHE['spotify']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/house')
def house():
    h = get_house_model()
    df = ml.load_and_clean_house()
    graphs = ml.house_graphs(df)
    return render_template('house.html',
                           metrics=h['metrics'],
                           feat_imp=h['feat_imp'],
                           graphs=graphs,
                           shape=df.shape)


@app.route('/house/predict', methods=['POST'])
def house_predict():
    h = get_house_model()
    form = request.form.to_dict()
    pred = ml.predict_house(h['model'], h['scaler'], h['encoders'], h['features'], form)
    return jsonify({'prediction': pred, 'unit': 'Crore'})


@app.route('/spotify')
def spotify():
    s = get_spotify_model()
    df = ml.load_and_clean_spotify()
    graphs = ml.spotify_graphs(df)
    genres = sorted(df['playlist_genre'].unique().tolist())
    subgenres = sorted(df['playlist_subgenre'].unique().tolist())
    return render_template('spotify.html',
                           metrics=s['metrics'],
                           feat_imp=s['feat_imp'],
                           graphs=graphs,
                           shape=df.shape,
                           genres=genres,
                           subgenres=subgenres)


@app.route('/spotify/predict', methods=['POST'])
def spotify_predict():
    s = get_spotify_model()
    form = request.form.to_dict()
    pred = ml.predict_spotify(s['model'], s['scaler'], s['encoders'], s['features'], form)
    return jsonify({'prediction': pred, 'unit': 'popularity score (0-100)'})


@app.route('/analyze')
def analyze():
    h = get_house_model()
    s = get_spotify_model()
    return render_template('analyze.html',
                           house_metrics=h['metrics'],
                           spotify_metrics=s['metrics'],
                           house_shape=ml.load_and_clean_house().shape,
                           spotify_shape=ml.load_and_clean_spotify().shape)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
