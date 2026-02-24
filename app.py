from flask import Flask, render_template, request, jsonify
import os
from models import db, Feedback
from model_utils import MusicEngine
from retrain_logic import get_user_preferences

app = Flask(__name__)

# DB Setup
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'feedback.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

engine = MusicEngine()

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/retrain', methods=['GET'])
def retrain():
    weights = get_user_preferences(app)
    return jsonify(weights)

@app.route('/stats', methods=['GET'])
def stats():
    weights = get_user_preferences(app)
    # Calculate DNA Deviation (Simplified)
    deviation = {f: round((w - 1.0) * 100, 1) for f, w in weights.items()}
    return jsonify({
        "weights": weights,
        "deviation": deviation
    })

@app.route('/recommend', methods=['POST'])
def recommend():
    req = request.json
    song_name = req.get('song_name', '').strip()
    
    if not song_name:
        return jsonify({"error": "Please enter a song name"}), 400

    limit = int(req.get('limit', 10))
    year_range = req.get('year_range')
    weights = req.get('weights') # New: Personalized weights from frontend
    
    # Parse year_range from frontend [startYear, endYear]
    if year_range and isinstance(year_range, list) and len(year_range) == 2:
        year_range = [int(year_range[0]), int(year_range[1])]
    else:
        year_range = None
    
    # Search for the song
    search_results = engine.search_song(song_name)
    
    if not search_results:
        return jsonify({"error": "No match found"}), 404

    best_match = search_results[0]
    
    # Get recommendations using embeddings + filters + weights
    result = engine.get_recommendations(
        seed_song_name=best_match['name'], 
        limit=limit,
        year_range=year_range,
        weights=weights
    )
    
    if result:
        return jsonify(result)
    else:
        # Fallback with basic match if recommendation failed (e.g. strict filters)
        return jsonify({
            "selected": best_match,
            "recommendations": search_results[1:] if len(search_results) > 1 else []
        })

@app.route('/recommend_mix', methods=['POST'])
def recommend_mix():
    req = request.json
    mix_items = req.get('songs', [])
    limit = int(req.get('limit', 3))
    
    if not mix_items:
        return jsonify({"error": "Mix is empty"}), 400

    # Calculate average features of the mix
    features = ['energy', 'valence', 'danceability', 'acousticness']
    avg_dna = {f: sum([s['features'][f] for s in mix_items]) / len(mix_items) for f in features}
    
    # Get recommendations based on these average features
    result = engine.get_recommendations_by_features(
        avg_dna,
        limit=limit,
        exclude_names=[s['name'] for s in mix_items]
    )
    
    return jsonify(result)

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    entry = Feedback(
        track_name=data.get('track_name', ''),
        artists=data.get('artists', ''),
        sentiment=data.get('sentiment', 0),
        energy=data.get('energy'),
        valence=data.get('valence'),
        danceability=data.get('danceability'),
        acousticness=data.get('acousticness')
    )
    db.session.add(entry)
    db.session.commit()
    return jsonify({"status": "saved"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)