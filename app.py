from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
from models import db, Feedback
from model_utils import get_recommendations

app = Flask(__name__)

# Database Config
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'feedback.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Load Data Once
print("Loading Engine...")
df = pd.read_csv('data/spotify_tracks.csv')
df.columns = df.columns.str.strip().str.lower()

with open('models/song_embeddings.pkl', 'rb') as f:
    song_embeddings = pickle.load(f)

# Initialize DB
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    req = request.json
    
    # Get limit from request, default to 10 if missing
    limit = req.get('limit', 10) 
    
    data = get_recommendations(
        req.get('song_name'), 
        df, 
        song_embeddings, 
        req.get('year_range'), 
        req.get('mood'),
        top_n=limit # Pass it to the function
    )
    return jsonify(data) if data else (jsonify({"error": "No match"}), 404)

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    entry = Feedback(
        track_name=data['track_name'], artists=data['artists'],
        sentiment=data['sentiment'], energy=data['energy'], valence=data['valence']
    )
    db.session.add(entry)
    db.session.commit()
    return jsonify({"status": "saved"})

if __name__ == '__main__':
    app.run(debug=True)