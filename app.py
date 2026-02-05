from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
from models import db, Feedback
from model_utils import MusicEngine

app = Flask(__name__)

# DB Setup
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'feedback.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Initialize Engine
print("Loading Data...")
df = pd.read_csv('data/spotify_tracks.csv')
df.columns = df.columns.str.strip().str.lower()

with open('models/song_embeddings.pkl', 'rb') as f:
    song_embeddings = pickle.load(f)

# Create the Engine Instance
engine = MusicEngine(df, song_embeddings)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    req = request.json
    
    data = engine.search_and_recommend(
        song_name=req.get('song_name'),
        year_range=req.get('year_range'),
        mood=req.get('mood'),
        limit=int(req.get('limit', 10))
    )
    
    if data:
        return jsonify(data)
    return jsonify({"error": "No match found"}), 404

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    # Note: We now expect clean data, so we might need to adjust what we save
    entry = Feedback(
        track_name=data['track_name'], 
        sentiment=data['sentiment']
    )
    db.session.add(entry)
    db.session.commit()
    return jsonify({"status": "saved"})

if __name__ == '__main__':
    app.run(debug=True)