from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    track_name = db.Column(db.String(200), nullable=False)
    artists = db.Column(db.String(200), nullable=True)
    sentiment = db.Column(db.Integer, nullable=False)  # 1 for Like, -1 for Dislike
    # Audio features for retraining
    energy = db.Column(db.Float, nullable=True)
    valence = db.Column(db.Float, nullable=True)
    danceability = db.Column(db.Float, nullable=True)
    acousticness = db.Column(db.Float, nullable=True)