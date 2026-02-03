from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    track_name = db.Column(db.String(200), nullable=False)
    artists = db.Column(db.String(200))
    sentiment = db.Column(db.Integer) # 1 for Like, -1 for Dislike
    # Store audio features to track preferences over time
    energy = db.Column(db.Float)
    valence = db.Column(db.Float)