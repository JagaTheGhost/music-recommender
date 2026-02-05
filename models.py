from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    track_name = db.Column(db.String(100), nullable=False)
    sentiment = db.Column(db.Integer, nullable=False) # 1 for Like, -1 for Dislike
    
    # We keep it simple now to match the "Universal Adapter" logic