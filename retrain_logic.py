import numpy as np
import pandas as pd
from models import db, Feedback
from flask import Flask

def get_user_preferences(app):
    with app.app_context():
        # 1. Fetch all likes and dislikes
        likes = Feedback.query.filter_by(sentiment=1).all()
        dislikes = Feedback.query.filter_by(sentiment=-1).all()
        
        features = ['energy', 'valence', 'danceability', 'acousticness']
        
        # 2. Calculate average DNA of songs you like vs dislike
        if not likes:
            return {f: 1.0 for f in features} # Default weight is 1.0
            
        like_avg = {f: np.mean([getattr(l, f) for l in likes]) for f in features}
        
        # 3. Create "Weight" adjustments
        # If you like songs with 0.8 energy, but the dataset average is 0.5,
        # we give 'energy' a higher weight in the similarity math.
        weights = {}
        for f in features:
            weights[f] = 1.0 + (like_avg[f] - 0.5) # Basic adjustment logic
            
        return weights