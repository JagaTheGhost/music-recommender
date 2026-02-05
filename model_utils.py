import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process
from config import Config

# 🎨 Placeholder Image (Since CSV has no images)
DEFAULT_IMAGE = "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300&h=300&fit=crop"

class MusicEngine:
    def __init__(self, df, embeddings):
        self.df = df
        self.embeddings = embeddings
        self.mode = Config.DATA_SOURCE

    def _format_spotify_response(self, row, score=None):
        """
        Converts a CSV row into a Fake Spotify API Object.
        """
        return {
            "id": row['id'],
            "name": row['track_name'],
            "year": int(row['year']), 
            "artists": [{"name": row['artists']}], 
            "album": {
                "name": row['album'],
                "images": [{"url": DEFAULT_IMAGE}] 
            },
            "features": {
                "danceability": row['danceability'],
                "energy": row['energy'],
                "valence": row['valence'],
                "acousticness": row['acousticness'],
                "speechiness": row['speechiness']
            },
            "match_score": score if score is not None else 100
        }

    def detect_mood(self, features):
        v, e = features['valence'], features['energy']
        if v > 0.6 and e > 0.5: return 'happy'
        if v > 0.4 and e < 0.5: return 'chill'
        if v < 0.4: return 'sad'
        return 'default'

    def search_and_recommend(self, song_name, year_range, mood, limit):
        if self.mode == 'SPOTIFY':
            return self._recommend_via_api(song_name, year_range, mood, limit)
        else:
            return self._recommend_via_local(song_name, year_range, mood, limit)

    def _recommend_via_local(self, song_name, year_range, mood, limit):
        # 1. Fuzzy Search
        choices = self.df['track_name'].tolist()
        best_match = process.extractOne(song_name, choices)
        
        if not best_match or best_match[1] < 60: 
            return None
            
        match_name = best_match[0]
        idx = self.df[self.df['track_name'] == match_name].index[0]
        
        # 2. Get Input Details
        input_row = self.df.iloc[idx]
        input_data = self._format_spotify_response(input_row)
        detected_mood = self.detect_mood(input_data['features'])

        # 3. Vector Similarity
        target_vector = self.embeddings[idx].reshape(1, -1)
        similarities = cosine_similarity(target_vector, self.embeddings).flatten()
        related_indices = similarities.argsort()[::-1]
        
        # 4. Filter & Format
        results = []
        for r_idx in related_indices:
            if r_idx == idx: continue
            row = self.df.iloc[r_idx]
            
            # Apply Filters
            if year_range and not (year_range[0] <= row['year'] <= year_range[1]): continue
            if mood and mood != 'default':
                v, e = row['valence'], row['energy']
                if mood == 'happy' and not (v > 0.6 and e > 0.5): continue
                if mood == 'chill' and not (v > 0.4 and e < 0.5): continue
                if mood == 'sad' and not (v < 0.4): continue

            # Format Response
            score = round(similarities[r_idx] * 100)
            results.append(self._format_spotify_response(row, score))
            
            if len(results) >= limit: break
        
        return {
            "detected_mood": detected_mood,
            "selected": input_data,
            "recommendations": results
        }

    def _recommend_via_api(self, song_name, year_range, mood, limit):
        # 🚧 FUTURE SPOTIFY LOGIC GOES HERE 🚧
        print("Connecting to Spotify API...")
        # 1. spotipy.search(song_name)
        # 2. spotipy.audio_features(id)
        # 3. spotipy.recommendations(seed_tracks=[id], target_valence=...)
        raise NotImplementedError("API Key missing")