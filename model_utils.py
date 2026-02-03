import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process

def detect_mood(features):
    """Detects mood based on Valence and Energy."""
    v = features['valence']
    e = features['energy']
    
    if v > 0.6 and e > 0.5:
        return 'happy'
    elif v > 0.4 and e < 0.5:
        return 'chill'
    elif v < 0.4:
        return 'sad'
    else:
        return 'default'

def get_recommendations(song_name, df, song_embeddings, year_range=None, mood=None, top_n=10):
    try:
        # 1. Fuzzy Search
        choices = df['track_name'].tolist()
        best_match = process.extractOne(song_name, choices)
        
        if not best_match or best_match[1] < 60: 
            return None
            
        match_name = best_match[0]
        idx = df[df['track_name'] == match_name].index[0]
        
        # 2. Extract Features & Detect Mood
        chart_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'valence']
        input_features = df.iloc[idx][chart_features].to_dict()
        detected_mood = detect_mood(input_features)

        # 3. Calculate Similarity
        target_vector = song_embeddings[idx].reshape(1, -1)
        similarities = cosine_similarity(target_vector, song_embeddings).flatten()
        related_indices = similarities.argsort()[::-1]
        
        filtered_results = []
        for r_idx in related_indices:
            if r_idx == idx: continue # Skip input song
            
            row = df.iloc[r_idx]
            
            # Era Filter
            if year_range:
                if not (year_range[0] <= row['year'] <= year_range[1]):
                    continue
            
            # Mood Filter
            if mood and mood != 'default':
                if mood == 'happy' and not (row['valence'] > 0.6 and row['energy'] > 0.5): continue
                if mood == 'chill' and not (row['valence'] > 0.4 and row['energy'] < 0.5): continue
                if mood == 'sad' and not (row['valence'] < 0.4): continue

            # Calculate Match Score (0-100)
            score = round(similarities[r_idx] * 100)
            
            # Prepare Song Data
            song_data = df.iloc[r_idx][['track_name', 'artists', 'album', 'year', 'id'] + chart_features].to_dict()
            song_data['match_score'] = score
            
            filtered_results.append(song_data)
            
            if len(filtered_results) >= top_n: break
        
        return {
            "detected_mood": detected_mood,
            "selected": {
                "name": match_name, 
                "artist": df.iloc[idx]['artists'], 
                "features": input_features
            },
            "recommendations": filtered_results
        }

    except Exception as e:
        print(f"Error in logic: {e}")
        return None