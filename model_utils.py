import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
import pickle
import os


class MusicEngine:

    def __init__(self):
        self.df = None
        self.embeddings = None
        self._load_data()

    def _load_data(self):
        """Loads CSV and PKL files with robust path checking."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define possible paths for the CSV
        possible_csv_paths = [
            os.path.join(base_dir, 'models', 'spotify_tracks.csv'),
            os.path.join(base_dir, 'spotify_tracks.csv')
        ]
        
        csv_path = None
        for path in possible_csv_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path:
            try:
                self.df = pd.read_csv(csv_path)
                self.df.columns = self.df.columns.str.strip().str.lower()
                self.df['search_str'] = self.df['track_name'].astype(str) + " " + self.df['artists'].astype(str)
                print(f"✅ Data Loaded from: {csv_path}")
            except Exception as e:
                print(f"❌ Error reading CSV: {e}")
        else:
            print("❌ CRITICAL ERROR: 'spotify_tracks.csv' not found in 'models/' or root folder.")
            self.df = pd.DataFrame()

        # Load Embeddings (PKL)
        possible_pkl_paths = [
            os.path.join(base_dir, 'models', 'song_embeddings.pkl'),
            os.path.join(base_dir, 'song_embeddings.pkl')
        ]
        
        for path in possible_pkl_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        self.embeddings = pickle.load(f)
                    print(f"✅ Embeddings Loaded from: {path}")
                    break
                except Exception as e:
                    print(f"❌ Error reading PKL: {e}")


    def search_song(self, query):
        """Local fuzzy search on CSV data."""
        if self.df is None or self.df.empty:
            print("❌ Error: No local data available.")
            return []

        if 'search_str' not in self.df.columns:
            self.df['search_str'] = self.df['track_name'].astype(str) + " " + self.df['artists'].astype(str)

        choices = self.df['search_str'].tolist()
        matches = process.extract(query, choices, limit=5, scorer=fuzz.partial_ratio)

        results = []
        for match_str, score, idx in matches:
            row = self.df.iloc[idx]
            results.append(self._format_csv_row(row, score))
            
        return results

    @staticmethod
    def _infer_genre(features):
        """Infers a genre label based on audio DNA features."""
        energy = features.get('energy', 0.5)
        valence = features.get('valence', 0.5)
        dance = features.get('danceability', 0.5)
        acoustic = features.get('acousticness', 0.5)
        speech = features.get('speechiness', 0.5)
        instr = features.get('instrumentalness', 0.0)

        if speech > 0.6:
            return "Hip-Hop"
        elif energy > 0.8 and valence < 0.5:
            return "Rock/Metal"
        elif dance > 0.7 and energy > 0.6:
            return "Pop/Dance"
        elif acoustic > 0.7:
            return "Acoustic"
        elif instr > 0.7:
            return "Instrumental"
        elif energy < 0.4 and valence < 0.4:
            return "Ambient/Chill"
        else:
            return "Alternative"

    @staticmethod
    def _parse_artists(artist_str):
        """Robustly parses artist strings from CSV, handling ['Name'] format."""
        import ast
        try:
            if isinstance(artist_str, str) and artist_str.startswith('['):
                artists = ast.literal_eval(artist_str)
                return [{"name": a} for a in artists]
            else:
                return [{"name": str(artist_str)}]
        except Exception:
            # Fallback for malformed strings
            return [{"name": str(artist_str).strip("[]'\"")}]

    def _format_csv_row(self, row, score=None):
        """Converts a CSV row into a structured song object."""
        features = {
            "danceability": float(row['danceability']),
            "energy": float(row['energy']),
            "valence": float(row['valence']),
            "acousticness": float(row['acousticness']),
            "speechiness": float(row['speechiness']),
            "instrumentalness": float(row.get('instrumentalness', 0))
        }
        
        return {
            "id": str(row['id']),
            "name": row['track_name'],
            "year": int(row['year']) if 'year' in row else 2020,
            "artists": self._parse_artists(row['artists']),
            "album": {
                "name": row['album'] if 'album' in row else "Unknown",
            },
            "genre": self._infer_genre(features),
            "features": features,
            "match_score": score if score else 100
        }

    def get_recommendations(self, seed_song_name, limit=10, year_range=None, weights=None):
        """
        Uses Local Vector Embeddings to find similar songs.
        Supports year_range and personalized weights.
        """
        if self.df is None or self.df.empty or self.embeddings is None:
            return None

        idx_list = self.df.index[self.df['track_name'].str.lower() == seed_song_name.lower()].tolist()
        
        if not idx_list:
            return None

        seed_idx = idx_list[0]
        seed_row = self.df.iloc[seed_idx]

        # Extract features
        chart_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'valence']
        input_features = {f: float(seed_row[f]) for f in chart_features}

        # Calculate Cosine Similarity
        if weights:
            feature_matrix = self.df[chart_features].values
            weight_vector = np.array([weights.get(f, 1.0) for f in chart_features])
            weighted_matrix = feature_matrix * weight_vector
            seed_row_features = np.array([float(seed_row[f]) for f in chart_features]).reshape(1, -1)
            weighted_seed = seed_row_features * weight_vector
            sim_scores = cosine_similarity(weighted_seed, weighted_matrix).flatten()
        else:
            seed_vector = self.embeddings[seed_idx].reshape(1, -1)
            sim_scores = cosine_similarity(seed_vector, self.embeddings).flatten()

        related_indices = sim_scores.argsort()[::-1]

        # Filter and collect results
        recommendations = []
        for r_idx in related_indices:
            if r_idx == seed_idx:
                continue
            
            row = self.df.iloc[r_idx]
            
            if year_range:
                row_year = int(row['year']) if 'year' in row else 2020
                if not (year_range[0] <= row_year <= year_range[1]):
                    continue
            

            score = int(sim_scores[r_idx] * 100)
            recommendations.append(self._format_csv_row(row, score))
            
            if len(recommendations) >= limit:
                break

        selected = self._format_csv_row(seed_row, 100)
        selected['features'] = input_features

        return {
            "selected": selected,
            "recommendations": recommendations
        }

    def get_recommendations_by_features(self, target_features, limit=3, exclude_names=None):
        """
        Finds songs closest to a target DNA vector (e.g. average of a playlist).
        """
        if self.df is None or self.df.empty:
            return []

        chart_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'valence']
        target_vector = np.array([target_features.get(f, 0.5) for f in chart_features]).reshape(1, -1)
        
        feature_matrix = self.df[chart_features].values
        sim_scores = cosine_similarity(target_vector, feature_matrix).flatten()
        
        related_indices = sim_scores.argsort()[::-1]
        
        results = []
        exclude_set = set([n.lower() for n in exclude_names]) if exclude_names else set()
        
        for r_idx in related_indices:
            row = self.df.iloc[r_idx]
            if row['track_name'].lower() in exclude_set:
                continue
                
            score = int(sim_scores[r_idx] * 100)
            results.append(self._format_csv_row(row, score))
            
            if len(results) >= limit:
                break
                
        return results
