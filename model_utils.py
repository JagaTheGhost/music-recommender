import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
import pickle
import os
try:
    from nlp_engine import parse_vibe as _nlp_parse_vibe
    _NLP_AVAILABLE = True
except ImportError:
    _NLP_AVAILABLE = False


class MusicEngine:
    def __init__(self):
        self.df = None
        self.embeddings = None
        self._load_data()

    def _load_data(self):
        """Loads CSV and PKL files with robust path checking (handles zipped CSV for hosting)."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        csv_name = 'spotify_tracks.csv'
        zip_name = 'spotify_tracks.zip'
        
        # Check if CSV exists, if not, try decompressing ZIP
        csv_path = os.path.join(base_dir, csv_name)
        zip_path = os.path.join(base_dir, zip_name)
        
        # Subdirectory fallback
        if not os.path.exists(csv_path) and not os.path.exists(zip_path):
            csv_path = os.path.join(base_dir, 'models', csv_name)
            zip_path = os.path.join(base_dir, 'models', zip_name)

        if not os.path.exists(csv_path) and os.path.exists(zip_path):
            print(f"📦 Unzipping {zip_name}...")
            try:
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(csv_path))
                print("✅ Unzipped successfully.")
            except Exception as e:
                print(f"❌ Error unzipping: {e}")

        if os.path.exists(csv_path):
            try:
                self.df = pd.read_csv(csv_path)
                self.df.columns = self.df.columns.str.strip().str.lower()
                self.df['search_str'] = self.df['track_name'].astype(str) + " " + self.df['artists'].astype(str)
                print(f"✅ Data Loaded from: {csv_path}")
            except Exception as e:
                print(f"❌ Error reading CSV: {e}")
        else:
            print("❌ CRITICAL ERROR: 'spotify_tracks.csv' or '.zip' not found.")
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
            results.append(self._format_json_song(row, score))
            
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

    def _format_json_song(self, row, score=None, rarity=None):
        """Converts a CSV row into a structured song object with optional score and rarity."""
        features = {
            "danceability": float(row['danceability']),
            "energy": float(row['energy']),
            "valence": float(row['valence']),
            "acousticness": float(row['acousticness']),
            "speechiness": float(row['speechiness']),
            "instrumentalness": float(row.get('instrumentalness', 0)),
            "tempo": float(row.get('tempo', 120))
        }
        
        # Heuristic rarity if not provided: distance from sonic equilibrium (0.5)
        if rarity is None:
            deviation = np.mean([abs(features[f] - 0.5) for f in ['danceability', 'energy', 'valence', 'acousticness']])
            rarity = int(min(100, deviation * 220))

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
            "match_score": score if score else 100,
            "rarity_score": rarity
        }

    def _apply_heuristics(self, related_indices, sim_scores, target_tempo=None):
        """Applies a relevance heuristic to the top 300 matches.
        Boosts newer tracks and penalizes excessively short/long tracks to substitute for lack of popularity data.
        """
        top_indices = related_indices[:300]
        years = pd.to_numeric(self.df.iloc[top_indices]['year'], errors='coerce').fillna(2000).values
        durations = pd.to_numeric(self.df.iloc[top_indices]['duration_ms'], errors='coerce').fillna(200000).values
        
        # Max +0.03 boost for years 1980-2025
        year_boost = np.clip((years - 1980) / 45.0, 0, 1) * 0.03
        # Penalty for intro/outro tracks (<90s) or extremely long tracks (>7m)
        dur_penalty = np.where((durations < 90000) | (durations > 420000), -0.04, 0)
        
        # Tempo penalty logic for Vibe-based searches
        tempo_penalty = np.zeros(len(top_indices))
        if target_tempo is not None:
            tempos = pd.to_numeric(self.df.iloc[top_indices]['tempo'], errors='coerce').fillna(120).values
            bpm_diff = np.abs(tempos - target_tempo)
            # -0.01 penalty for every 5 BPM difference, max penalty of -0.20
            tempo_penalty = -np.clip(bpm_diff / 500.0, 0, 0.20)
        
        adjusted_scores = sim_scores[top_indices] + year_boost + dur_penalty + tempo_penalty
        resorted_args = adjusted_scores.argsort()[::-1]
        
        # Return sorted top indices and their corresponding original scores
        return top_indices[resorted_args], sim_scores[top_indices[resorted_args]]

    def get_recommendations(self, seed_song_name, limit=10, allow_explicit=True):
        """Standard recommendation flow with seed song, using pure embeddings matching."""
        if self.df is None or self.df.empty:
            return None

        idx_list = self.df.index[self.df['track_name'].str.lower() == seed_song_name.lower()].tolist()
        if not idx_list:
            return None

        seed_idx = idx_list[0]
        seed_row = self.df.iloc[seed_idx]
        seed_embedding = self.embeddings[seed_idx].copy().reshape(1, -1)

        sim_scores = cosine_similarity(seed_embedding, self.embeddings).flatten()
        related_indices = sim_scores.argsort()[::-1]
        
        # Apply Heuristics
        final_indices, final_scores = self._apply_heuristics(related_indices, sim_scores)

        recommendations = []
        for i, r_idx in enumerate(final_indices):
            if r_idx == seed_idx: continue
            
            row = self.df.iloc[r_idx]
            
            if not allow_explicit and 'explicit' in row and str(row['explicit']).upper() == 'TRUE':
                continue

            score = int(final_scores[i] * 100)
            recommendations.append(self._format_json_song(row, score))
            if len(recommendations) >= limit: break

        return {
            "selected": self._format_json_song(seed_row, score=100),
            "recommendations": recommendations
        }

    def get_recommendations_by_features(self, target_features, limit=3, exclude_names=None, allow_explicit=True):
        """
        Finds songs closest to a target DNA vector (e.g. average of a playlist).
        """
        if self.df is None or self.df.empty:
            return []

        # Unified feature set (matches get_recommendations) to prevent silent feature drops
        chart_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'valence', 'instrumentalness']
        target_vector = np.array([target_features.get(f, 0.5) for f in chart_features]).reshape(1, -1)
        
        feature_matrix = self.df[chart_features].values
        sim_scores = cosine_similarity(target_vector, feature_matrix).flatten()
        
        # Initial sort
        related_indices = sim_scores.argsort()[::-1]
        
        # Apply Heuristics with calculated target tempo
        target_tempo = target_features.get('tempo')
        final_indices, final_scores = self._apply_heuristics(related_indices, sim_scores, target_tempo=target_tempo)
        
        results = []
        exclude_set = set([n.lower() for n in exclude_names]) if exclude_names else set()
        
        for i, r_idx in enumerate(final_indices):
            row = self.df.iloc[r_idx]
            if row['track_name'].lower() in exclude_set:
                continue
                
            if not allow_explicit and 'explicit' in row and str(row['explicit']).upper() == 'TRUE':
                continue
                
            score = int(final_scores[i] * 100)
            results.append(self._format_json_song(row, score))
            
            if len(results) >= limit:
                break
                
        return results

    def get_bridge_recommendation(self, song_a_name, song_b_name, limit=5, allow_explicit=True):
        """Finds tracks that are sonically midway between two target tracks."""
        idx_a = self.df.index[self.df['track_name'].str.lower() == song_a_name.lower()].tolist()
        idx_b = self.df.index[self.df['track_name'].str.lower() == song_b_name.lower()].tolist()
        
        if not idx_a or not idx_b: return None
        
        row_a = self.df.iloc[idx_a[0]]
        row_b = self.df.iloc[idx_b[0]]
        
        chart_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'valence', 'instrumentalness']
        feat_a = np.array([float(row_a.get(f, 0.5)) for f in chart_features])
        feat_b = np.array([float(row_b.get(f, 0.5)) for f in chart_features])
        
        # Calculate Midpoint Vector
        midpoint = (feat_a + feat_b) / 2.0
        
        return self.get_recommendations_by_features({chart_features[i]: midpoint[i] for i in range(len(chart_features))}, limit=limit, allow_explicit=allow_explicit)

    def resolve_mood(self, mood_query):
        """
        Maps a natural language string to a target DNA feature profile.
        """
        if _NLP_AVAILABLE:
            result = _nlp_parse_vibe(mood_query)
            if result and result.get('features') and result['confidence'] > 0:
                print(f"🧠 NLP parse: terms={result['matched_terms']} conf={result['confidence']}")
                return result['features']

        return None

    def parse_vibe_full(self, text):
        """
        Returns the full NLP result dict (features + tags + matched_terms + confidence).
        Used by the /parse_vibe API endpoint for live chip previews.
        """
        if _NLP_AVAILABLE:
            return _nlp_parse_vibe(text)
        return {"features": None, "tags": [], "matched_terms": [], "confidence": 0.0}
