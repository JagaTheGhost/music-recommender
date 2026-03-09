from flask import Flask, render_template, request, jsonify
import os
from model_utils import MusicEngine

app = Flask(__name__)

engine = MusicEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/parse_vibe', methods=['GET'])
def parse_vibe():
    """Live NLP endpoint — called by the frontend as user types a vibe description."""
    text = request.args.get('q', '').strip()
    if not text:
        return jsonify({"features": None, "tags": [], "matched_terms": [], "confidence": 0.0})
    result = engine.parse_vibe_full(text)
    return jsonify(result)

@app.route('/recommend', methods=['POST'])
def recommend():
    req = request.json
    song_name = req.get('song_name', '').strip()
    
    if not song_name:
        return jsonify({"error": "Please enter a song name"}), 400

    limit = int(req.get('limit', 10))
    is_vibe_mode = req.get('is_vibe_mode', False)
    vibe_features = req.get('vibe_features')  # Pre-parsed features from frontend NLP
    allow_explicit = req.get('allow_explicit', True)

    # ── FAST PATH: Vibe mode with pre-parsed features ────────────
    if is_vibe_mode and vibe_features:
        recommendations = engine.get_recommendations_by_features(vibe_features, limit=limit, allow_explicit=allow_explicit)
        vibe_result = engine.parse_vibe_full(song_name)
        return jsonify({
            "selected": {
                "id": "vibe-" + song_name.lower().replace(" ", "-")[:30],
                "name": f'✦ {song_name.upper()}',
                "artists": [{"name": "VIBE PROBE"}],
                "year": "2026",
                "features": vibe_features,
                "genre": "Vibe Discovery",
                "rarity_score": 0,
                "vibe_tags": vibe_result.get('tags', []) if vibe_result else [],
                "vibe_terms": vibe_result.get('matched_terms', []) if vibe_result else [],
            },
            "recommendations": recommendations
        })

    # Search for the song
    search_results = engine.search_song(song_name)
    
    # NEW: Mood/Vibe NLP Fallback
    if not search_results or (len(search_results) > 0 and search_results[0].get('match_score', 0) < 70):
        vibe_result = engine.parse_vibe_full(song_name)
        target_dna = vibe_result.get('features') if vibe_result else None
        if not target_dna:
            target_dna = engine.resolve_mood(song_name)  # Legacy fallback
        if target_dna:
            # Request limit + 1 so we can use the top result as the "selected" card
            recommendations = engine.get_recommendations_by_features(target_dna, limit=limit + 1, allow_explicit=allow_explicit)
            
            if recommendations:
                selected = recommendations.pop(0)
                # Inject the vibe context into the real track so the UI chips display correctly
                selected["vibe_tags"] = vibe_result.get('tags', []) if vibe_result else []
                selected["vibe_terms"] = vibe_result.get('matched_terms', []) if vibe_result else []
                
                return jsonify({
                    "selected": selected,
                    "recommendations": recommendations[:limit]
                })

    if not search_results:
        return jsonify({"error": "No match found. Try a song or mood (e.g. 'Cyberpunk', 'Sunset')"}), 404

    best_match = search_results[0]
    
    # Get recommendations using embeddings + filters
    result = engine.get_recommendations(
        seed_song_name=best_match['name'], 
        limit=limit,
        allow_explicit=allow_explicit
    )
    
    if result:
        return jsonify(result)
    else:
        # Fallback with basic match if recommendation failed (e.g. strict filters)
        return jsonify({
            "selected": best_match,
            "recommendations": search_results[1:] if len(search_results) > 1 else []
        })

@app.route('/recommend_mix', methods=['POST'])
def recommend_mix():
    req = request.json
    mix_items = req.get('songs', [])
    limit = int(req.get('limit', 3))
    allow_explicit = req.get('allow_explicit', True)
    
    if not mix_items:
        return jsonify({"error": "Mix is empty"}), 400

    features_to_blend = ['energy', 'valence', 'danceability', 'acousticness', 'speechiness', 'instrumentalness']

    # NEW: The Alchemist Bridge - Special case for exactly 2 tracks
    if len(mix_items) == 2:
        f1 = mix_items[0].get('features', {})
        f2 = mix_items[1].get('features', {})
        avg_dna = {f: (f1.get(f, 0.5) + f2.get(f, 0.5)) / 2.0 for f in features_to_blend}
        
        recs = engine.get_recommendations_by_features(avg_dna, limit=limit, exclude_names=[s.get('name') for s in mix_items], allow_explicit=allow_explicit)
        if recs:
            return jsonify({
                "type": "bridge",
                "recommendations": recs,
                "note": "Alchemist Bridge: Finding sonic midpoints"
            })
        else:
            return jsonify({"error": "Could not find bridge tracks"}), 404

    # Standard Ensemble Logic
    avg_dna = {f: sum([s.get('features', {}).get(f, 0.5) for s in mix_items]) / len(mix_items) for f in features_to_blend}
    
    recs = engine.get_recommendations_by_features(
        avg_dna,
        limit=limit,
        exclude_names=[s.get('name') for s in mix_items],
        allow_explicit=allow_explicit
    )
    
    if recs:
        return jsonify({
            "type": "ensemble",
            "recommendations": recs
        })
    else:
        return jsonify({"error": "Could not find matching tracks"}), 404



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)