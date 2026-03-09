"""
nlp_engine.py — Sound Prism Vibe-to-DNA NLP Parser
Converts natural language descriptions into audio feature vectors.
No external NLP libraries required — pure Python rule-based system.
"""
import re

# ============================================================
#  VIBE LEXICON — 80+ descriptors → feature deltas
# ============================================================
VIBE_LEXICON = {
    # ── ENERGY ──────────────────────────────────────────────────
    "upbeat":        {"energy": 0.85, "valence": 0.8, "danceability": 0.75, "tempo": 128},
    "banger":        {"energy": 0.95, "danceability": 0.85, "tempo": 130},
    "intense":       {"energy": 0.95, "valence": 0.3, "danceability": 0.6,  "tempo": 140},
    "aggressive":    {"energy": 0.95, "valence": 0.2, "acousticness": 0.05, "tempo": 150},
    "energetic":     {"energy": 0.9,  "danceability": 0.75,                 "tempo": 135},
    "hype":          {"energy": 0.9,  "danceability": 0.8,  "valence": 0.6, "tempo": 140},
    "hyper":         {"energy": 0.95, "danceability": 0.7,  "valence": 0.5, "tempo": 150},
    "mellow":        {"energy": 0.3,  "valence": 0.5, "acousticness": 0.5,  "tempo": 90},
    "slow":          {"energy": 0.25, "danceability": 0.3,                  "tempo": 75},
    "fast":          {"energy": 0.85, "danceability": 0.8,                  "tempo": 150},
    "chill":         {"energy": 0.3,  "valence": 0.55,"acousticness": 0.55, "tempo": 85},
    "calm":          {"energy": 0.2,  "valence": 0.5, "acousticness": 0.65, "tempo": 80},
    "driving":       {"energy": 0.8,  "danceability": 0.7,                  "tempo": 130},
    "powerful":      {"energy": 0.9,  "valence": 0.4, "acousticness": 0.1,  "tempo": 135},
    "soft":          {"energy": 0.2,  "acousticness": 0.7, "valence": 0.55, "tempo": 80},
    "heavy":         {"energy": 0.9,  "acousticness": 0.05, "valence": 0.3},
    "light":         {"energy": 0.3,  "acousticness": 0.8, "valence": 0.6},
    "up tempo":      {"energy": 0.8,  "tempo": 135},
    "downtempo":     {"energy": 0.3,  "tempo": 70},

    # ── MOOD / VALENCE ───────────────────────────────────────────
    "happy":         {"valence": 0.9,  "energy": 0.7, "danceability": 0.7,  "tempo": 120},
    "sad":           {"valence": 0.15, "energy": 0.25,"acousticness": 0.55, "tempo": 75},
    "melancholic":   {"valence": 0.2,  "energy": 0.3, "acousticness": 0.6,  "tempo": 80},
    "euphoric":      {"valence": 0.95, "energy": 0.9, "danceability": 0.85, "tempo": 138},
    "dark":          {"valence": 0.15, "energy": 0.65,"acousticness": 0.2,  "tempo": 100},
    "nostalgic":     {"valence": 0.5,  "energy": 0.4, "acousticness": 0.55, "tempo": 90},
    "romantic":      {"valence": 0.7,  "energy": 0.35,"acousticness": 0.55, "tempo": 85},
    "angry":         {"valence": 0.1,  "energy": 0.95,"acousticness": 0.05, "tempo": 155},
    "peaceful":      {"valence": 0.65, "energy": 0.15,"acousticness": 0.8,  "tempo": 65},
    "anxious":       {"valence": 0.25, "energy": 0.75,"acousticness": 0.15, "tempo": 145},
    "lonely":        {"valence": 0.2,  "energy": 0.2, "acousticness": 0.7,  "tempo": 72},
    "excited":       {"valence": 0.85, "energy": 0.85,"danceability": 0.8,  "tempo": 135},
    "bittersweet":   {"valence": 0.45, "energy": 0.4, "acousticness": 0.5,  "tempo": 88},
    "dreamy":        {"valence": 0.6,  "energy": 0.3, "acousticness": 0.7, "tempo": 85},
    "creepy":        {"valence": 0.1,  "energy": 0.4, "acousticness": 0.4, "tempo": 90},
    "ethereal":      {"valence": 0.5,  "energy": 0.4, "acousticness": 0.6, "instrumentalness": 0.6, "tempo": 95},
    "sadgirl":       {"valence": 0.2,  "energy": 0.3, "acousticness": 0.8, "speechiness": 0.1, "tempo": 80},

    # ── ENVIRONMENT / CONTEXT ────────────────────────────────────
    "rainy":         {"energy": 0.2,  "valence": 0.2, "acousticness": 0.8,  "tempo": 75},
    "rain":          {"energy": 0.2,  "valence": 0.2, "acousticness": 0.8,  "tempo": 75},
    "summer":        {"energy": 0.8,  "valence": 0.9, "danceability": 0.8,  "tempo": 125},
    "night":         {"energy": 0.5,  "valence": 0.4, "acousticness": 0.3,  "tempo": 105},
    "sunset":        {"energy": 0.45, "valence": 0.65,"acousticness": 0.5,  "tempo": 95},
    "midnight":      {"energy": 0.55, "valence": 0.35,"acousticness": 0.2,  "tempo": 108},
    "gym":           {"energy": 0.95, "valence": 0.55,"danceability": 0.7,  "tempo": 145},
    "workout":       {"energy": 0.95, "valence": 0.55,"danceability": 0.7,  "tempo": 145},
    "focus":         {"energy": 0.4,  "valence": 0.5, "instrumentalness": 0.7,"tempo": 95},
    "study":         {"energy": 0.3,  "valence": 0.5, "instrumentalness": 0.65,"tempo": 85},
    "party":         {"energy": 0.9,  "valence": 0.85,"danceability": 0.9,  "tempo": 128},
    "club":          {"energy": 0.9,  "danceability": 0.9, "tempo": 128},
    "road trip":     {"energy": 0.75, "valence": 0.75,"danceability": 0.65, "tempo": 120},
    "morning":       {"energy": 0.5,  "valence": 0.7, "acousticness": 0.4,  "tempo": 100},
    "night drive":   {"energy": 0.65, "valence": 0.4, "acousticness": 0.2,  "tempo": 110},
    "sleep":         {"energy": 0.1,  "valence": 0.5, "acousticness": 0.85, "instrumentalness": 0.7, "tempo": 60},
    "sleepy":        {"energy": 0.15, "valence": 0.4, "acousticness": 0.8, "tempo": 65},
    "meditation":    {"energy": 0.1,  "valence": 0.55,"acousticness": 0.9,  "instrumentalness": 0.85,"tempo": 60},

    # ── TIMBRE / TEXTURE ─────────────────────────────────────────
    "acoustic":      {"acousticness": 0.9,  "energy": 0.3,  "instrumentalness": 0.4},
    "electronic":    {"acousticness": 0.05, "energy": 0.75, "danceability": 0.75},
    "raw":           {"acousticness": 0.6,  "energy": 0.7,  "valence": 0.3},
    "distorted":     {"acousticness": 0.05, "energy": 0.9,  "valence": 0.2},
    "orchestral":    {"instrumentalness": 0.85,"acousticness": 0.5, "energy": 0.55},
    "lo-fi":         {"acousticness": 0.6,  "energy": 0.25, "valence": 0.5,  "tempo": 80},
    "lofi":          {"acousticness": 0.6,  "energy": 0.25, "valence": 0.5,  "tempo": 80},
    "instrumental":  {"instrumentalness": 0.9,"speechiness": 0.03},
    "vocal":         {"instrumentalness": 0.05,"speechiness": 0.1},
    "synth":         {"acousticness": 0.05, "energy": 0.6},
    "bouncy":        {"danceability": 0.9, "energy": 0.75, "valence": 0.8, "tempo": 120},

    # ── GENRE HINTS / SLANG ──────────────────────────────────────
    "jazz":          {"acousticness": 0.6,  "instrumentalness": 0.55,"valence": 0.6, "energy": 0.45, "tempo": 110},
    "classical":     {"acousticness": 0.8,  "instrumentalness": 0.9, "energy": 0.4,  "valence": 0.55},
    "hip-hop":       {"danceability": 0.8,  "speechiness": 0.25,      "energy": 0.75, "tempo": 95},
    "hiphop":        {"danceability": 0.8,  "speechiness": 0.25,      "energy": 0.75, "tempo": 95},
    "rap":           {"speechiness": 0.4,   "danceability": 0.75,     "energy": 0.8,  "tempo": 98},
    "drill":         {"speechiness": 0.3,   "danceability": 0.8,      "energy": 0.85, "tempo": 140, "valence": 0.2},
    "pop":           {"danceability": 0.75, "energy": 0.7,            "valence": 0.75,"acousticness": 0.15},
    "indie":         {"acousticness": 0.5,  "energy": 0.45,           "valence": 0.5, "instrumentalness": 0.1},
    "rock":          {"energy": 0.85,       "acousticness": 0.1,      "valence": 0.45},
    "metal":         {"energy": 0.98,       "valence": 0.15,          "acousticness": 0.02,"tempo": 165},
    "edm":           {"energy": 0.9,        "danceability": 0.9,      "acousticness": 0.02,"tempo": 138},
    "house":         {"energy": 0.8,        "danceability": 0.9,      "tempo": 125, "instrumentalness": 0.4},
    "techno":        {"energy": 0.9,        "danceability": 0.8,      "tempo": 135, "instrumentalness": 0.7, "valence": 0.3},
    "hardstyle":     {"energy": 0.99,       "danceability": 0.6,      "tempo": 150, "acousticness": 0},
    "phonk":         {"energy": 0.8,        "danceability": 0.85,     "tempo": 115, "valence": 0.2, "instrumentalness": 0.5},
    "synthwave":     {"energy": 0.7,        "danceability": 0.6,      "tempo": 100, "instrumentalness": 0.8, "valence": 0.4},
    "dance":         {"danceability": 0.9,  "energy": 0.8,            "valence": 0.75, "tempo": 128},
    "folk":          {"acousticness": 0.75, "energy": 0.3,            "valence": 0.55, "instrumentalness": 0.3},
    "r&b":           {"danceability": 0.75, "valence": 0.65,          "energy": 0.55,  "speechiness": 0.1},
    "rnb":           {"danceability": 0.75, "valence": 0.65,          "energy": 0.55,  "speechiness": 0.1},
    "blues":         {"acousticness": 0.55, "valence": 0.35,          "energy": 0.5,   "tempo": 95},
    "soul":          {"valence": 0.65,      "acousticness": 0.4,      "danceability": 0.6,"energy": 0.55},
    "ambient":       {"energy": 0.1,        "instrumentalness": 0.85, "acousticness": 0.7,"tempo": 70},
    "vaporwave":     {"danceability": 0.6,  "energy": 0.5,            "valence": 0.7,  "acousticness": 0.2},
    "cyberpunk":     {"energy": 0.9,        "danceability": 0.7,      "valence": 0.4,  "acousticness": 0.05, "instrumentalness": 0.5},

    # ── DESCRIPTIVE PHRASES ──────────────────────────────────────
    "cry":           {"valence": 0.1,  "energy": 0.2,  "acousticness": 0.7},
    "tears":         {"valence": 0.1,  "energy": 0.2,  "acousticness": 0.7},
    "groove":        {"danceability": 0.85,"energy": 0.7, "valence": 0.7},
    "headbang":      {"energy": 0.98,  "valence": 0.2,  "acousticness": 0.02},
    "cinematic":     {"instrumentalness": 0.7,"energy": 0.6,"acousticness": 0.3, "tempo": 90},
    "epic":          {"energy": 0.9,   "instrumentalness": 0.6,"valence": 0.6,   "tempo": 120},
}

MODIFIERS = {
    "very": 1.5,
    "super": 1.5,
    "extremely": 1.5,
    "highly": 1.5,
    "slightly": 0.5,
    "a bit": 0.5,
    "kinda": 0.5,
    "sorta": 0.5,
    "really": 1.2
}

NEGATORS = ["not", "no", "anti", "without", "less", "non"]

# Feature keys used in the recommendation engine
FEATURE_KEYS = ["energy", "valence", "danceability", "acousticness",
                "speechiness", "instrumentalness", "tempo"]

# Tag display labels for the UI chips
TAG_LABELS = {
    "energy":           ["Low Energy",  None,             "Mid Energy",  None,             "High Energy"],
    "valence":          ["Dark Mood",   None,             "Neutral Mood",None,             "Happy Mood"],
    "danceability":     ["Not Danceable",None,            "Mid Dance",   None,             "Very Danceable"],
    "acousticness":     ["Electronic",   None,            "Blended",     None,             "Acoustic"],
    "speechiness":      [None,             None,             None,             "Vocal",       "Heavy Vocals"],
    "instrumentalness": [None,             None,             None,             "Instrumental","Pure Instrumental"],
    "tempo":            ["Slow Tempo",  None,             "Mid Tempo",   None,             "Fast Tempo"],
}

BPM_RE = re.compile(r'\b(\d{2,3})\s*bpm\b', re.IGNORECASE)


def parse_vibe(text: str) -> dict:
    text_lower = text.lower().strip()
    
    matched_entries = []
    
    accum = {k: [] for k in FEATURE_KEYS}
    
    # Check for BPM
    bpm_match = BPM_RE.search(text_lower)
    explicit_tempo = None
    if bpm_match:
        explicit_tempo = int(bpm_match.group(1))
    
    # Sort lexicon by word count (longest first)
    sorted_terms = sorted(VIBE_LEXICON.keys(), key=lambda x: len(x.split()), reverse=True)
    
    remaining = text_lower
    
    for term in sorted_terms:
        # Regex to find term as whole word
        pattern = r'\b' + re.escape(term) + r'\b'
        matches = list(re.finditer(pattern, remaining))
        
        for match in matches:
            idx = match.start()
            
            # Look backwards for modifiers or negators
            preceding_text = remaining[:idx].strip()
            preceding_words = preceding_text.split()[-2:] if preceding_text else []
            
            modifier_val = 1.0
            is_negated = False
            
            # Check for modifiers and negators
            for w in preceding_words:
                if w in NEGATORS:
                    is_negated = True
                if w in MODIFIERS:
                    modifier_val *= MODIFIERS[w]
                if w + " " + preceding_words[-1] in MODIFIERS: # e.g. "a bit"
                    modifier_val *= MODIFIERS[w + " " + preceding_words[-1]]
            
            entry = VIBE_LEXICON[term]
            adjusted_entry = {}
            for feat, val in entry.items():
                if feat == "tempo":
                    adjusted_entry[feat] = val if not is_negated else max(60, 200 - val)
                else:
                    base_delta = (val - 0.5) if val != 0.5 else 0
                    if is_negated:
                        base_delta = -base_delta
                    
                    adjusted_delta = base_delta * modifier_val
                    adjusted_val = 0.5 + adjusted_delta
                    adjusted_entry[feat] = max(0.0, min(1.0, adjusted_val))
            
            term_display = term
            if is_negated: term_display = "not " + term
            elif modifier_val != 1.0: term_display = "modified " + term
            
            matched_entries.append((term_display, adjusted_entry))
            
            # replace matched term with spaces so we don't match substrings
            remaining = remaining[:idx] + " " * len(term) + remaining[match.end():]

    if not matched_entries and explicit_tempo is None:
        return {"features": None, "tags": [], "matched_terms": [], "confidence": 0.0}

    # Gather features
    for _, entry in matched_entries:
        for feat, val in entry.items():
            accum[feat].append(val)

    features = {}
    for feat in FEATURE_KEYS:
        vals = accum[feat]
        if vals:
            features[feat] = round(sum(vals) / len(vals), 3)

    if explicit_tempo is not None:
        features["tempo"] = float(explicit_tempo)

    # Defaults
    defaults = {"energy": 0.5, "valence": 0.5, "danceability": 0.5,
                "acousticness": 0.4, "speechiness": 0.05,
                "instrumentalness": 0.0, "tempo": 110.0}
    for k, v in defaults.items():
        if k not in features:
            features[k] = v

    # Clamp
    for k in features:
        if k != "tempo":
            features[k] = max(0.0, min(1.0, features[k]))

    tags = _generate_tags(features)
    confidence = min(1.0, len(matched_entries) * 0.2 + (0.2 if explicit_tempo else 0))

    return {
        "features": features,
        "tags": tags,
        "matched_terms": [t for t, _ in matched_entries],
        "confidence": round(confidence, 2),
    }

def _generate_tags(features: dict) -> list:
    tags = []
    energy = features.get("energy", 0.5)
    if energy < 0.3: tags.append("Low Energy")
    elif energy > 0.75: tags.append("High Energy")

    valence = features.get("valence", 0.5)
    if valence < 0.3: tags.append("Dark Mood")
    elif valence > 0.75: tags.append("Happy Vibe")

    dance = features.get("danceability", 0.5)
    if dance > 0.75: tags.append("Highly Danceable")
    elif dance < 0.3: tags.append("Not Danceable")

    acoustic = features.get("acousticness", 0.5)
    if acoustic > 0.7: tags.append("Acoustic")
    elif acoustic < 0.15: tags.append("Electronic")

    instr = features.get("instrumentalness", 0.0)
    if instr > 0.6: tags.append("Instrumental")

    speech = features.get("speechiness", 0.05)
    if speech > 0.3: tags.append("Vocal-Heavy")

    tempo = features.get("tempo", 110)
    if tempo < 85: tags.append("Slow Tempo")
    elif tempo > 135: tags.append("Fast Tempo")

    return tags

if __name__ == "__main__":
    tests = [
        "not sad rainy day very acoustic folk",
        "extremely intense gym workout without edm",
        "super chill lo-fi study beats 90 bpm",
        "upbeat summer pop party",
        "not happy cyberpunk night drive",
    ]
    for t in tests:
        result = parse_vibe(t)
        print(f"\\n📝 '{t}'")
        print(f"   Terms:  {result['matched_terms']}")
        print(f"   Tags:   {result['tags']}")
        print(f"   Conf:   {result['confidence']}")
        print(f"   DNA:    {result['features']}")
