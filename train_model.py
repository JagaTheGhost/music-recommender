import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# 1. Load Data
df = pd.read_csv('spotify_tracks.csv')
df.columns = df.columns.str.strip().str.lower()

# 2. Select the specific DL features from your image
features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo'
]

# 3. Preprocessing
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features].fillna(0))

# 4. Build the Autoencoder
input_dim = len(features)
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(16, activation='relu')(input_layer)
bottleneck = layers.Dense(8, activation='relu', name='latent_space')(encoded)
decoded = layers.Dense(16, activation='relu')(bottleneck)
output_layer = layers.Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = models.Model(input_layer, output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# 5. Train (Leveraging your GPU)
print("Training started...")
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=64, shuffle=True)

# 6. Save the Encoder and Embeddings
encoder = models.Model(input_layer, bottleneck)
encoder.save('models/encoder_model.h5')

song_embeddings = encoder.predict(X_scaled)
with open('models/song_embeddings.pkl', 'wb') as f:
    pickle.dump(song_embeddings, f)

print("Success! Models saved in /models folder.")