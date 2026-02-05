import os

class Config:
    # Change to 'SPOTIFY' later when you get an API Key
    DATA_SOURCE = 'LOCAL' 
    
    # Credentials (leave empty for now)
    SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID')
    SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET')