import requests

# Spotify API credentials
SPOTIPY_CLIENT_ID = "584d6c6043e6476389fc218168cf3a4e"
SPOTIPY_CLIENT_SECRET = "f6f7cb30d99643bda29c9d55d4a9b043"

def get_access_token():
    """Get an access token to use with Spotify API."""
    url = "https://accounts.spotify.com/api/token"
    data = {"grant_type": "client_credentials"}
    try:
        response = requests.post(url, data=data, auth=(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET))
        response.raise_for_status()
        token = response.json().get("access_token")
        print("Access token retrieved successfully.")
        return token
    except requests.exceptions.RequestException as e:
        print(f"Error during token retrieval: {e}")
        return None

def fetch_track_details(track_id, token):
    """Fetch details for a single track."""
    url = f"https://api.spotify.com/v1/tracks/{track_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 429:  # Rate limited
            retry_after = int(response.headers.get("Retry-After", 1))
            print(f"Rate limited. Retry after {retry_after} seconds.")
            return None
        response.raise_for_status()
        track = response.json()
        print(f"Track Name: {track['name']}")
        print(f"Artist: {track['artists'][0]['name']}")
        print(f"Album: {track['album']['name']}")
        print(f"Preview URL: {track['preview_url']}")
        return track
    except requests.exceptions.RequestException as e:
        print(f"Error fetching track details: {e}")
        return None

if __name__ == "__main__":
    # Test Spotify track ID (Mr. Brightside by The Killers)
    test_track_id = "11dFghVXANMlKmJXsNCbNl"
    
    print("Authenticating with Spotify API...")
    token = get_access_token()
    if token:
        print("Fetching track details...")
        fetch_track_details(test_track_id, token)
    else:
        print("Failed to authenticate with Spotify API.")