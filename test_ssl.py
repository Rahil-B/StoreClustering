import openrouteservice
import pandas as pd
import ssl
import urllib3
import os
import requests

# Disable SSL warnings and certificate verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Create unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Patch requests to disable SSL verification globally
original_request = requests.request


def patched_request(*args, **kwargs):
    kwargs['verify'] = False
    return original_request(*args, **kwargs)


requests.request = patched_request

# Patch the post method specifically which is used by openrouteservice
original_post = requests.post


def patched_post(*args, **kwargs):
    kwargs['verify'] = False
    return original_post(*args, **kwargs)


requests.post = patched_post

# Test configuration
API_KEY = "your_api_key_here"  # Replace with your actual OpenRouteService API key

print("Testing SSL configuration...")
try:
    client = openrouteservice.Client(key=API_KEY)
    print("✅ Client created successfully!")

    # Test with a simple request (2 cities only)
    # Bangalore coordinates
    test_coords = [[77.5946, 12.9716], [77.6413, 13.0827]]

    print("Testing distance matrix API call...")
    matrix = client.distance_matrix(
        locations=test_coords,
        profile="driving-car",
        metrics=["distance"],
        units="km"
    )

    print("✅ API call successful!")
    print(f"Distance matrix: {matrix['distances']}")

except Exception as e:
    print(f"❌ Error: {e}")
    print(f"Error type: {type(e)}")
