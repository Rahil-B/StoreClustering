import requests.adapters
import openrouteservice
import pandas as pd
import ssl
import urllib3
import os
import requests
import math
import numpy as np
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# COMPREHENSIVE SSL DISABLING
print("Configuring SSL settings...")

# 1. Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 2. Set environment variables
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

# 3. Create unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# 4. Globally patch requests to disable SSL verification
original_send = requests.adapters.HTTPAdapter.send


def no_ssl_send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
    # Force verify to False to disable SSL verification
    return original_send(self, request, stream=stream, timeout=timeout, verify=False, cert=cert, proxies=proxies)


# Use setattr to avoid type checker issues
setattr(requests.adapters.HTTPAdapter, 'send', no_ssl_send)

print("[SUCCESS] SSL configuration complete")

# ------------------------
# Load Environment Variables
# ------------------------
print("Loading environment variables...")
load_dotenv()  # Load environment variables from .env file
print("[SUCCESS] Environment variables loaded")

# ------------------------
# Config
# ------------------------
# Get configuration from environment variables
API_KEY = os.getenv("API_KEY")
MAX_DISTANCE_KM = int(os.getenv("MAX_DISTANCE_KM", 70)
                      )  # Default to 70 if not set

if not API_KEY:
    raise ValueError(
        "API_KEY not found in environment variables. Please check your .env file.")

print(f"[CONFIG] Using MAX_DISTANCE_KM: {MAX_DISTANCE_KM} km")
print(
    f"[CONFIG] API_KEY loaded: {'*' * (len(API_KEY) - 10) + API_KEY[-10:] if API_KEY else 'NOT FOUND'}")

# ------------------------
# Load Data
# ------------------------
# Columns: City, Latitude, Longitude
print("Loading CSV data...")
coordinate_file = os.getenv("COORDINATE_FILE")
if not coordinate_file:
    raise ValueError(
        "COORDINATE_FILE not found in environment variables. Please check .env file.")

try:
    df = pd.read_csv(coordinate_file, encoding='utf-8')
    print(
        f"[SUCCESS] Successfully loaded {len(df)} cities with UTF-8 encoding")
except UnicodeDecodeError:
    try:
        df = pd.read_csv(coordinate_file, encoding='latin-1')
        print(
            f"[SUCCESS] Successfully loaded {len(df)} cities with Latin-1 encoding")
    except UnicodeDecodeError:
        df = pd.read_csv(coordinate_file, encoding='cp1252')
        print(
            f"[SUCCESS] Successfully loaded {len(df)} cities with CP1252 encoding")

print(f"Columns in dataset: {list(df.columns)}")
print(f"First few rows:\n{df.head()}")

# Initialize ORS client
print("Initializing OpenRouteService client...")
client = openrouteservice.Client(key=API_KEY)
print("[SUCCESS] Client initialized successfully")

# ------------------------
# Helper Functions
# ------------------------


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * \
        math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6371
    return c * r

# ------------------------
# Caching Functions
# ------------------------


def get_cache_key(coords):
    """Generate a unique cache key for a set of coordinates"""
    # Sort coordinates to ensure consistent key regardless of order
    sorted_coords = sorted(coords)
    coord_str = json.dumps(sorted_coords, sort_keys=True)
    return hashlib.md5(coord_str.encode()).hexdigest()


def load_distance_cache():
    """Load the distance cache from file"""
    cache_file = Path("data/distance_cache.json")
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            print(f"[SUCCESS] Loaded distance cache with {len(cache)} entries")
            return cache
        except Exception as e:
            print(f"[WARNING] Could not load cache file: {e}")
            return {}
    else:
        print("[INFO] No cache file found, starting with empty cache")
        return {}


def save_distance_cache(cache):
    """Save the distance cache to file"""
    try:
        with open("data/distance_cache.json", 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"[SUCCESS] Saved distance cache with {len(cache)} entries")
    except Exception as e:
        print(f"[WARNING] Could not save cache file: {e}")


def get_cached_or_fetch_distances(client, coords, cache, cache_stats):
    """Get distances from cache or fetch from API"""
    cache_key = get_cache_key(coords)

    if cache_key in cache:
        print(f"[CACHE HIT] Using cached distances for {len(coords)} cities")
        cache_stats['hits'] += 1
        return cache[cache_key]
    else:
        print(
            f"[CACHE MISS] Fetching distances from API for {len(coords)} cities")
        cache_stats['misses'] += 1

        # Fetch from API
        matrix = client.distance_matrix(
            locations=coords,
            profile="driving-car",
            metrics=["distance"],
            units="km"
        )

        distances = matrix["distances"]

        # Store in cache
        cache[cache_key] = distances

        # Save cache immediately to prevent loss
        save_distance_cache(cache)

        return distances


def create_initial_clusters(coords, city_names, max_cluster_size=50, proximity_km=100):
    """
    Create initial clusters using mathematical distance (Haversine)
    """
    clusters = []
    assigned = set()

    for i, (lon, lat) in enumerate(coords):
        if i in assigned:
            continue

        # Start a new cluster with current city as center
        cluster_indices = [i]
        cluster_coords = [(lon, lat)]
        cluster_cities = [city_names[i]]
        assigned.add(i)

        # Find nearby cities within proximity_km using Haversine distance
        for j, (other_lon, other_lat) in enumerate(coords):
            if j in assigned or len(cluster_indices) >= max_cluster_size:
                continue

            distance = haversine_distance(lat, lon, other_lat, other_lon)
            if distance <= proximity_km:
                cluster_indices.append(j)
                cluster_coords.append((other_lon, other_lat))
                cluster_cities.append(city_names[j])
                assigned.add(j)

        clusters.append({
            'indices': cluster_indices,
            'coords': cluster_coords,
            'cities': cluster_cities,
            'center_city': city_names[i]
        })

    return clusters


# ------------------------
# Two-Level Clustering Approach
# ------------------------
print("Starting 2-level clustering approach...")

# Initialize distance cache
print("Initializing distance cache...")
distance_cache = load_distance_cache()
cache_stats = {'hits': 0, 'misses': 0}

print("Level 1: Creating initial clusters using mathematical distance (Haversine)")

coords = list(zip(df["Longitude"], df["Latitude"]))  # ORS expects (lon, lat)
city_names = df["City"].tolist()

# Level 1: Create initial clusters using mathematical distance
# Use a larger proximity for initial clustering (e.g., 100km) to group nearby cities
initial_clusters = create_initial_clusters(
    coords, city_names, max_cluster_size=40, proximity_km=80)

print(
    f"[SUCCESS] Created {len(initial_clusters)} initial clusters using mathematical distance")
for i, cluster in enumerate(initial_clusters):
    print(
        f"Cluster {i+1}: {len(cluster['cities'])} cities (Center: {cluster['center_city']})")

print("\nLevel 2: Refining clusters using precise route distances...")

# Level 2: Within each cluster, use OpenRouteService for precise route distances
final_groups = []
total_api_calls = 0

for cluster_idx, cluster in enumerate(initial_clusters):
    print(
        f"\nProcessing cluster {cluster_idx + 1}/{len(initial_clusters)}: {cluster['center_city']}")

    cluster_coords = cluster['coords']
    cluster_cities = cluster['cities']
    cluster_size = len(cluster_coords)

    if cluster_size == 1:
        # Single city cluster - no API call needed
        final_groups.append({
            "Center_City": cluster['center_city'],
            "Cities": cluster_cities,
            "Avg_Distance_From_Center_KM": 0.0,
            "Max_Distance_From_Center_KM": 0.0,
            "City_Count": 1,
            "Individual_Distances": [0.0]
        })
        print(f"Single city cluster - no API call needed")
        continue

    print(f"Getting precise route distances for {cluster_size} cities...")

    try:
        # Get distance matrix for this cluster (with caching)
        distances = get_cached_or_fetch_distances(
            client, cluster_coords, distance_cache, cache_stats)

        total_api_calls += cache_stats['misses']  # Only count actual API calls
        print(
            f"[SUCCESS] Retrieved distance matrix (Total API calls: {cache_stats['misses']}, Cache hits: {cache_stats['hits']})")

        # Apply final clustering within this group using precise distances and MAX_DISTANCE_KM
        cluster_assigned = set()

        for i, city in enumerate(cluster_cities):
            if city in cluster_assigned:
                continue

            center_city = city
            center_city_index = i
            group = [center_city]
            group_distances = []  # Only store distances from center to other cities
            cluster_assigned.add(center_city)

            # Find cities within MAX_DISTANCE_KM using route distances
            for j, other_city in enumerate(cluster_cities):
                if i != j and other_city not in cluster_assigned and distances[i][j] <= MAX_DISTANCE_KM:
                    group.append(other_city)
                    group_distances.append(distances[i][j])
                    cluster_assigned.add(other_city)

            # Calculate average distance from center (only for non-center cities)
            if len(group_distances) > 0:
                avg_distance_from_center = sum(
                    group_distances) / len(group_distances)
            else:
                avg_distance_from_center = 0.0  # Single city group

            final_groups.append({
                "Center_City": center_city,
                "Cities": group,
                "Avg_Distance_From_Center_KM": round(avg_distance_from_center, 2),
                "Max_Distance_From_Center_KM": round(max(group_distances), 2) if group_distances else 0.0,
                "City_Count": len(group),
                # Center city distance is 0, then other cities
                "Individual_Distances": [0.0] + [round(d, 2) for d in group_distances]
            })

        print(
            f"[SUCCESS] Created {len([g for g in final_groups if g['Center_City'] in cluster_cities])} final groups from this cluster")

    except Exception as e:
        print(f"[ERROR] Error processing cluster {cluster_idx + 1}: {e}")
        # Fallback: treat each city as its own group
        for city in cluster_cities:
            final_groups.append({
                "Center_City": city,
                "Cities": [city],
                "Avg_Distance_From_Center_KM": 0.0,
                "Max_Distance_From_Center_KM": 0.0,
                "City_Count": 1,
                "Individual_Distances": [0.0]
            })
        print(
            f"[FALLBACK] Created individual groups for {cluster_size} cities")

print(f"\n[SUCCESS] 2-level clustering complete!")
print(f"Total API calls made: {total_api_calls}")
print(f"Created {len(final_groups)} final city groups")

# ------------------------
# Save Results
# ------------------------
print("\nSaving results...")
group_df = pd.DataFrame(final_groups)
group_df.insert(0, "Group_ID", range(1, len(final_groups) + 1))

# Save main results file
main_output_file = str(MAX_DISTANCE_KM) + \
    "_city_groups_nonoverlap_with_center.csv"
group_df.to_csv(main_output_file, index=False)

# Create a detailed analysis DataFrame for better readability
detailed_analysis = []
for idx, group in enumerate(final_groups, 1):
    for i, city in enumerate(group['Cities']):
        detailed_analysis.append({
            'Group_ID': idx,
            'Center_City': group['Center_City'],
            'City': city,
            'Distance_From_Center_KM': group['Individual_Distances'][i],
            'Group_Avg_Distance_KM': group['Avg_Distance_From_Center_KM'],
            'Group_Max_Distance_KM': group['Max_Distance_From_Center_KM'],
            'Group_City_Count': group['City_Count'],
            'Is_Center': 'Yes' if city == group['Center_City'] else 'No'
        })

detailed_df = pd.DataFrame(detailed_analysis)
detailed_output_file = str(MAX_DISTANCE_KM) + \
    "_detailed_city_clustering_analysis.csv"
detailed_df.to_csv(detailed_output_file, index=False)

print(f"[SUCCESS] Grouping complete!")
print(f"Main results: '{main_output_file}'")
print(f"Detailed analysis: '{detailed_output_file}'")
print(
    f"Output files contain {len(group_df)} groups and {len(detailed_df)} city records")

# Final cache save
save_distance_cache(distance_cache)

print(f"\nAPI and Cache Statistics:")
print(f"- Total API calls made: {cache_stats['misses']}")
print(f"- Cache hits: {cache_stats['hits']}")
print(f"- Cache efficiency: {(cache_stats['hits'] / (cache_stats['hits'] + cache_stats['misses']) * 100):.1f}%" if (
    cache_stats['hits'] + cache_stats['misses']) > 0 else "N/A")
print(f"- Cache entries saved: {len(distance_cache)}")
print(
    f"- Much more efficient than {len(coords)}² = {len(coords)**2} calls without optimization!")

# Display summary statistics
single_city_groups = len([g for g in final_groups if len(g['Cities']) == 1])
multi_city_groups = len(final_groups) - single_city_groups
max_group_size = max(len(g['Cities']) for g in final_groups)
avg_group_size = sum(len(g['Cities'])
                     for g in final_groups) / len(final_groups)

# Distance statistics for multi-city groups only
multi_city_group_data = [g for g in final_groups if len(g['Cities']) > 1]
if multi_city_group_data:
    avg_distance_overall = sum(g['Avg_Distance_From_Center_KM']
                               for g in multi_city_group_data) / len(multi_city_group_data)
    max_avg_distance = max(g['Avg_Distance_From_Center_KM']
                           for g in multi_city_group_data)
    min_avg_distance = min(g['Avg_Distance_From_Center_KM']
                           for g in multi_city_group_data)
    max_distance_in_any_group = max(
        g['Max_Distance_From_Center_KM'] for g in multi_city_group_data)
else:
    avg_distance_overall = 0
    max_avg_distance = 0
    min_avg_distance = 0
    max_distance_in_any_group = 0

print(f"\nClustering Summary:")
print(f"- Total groups: {len(final_groups)}")
print(f"- Single-city groups: {single_city_groups}")
print(f"- Multi-city groups: {multi_city_groups}")
print(f"- Largest group size: {max_group_size} cities")
print(f"- Average group size: {avg_group_size:.1f} cities")

if multi_city_groups > 0:
    print(f"\nDistance Analysis (Multi-city groups only):")
    print(
        f"- Average distance from center (overall): {avg_distance_overall:.2f} km")
    print(f"- Minimum average distance: {min_avg_distance:.2f} km")
    print(f"- Maximum average distance: {max_avg_distance:.2f} km")
    print(
        f"- Maximum distance in any group: {max_distance_in_any_group:.2f} km")
    print(f"- Clustering threshold used: {MAX_DISTANCE_KM} km")
