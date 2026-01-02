"""Simple script to test the API."""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_query(query: str):
    """Test query endpoint."""
    print(f"Testing /query endpoint with: '{query}'...")
    payload = {
        "query": query,
        "k": 3,
        "num_variations": 2
    }
    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Answer: {data['answer']}")
    else:
        print(f"Error: {response.text}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("API Test Script")
    print("=" * 60)
    print()
    print("Make sure the API server is running:")
    print("  python run_api.py")
    print()
    print("=" * 60)
    print()
    
    try:
        # Test queries
        test_query("What is PMAY?")
        test_query("What are the eligibility criteria for PMAY?")
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API server.")
        print("Make sure the server is running: python run_api.py")
    except Exception as e:
        print(f"ERROR: {e}")

