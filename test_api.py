import requests
import json

# Define the API endpoint URL (assuming the server is running locally on port 8000)
api_url = "http://127.0.0.1:8000/ner"

# Sample text to send for NER
sample_text = "Apple Inc. is a technology company based in Cupertino, California. Tim Cook is the current CEO."

# Create the payload for the POST request
payload = {"text": sample_text}

# Send the POST request
try:
    response = requests.post(api_url, json=payload)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        result = response.json()
        print("API Response:")
        print(json.dumps(result, indent=4)) # Print formatted JSON output
    else:
        print(f"Error: API returned status code {response.status_code}")
        print(f"Response body: {response.text}")

except requests.exceptions.ConnectionError as e:
    print(f"Error: Could not connect to the API. Please ensure the FastAPI server is running.")
    print(f"Details: {e}")
except Exception as e:
    print(f"An error occurred: {e}")