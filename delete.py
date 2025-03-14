import os
from dotenv import load_dotenv
from sec_api import QueryApi
import json

# Load environment variables
load_dotenv()

# Get API key
SEC_API_KEY = os.getenv("SEC_API_KEY")

# Create API client
query_api = QueryApi(api_key=SEC_API_KEY)

# Define a simple query
query = {
    "query": "ticker:AAPL AND formType:\"10-K\"",
    "from": "0",
    "size": "10"
}

# Execute query
print("Executing direct SEC API query...")
response = query_api.get_filings(query)

# Print response
print("\nAPI Response:")
print(json.dumps(response, indent=2))