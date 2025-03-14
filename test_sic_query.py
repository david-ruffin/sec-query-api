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

# Test query
query = {
    "query": 'formType:"10-K" AND entities.sic:7370 AND filedAt:[2023-10-01 TO 2023-12-31]',
    "from": "0",
    "size": "5"
}

print(f"=== Testing Query ===")
print(f"Query: {query['query']}")

try:
    response = query_api.get_filings(query)
    total = response.get("total", {}).get("value", 0)
    print(f"Total results: {total}")
    
    if response.get("filings"):
        print("\nSample filings:")
        for i, filing in enumerate(response["filings"][:5]):
            print(f"{i+1}. {filing.get('companyName', 'N/A')} - Filed: {filing.get('filedAt', 'N/A').split('T')[0]} - {filing.get('formType', 'N/A')}")
            print(f"   SIC: {filing.get('entities', [{}])[0].get('sic', 'N/A') if filing.get('entities') else 'N/A'}")
            print(f"   Description: {filing.get('description', 'N/A')}")
            print(f"   Link: {filing.get('linkToFilingDetails', 'N/A')}")
            print()
    else:
        print("No filings found.")
        
except Exception as e:
    print(f"Error executing query: {e}")
