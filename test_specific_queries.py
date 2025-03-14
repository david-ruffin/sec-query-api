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

# Test queries
queries = [
    {
        "name": "Apple 10-K filings from 2000-2020",
        "query": {
            "query": 'ticker:AAPL AND formType:"10-K" AND filedAt:[2000-01-01 TO 2020-12-31]',
            "from": "0",
            "size": "5"
        }
    },
    {
        "name": "S-1 filings from 2021",
        "query": {
            "query": 'formType:"S-1" AND filedAt:[2021-01-01 TO 2021-12-31]',
            "from": "0",
            "size": "5"
        }
    }
]

# Execute queries and print results
for test in queries:
    print(f"\n=== Testing: {test['name']} ===")
    print(f"Query: {test['query']['query']}")
    
    try:
        response = query_api.get_filings(test['query'])
        total = response.get("total", {}).get("value", 0)
        print(f"Total results: {total}")
        
        if response.get("filings"):
            print("\nSample filings:")
            for i, filing in enumerate(response["filings"][:3]):
                print(f"{i+1}. {filing.get('companyName', 'N/A')} - Filed: {filing.get('filedAt', 'N/A').split('T')[0]} - {filing.get('formType', 'N/A')}")
                print(f"   Description: {filing.get('description', 'N/A')}")
                print(f"   Link: {filing.get('linkToFilingDetails', 'N/A')}")
                print()
        else:
            print("No filings found.")
            
    except Exception as e:
        print(f"Error executing query: {e}")
