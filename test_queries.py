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
        "name": "10-K filings with Exhibit 21",
        "query": {
            "query": 'formType:"10-K" AND documentFormatFiles.type:"EX-21"',
            "from": "0",
            "size": "5"
        }
    },
    {
        "name": "Filings with GRAPHIC documents in 2023",
        "query": {
            "query": 'formType:* AND documentFormatFiles.type:GRAPHIC AND filedAt:[2023-01-01 TO 2023-12-31]',
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
        print(f"Total results: {response['total']['value']}")
        
        if response['filings']:
            print("\nSample filings:")
            for i, filing in enumerate(response['filings'][:3]):
                print(f"{i+1}. {filing.get('companyName', 'N/A')} - Filed: {filing.get('filedAt', 'N/A').split('T')[0]}")
                
                # For the first query, show exhibit details
                if "EX-21" in test['query']['query']:
                    print("   Exhibits:")
                    for doc in filing.get('documentFormatFiles', []):
                        if doc.get('type') == 'EX-21':
                            print(f"   - {doc.get('description', 'N/A')} - URL: {doc.get('documentUrl', 'N/A')}")
        else:
            print("No filings found.")
            
    except Exception as e:
        print(f"Error executing query: {e}")
