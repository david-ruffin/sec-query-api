import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_sec_query_agent import QueryTranslator
from sec_api import QueryApi
import json

# Load environment variables
load_dotenv()

# Initialize OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
SEC_API_KEY = os.getenv("SEC_API_KEY")

# Initialize components
llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
query_translator = QueryTranslator(llm=llm)
query_api = QueryApi(api_key=SEC_API_KEY)

# Test questions
test_questions = [
    "Which 10-K filings include Exhibit 21 (Subsidiaries of the Registrant)?",
    "What filings from 2023 include graphical exhibits?",
    "What are the 8-K filings from the past year with Item 2.02 (Results of Operations)?"
]

# Test each question
for question in test_questions:
    print(f"\n=== Testing: {question} ===")
    
    # Translate the question to a query
    query = query_translator.translate(question)
    print(f"Translated query: {query}")
    
    # Execute the query
    search_params = {
        "query": query,
        "from": "0",
        "size": "5"
    }
    
    try:
        response = query_api.get_filings(search_params)
        total = response.get("total", {}).get("value", 0)
        print(f"Total results: {total}")
        
        if response.get("filings"):
            print("\nSample filings:")
            for i, filing in enumerate(response["filings"][:3]):
                print(f"{i+1}. {filing.get('companyName', 'N/A')} - Filed: {filing.get('filedAt', 'N/A').split('T')[0]}")
        else:
            print("No filings found.")
    except Exception as e:
        print(f"Error executing query: {e}")
