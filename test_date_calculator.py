"""
Test the DateCalculator implementation for the SEC Query Agent.
"""

import os
from dotenv import load_dotenv
from langchain_sec_query_agent import DateCalculator, QueryTranslator
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Get API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def test_date_calculator():
    """Test the DateCalculator class directly."""
    print("=== Testing DateCalculator ===")
    
    test_cases = [
        "last month",
        "last 3 months",
        "last 6 months",
        "last year",
        "last quarter",
        "Q1 2024",
        "Q4 2024"
    ]
    
    for time_period in test_cases:
        date_range = DateCalculator.get_date_range(time_period)
        print(f"{time_period} → {date_range}")
    
    print()

def test_query_translator():
    """Test the QueryTranslator with date references."""
    print("=== Testing QueryTranslator with Date References ===")
    
    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=OPENAI_API_KEY
    )
    
    # Initialize the query translator
    query_translator = QueryTranslator(llm=llm)
    
    # Test cases
    test_questions = [
        "Show me Apple's 10-K filings from last year",
        "Find Tesla's 8-K filings from the last 3 months",
        "What filings did Microsoft submit in the last quarter?",
        "Get me the 10-Q filings for tech companies from Q1 2024"
    ]
    
    for question in test_questions:
        print(f"Question: {question}")
        query = query_translator.translate(question)
        print(f"Translated Query: {query}")
        print()

if __name__ == "__main__":
    test_date_calculator()
    test_query_translator()
