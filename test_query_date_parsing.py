#!/usr/bin/env python3
"""
Script to test the SEC Query Agent with specific focus on date parsing and output quality.
Uses test cases from test-queries.txt and evaluates the outputs against expected results.
"""

import os
import sys
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI

from langchain_sec_query_agent import QueryTranslator, get_date_range

# Load environment variables
load_dotenv()

# Get API keys from environment
SEC_API_KEY = os.getenv("SEC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Check for required API keys
if not SEC_API_KEY:
    print("Error: SEC_API_KEY not found in environment variables")
    sys.exit(1)

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found in environment variables")
    sys.exit(1)

# Initialize LLM
llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY
)

# Parse test queries file to extract questions and expected queries
def parse_test_queries(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split content into question-query pairs
    sections = content.split('\n\n')
    
    test_cases = []
    current_question = None
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        if section.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
            # This is a question
            question_parts = section.split('\n', 1)
            if len(question_parts) >= 1:
                # Extract just the question text
                q_text = question_parts[0].split('. ', 1)
                if len(q_text) >= 2:
                    current_question = q_text[1].strip()
                    test_cases.append({'question': current_question, 'expected_query': None})
        
        elif section.startswith('Query:'):
            # This is a query
            if current_question and test_cases:
                # Extract the query JSON
                query_lines = section.split('\n')
                json_start = False
                json_text = ""
                
                for line in query_lines:
                    if line.strip() == '{':
                        json_start = True
                    
                    if json_start:
                        json_text += line.strip() + "\n"
                
                try:
                    # Parse the JSON query
                    if json_text:
                        query_obj = json.loads(json_text)
                        test_cases[-1]['expected_query'] = query_obj
                except json.JSONDecodeError:
                    print(f"Error parsing JSON for question: {current_question}")
    
    return [case for case in test_cases if case['expected_query'] is not None]

def test_date_parsing():
    """Test date parsing functionality with various time formats."""
    print("\n=== Testing Date Parsing ===")
    
    # Current date for reference
    now = datetime.now()
    current_year = now.year
    
    test_cases = [
        {"input": "last month", "expected_start": now.replace(month=now.month-1 if now.month > 1 else 12, year=current_year if now.month > 1 else current_year-1)},
        {"input": "last 3 months", "expected_start": now - timedelta(days=90)},
        {"input": "last year", "expected_start": now.replace(year=current_year-1)},
        {"input": "last quarter", "expected_start": now - timedelta(days=90)},
        {"input": "2023", "expected_start": datetime(2023, 1, 1)},
        {"input": "Q1 2024", "expected_start": datetime(2024, 1, 1)},
        {"input": "since 2020", "expected_start": datetime(2020, 1, 1)},
        {"input": "past 6 months", "expected_start": now - timedelta(days=180)},
    ]
    
    for case in test_cases:
        try:
            date_range = get_date_range(case["input"])
            print(f"\nInput: '{case['input']}'")
            print(f"Date Range: {date_range}")
            
            # Check if the date range contains the expected timeframe
            if "expected_start" in case:
                expected_date = case["expected_start"].strftime("%Y-%m-%d")
                print(f"Expected start around: {expected_date}")
                
                if expected_date in date_range:
                    print("✓ Date range contains expected start date")
                else:
                    print("✗ Date range does not contain expected start date")
        except Exception as e:
            print(f"Error processing '{case['input']}': {e}")

def test_query_translation():
    """Test the Query Translator with questions from test-queries.txt."""
    print("\n=== Testing Query Translation with test-queries.txt ===")
    
    # Initialize translator
    translator = QueryTranslator(llm=llm)
    
    # Get test cases from file
    test_cases = parse_test_queries('test-queries.txt')
    
    # Test date-specific cases
    date_cases = [
        case for case in test_cases 
        if any(term in case['question'].lower() 
              for term in ['month', 'year', 'quarter', 'day', 'week', 'past', 'recent', 'last', 'since', '20', '19'])
    ]
    
    print(f"\nTesting {len(date_cases)} date-related queries...")
    
    for i, case in enumerate(date_cases, 1):
        print(f"\n{i}. Question: {case['question']}")
        expected_query = case['expected_query']['query']
        print(f"Expected: {expected_query}")
        
        try:
            translated_query = translator.translate(case['question'])
            print(f"Actual: {translated_query}")
            
            # Check for date range in the queries
            if "filedAt:" in expected_query and "filedAt:" in translated_query:
                print("✓ Date range detected in both queries")
                
                # Check if dates are properly formatted (YYYY-MM-DD)
                if "TO" in translated_query:
                    date_parts = translated_query.split("filedAt:", 1)[1].split("]", 1)[0].strip()
                    if re.search(r'\d{4}-\d{2}-\d{2}', date_parts):
                        print("✓ Dates properly formatted as YYYY-MM-DD")
                    else:
                        print("✗ Dates not properly formatted")
            elif "filedAt:" in expected_query and "filedAt:" not in translated_query:
                print("✗ Missing date range in translated query")
            elif "filedAt:" not in expected_query and "filedAt:" in translated_query:
                print("? Added date range in translated query (might be appropriate)")
        except Exception as e:
            print(f"Error translating query: {e}")

if __name__ == "__main__":
    print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
    print("Testing SEC Query Agent functionality with focus on date parsing...")
    
    # Test date parsing
    test_date_parsing()
    
    # Test query translation with test-queries.txt
    test_query_translation()
