#!/usr/bin/env python3
"""
Script to test the SEC Query Agent with questions from test-queries.txt
"""

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_sec_query_agent import QueryTranslator, create_sec_query_agent

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

# Parse test queries file
def parse_test_queries(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract questions (numbered items before "Query:")
    questions = []
    for line in content.split('\n'):
        if line.strip() and line[0].isdigit() and '. ' in line:
            questions.append(line.split('. ', 1)[1].strip())
    
    return questions

def test_query_translator_with_file():
    """Test the Query Translator with questions from test-queries.txt."""
    print("\n=== Testing Query Translator with test-queries.txt ===")
    
    # Initialize translator
    translator = QueryTranslator(llm=llm)
    
    # Get questions from file
    questions = parse_test_queries('test-queries.txt')
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        try:
            translated_query = translator.translate(question)
            print(f"Translated: {translated_query}")
        except Exception as e:
            print(f"Error translating query: {e}")

def test_complete_agent_with_file():
    """Test the complete SEC Query Agent with questions from test-queries.txt."""
    print("\n=== Testing Complete SEC Query Agent with test-queries.txt ===")
    
    # Create agent
    agent = create_sec_query_agent(llm=llm, sec_api_key=SEC_API_KEY)
    
    # Get questions from file
    questions = parse_test_queries('test-queries.txt')
    
    # Process first 3 questions to avoid long execution times
    for i, question in enumerate(questions[:3], 1):
        print(f"\nQuestion {i}: {question}")
        try:
            result = agent.process_question(question)
            print(f"Response: {result}")
        except Exception as e:
            print(f"Error processing question: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If a specific question number is provided, only run that one
        try:
            question_num = int(sys.argv[1])
            questions = parse_test_queries('test-queries.txt')
            if 1 <= question_num <= len(questions):
                question = questions[question_num - 1]
                print(f"Testing question {question_num}: {question}")
                
                # Initialize translator
                translator = QueryTranslator(llm=llm)
                translated_query = translator.translate(question)
                print(f"\nTranslated query: {translated_query}")
                
                # Test with complete agent
                agent = create_sec_query_agent(llm=llm, sec_api_key=SEC_API_KEY)
                result = agent.process_question(question)
                print(f"\nFull response: {result}")
            else:
                print(f"Error: Question number must be between 1 and {len(questions)}")
        except ValueError:
            print("Error: Please provide a valid question number")
    else:
        # Run just the query translator test
        test_query_translator_with_file()
