# test_query_agent.py
import os
from dotenv import load_dotenv
from sec_query_api import SecQueryTool

# Load environment variables
load_dotenv()

SEC_API_KEY = os.getenv("SEC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

def test_query_agent():
    # Initialize the tool
    sec_tool = SecQueryTool(SEC_API_KEY, OPENAI_API_KEY, OPENAI_MODEL)
    
    # Test questions
    test_questions = [
        "Find Apple's latest 10-K filing",
        "Get Microsoft's quarterly reports from 2023",
        "Show me Tesla's 8-K filings from the last 6 months"
    ]
    
    # Test each question
    for i, question in enumerate(test_questions):
        print(f"\n{'='*80}")
        print(f"Test {i+1}: {question}")
        print(f"{'='*80}")
        
        try:
            result = sec_tool.process_user_question(question)
            print(f"\nFormatted Response:\n{result['formatted_response']}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_query_agent()