#!/usr/bin/env python3
"""
Interactive test script for the SEC Query Agent
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_sec_query_agent import create_sec_query_agent

# Load environment variables
load_dotenv()

# Get API keys from environment variables
SEC_API_KEY = os.getenv("SEC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

if not SEC_API_KEY:
    raise ValueError("SEC_API_KEY environment variable is not set")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

def main():
    # Initialize the language model
    llm = ChatOpenAI(
        model_name=OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.0
    )
    
    # Create the SEC Query Agent
    agent = create_sec_query_agent(llm=llm, sec_api_key=SEC_API_KEY)
    
    print("\n=== SEC Query Agent Interactive Test ===")
    print("Type 'exit' to quit")
    
    while True:
        # Get user question
        question = input("\nEnter your question about SEC filings: ")
        
        if question.lower() == 'exit':
            break
        
        print("\nProcessing your question...")
        
        # Process the question
        try:
            result = agent.process_question(question)
            
            # Display the formatted response
            print("\nResponse:")
            print("-" * 80)
            if "formatted_response" in result and result["formatted_response"]:
                print(result["formatted_response"])
            elif "raw_result" in result and "output" in result["raw_result"]:
                print(result["raw_result"]["output"])
            else:
                print("No results found. Please try a different query.")
                
                # Provide helpful suggestions
                print("\nTry one of these example queries:")
                print("- Show me 10-K filings for Apple from 2020")
                print("- What S-1 filings were submitted in 2021?")
                print("- Find 8-K filings with Item 2.02 from the last year")
                print("- Which 10-K filings include Exhibit 21?")
            print("-" * 80)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print("\nTry simplifying your query or check your API credentials.")
    
    print("\nThank you for using the SEC Query Agent!")

if __name__ == "__main__":
    main()