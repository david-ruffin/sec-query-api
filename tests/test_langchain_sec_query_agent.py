"""
Test script for the Langchain SEC Query Agent.

This script tests each component of the SEC Query Agent:
1. SEC Query Tool
2. Query Translator
3. Response Formatter
4. Complete Agent
"""

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_sec_query_agent import (
    SECQueryTool,
    QueryTranslator,
    ResponseFormatter,
    SECQueryAgent,
    create_sec_query_agent
)

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

def test_sec_query_tool():
    """Test the SEC Query Tool directly with comprehensive parameter support."""
    print("\n=== Testing SEC Query Tool ===")
    
    # Initialize the tool
    query_tool = SECQueryTool(sec_api_key=SEC_API_KEY)
    
    # Test cases for different query patterns
    test_cases = [
        {
            "name": "Basic Company Query",
            "query": 'ticker:AAPL AND formType:"10-K"',
            "from_param": 0,
            "size": 3,
            "sort": [{"filedAt": {"order": "desc"}}]
        },
        {
            "name": "Custom Sort Order",
            "query": 'ticker:MSFT AND formType:"10-Q"',
            "from_param": 0,
            "size": 3,
            "sort": [{"filedAt": {"order": "asc"}}]
        },
        {
            "name": "Industry Query",
            "query": 'sicDescription:"software" AND formType:"10-K"',
            "from_param": 0,
            "size": 3,
            "sort": None  # Test default sort
        },
        {
            "name": "Size Validation",
            "query": 'ticker:TSLA AND formType:"8-K"',
            "from_param": 0,
            "size": 150,  # Should be capped at 100
            "sort": None
        }
    ]
    
    # Run each test case
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print(f"Executing query: {test_case['query']}")
        
        # Execute the query
        result = query_tool._run(
            query=test_case['query'], 
            from_param=test_case['from_param'], 
            size=test_case['size'], 
            sort=test_case['sort']
        )
        
        # Check if we got results
        if "filings" in result and len(result["filings"]) > 0:
            print(f"✅ Success! Found {len(result['filings'])} filings")
            print(f"First filing: {result['filings'][0]['companyName']} - {result['filings'][0]['formType']} - {result['filings'][0]['filedAt']}")
            
            # Verify query_info metadata was added
            if "query_info" in result:
                print(f"✅ Query info metadata present: {result['query_info']}")
            else:
                print("❌ Query info metadata missing")
                
            # Verify sort order if custom sort was specified
            if test_case['sort'] and test_case['sort'][0]['filedAt']['order'] == 'asc' and len(result['filings']) > 1:
                first_date = result['filings'][0]['filedAt']
                last_date = result['filings'][-1]['filedAt']
                if first_date <= last_date:
                    print("✅ Sort order verified (ascending)")
                else:
                    print("❌ Sort order incorrect")
        else:
            print("❌ Error: No filings found or API error")
            print(f"Response: {result}")
            
        # Test error handling by using an invalid API key
        if test_case['name'] == "Basic Company Query":
            print("\nTesting error handling with invalid API key")
            invalid_tool = SECQueryTool(sec_api_key="invalid_key")
            error_result = invalid_tool._run(test_case['query'])
            if "error" in error_result:
                print(f"✅ Error handling works: {error_result['error']}")
            else:
                print("❌ Error handling failed")

def test_query_translator():
    """Test the Query Translator component with all query patterns."""
    print("\n=== Testing Query Translator ===")
    
    # Initialize the translator
    translator = QueryTranslator(llm=llm)
    
    # Test questions covering all query patterns
    test_questions = [
        # Basic patterns
        {
            "pattern": "Basic Company Query",
            "question": "Find Apple's latest 10-K",
            "expected_elements": ["ticker:AAPL", "formType:\"10-K\""]
        },
        {
            "pattern": "Time-Based Query",
            "question": "Show me Tesla's 8-K filings from last year",
            "expected_elements": ["ticker:TSLA", "formType:\"8-K\"", "filedAt"]
        },
        # Advanced patterns
        {
            "pattern": "Industry + Policy Pattern",
            "question": "What did the largest companies in the farming industry report as their revenue recognition policy in their 2023 10-K?",
            "expected_elements": ["sicDescription", "text:\"revenue recognition policy\"", "formType:\"10-K\""]
        },
        {
            "pattern": "Single Company + Section Pattern",
            "question": "What did Microsoft list as the recent accounting guidance on their last 10K?",
            "expected_elements": ["ticker:MSFT", "formType:\"10-K\"", "text:\"recent accounting guidance\""]
        },
        {
            "pattern": "Industry + Financial Item Pattern",
            "question": "Which large software public companies have inventory on their last 10-k?",
            "expected_elements": ["sicDescription:\"software\"", "text:\"inventory\"", "formType:\"10-K\"", "marketCapitalization"]
        },
        {
            "pattern": "Industry + Event Pattern",
            "question": "List all public beverage companies that had acquisition in the last 10-K",
            "expected_elements": ["sicDescription:\"beverage\"", "text:\"acquisition\"", "formType:\"10-K\""]
        },
        {
            "pattern": "Filing Status + Time Pattern",
            "question": "How many large accelerated filers have filed a 10k in the last 30 days?",
            "expected_elements": ["filerCategory", "formType:\"10-K\"", "filedAt:[NOW-30DAYS TO NOW]"]
        },
        {
            "pattern": "Company Comparison Pattern",
            "question": "Compare the risk factors of Apple and Microsoft in their latest 10-K filings",
            "expected_elements": ["ticker:AAPL", "ticker:MSFT", "OR", "text:\"risk factors\"", "formType:\"10-K\""]
        },
        {
            "pattern": "Financial Metric Pattern",
            "question": "Find companies with revenue over 1 billion in their 2023 10-K",
            "expected_elements": ["text:\"revenue\"", "formType:\"10-K\"", "filedAt"]
        }
    ]
    
    # Test each question
    for test_case in test_questions:
        print(f"\nPattern: {test_case['pattern']}")
        print(f"Question: {test_case['question']}")
        
        # Get translated query
        query = translator.translate(test_case['question'])
        print(f"Translated query: {query}")
        
        # Validate query contains expected elements
        missing_elements = []
        for element in test_case['expected_elements']:
            if element not in query:
                missing_elements.append(element)
        
        if not missing_elements:
            print(f"✅ Query contains all expected elements")
        else:
            print(f"⚠️ Query missing elements: {missing_elements}")
            
        # Check for proper formatting
        if "formType:" in query and not "formType:\"" in query:
            print("❌ Form type not properly quoted")
        else:
            print("✅ Form type formatting correct")
            
        # Verify preprocessing adds hints
        processed = translator._preprocess_question(test_case['question'])
        if "Hints:" in processed and processed != test_case['question']:
            print("✅ Preprocessing added hints")
        else:
            print("⚠️ No preprocessing hints added")

def test_response_formatter():
    """Test the Response Formatter component with different query types and patterns."""
    print("\n=== Testing Response Formatter ===")
    
    # Initialize the formatter
    formatter = ResponseFormatter(llm=llm)
    
    # Initialize the query tool to get real data
    query_tool = SECQueryTool(sec_api_key=SEC_API_KEY)
    
    # Test cases for different response types and query patterns
    test_cases = [
        # 1. Company-specific query pattern
        {
            "name": "Company-specific query",
            "query": 'ticker:AAPL AND formType:"10-K"',
            "question": "Find Apple's latest 10-K",
            "size": 3,
            "expected_pattern": "company_specific"
        },
        # 2. Industry query pattern
        {
            "name": "Industry query",
            "query": 'sicDescription:"software" AND formType:"10-K"',
            "question": "What are the latest 10-K filings from software companies?",
            "size": 5,
            "expected_pattern": "industry"
        },
        # 3. Company section pattern (company + text search)
        {
            "name": "Company section query",
            "query": 'ticker:MSFT AND formType:"10-K" AND text:"risk factors"',
            "question": "What are the risk factors in Microsoft's latest 10-K?",
            "size": 2,
            "expected_pattern": "company_section"
        },
        # 4. Industry policy pattern (industry + text search)
        {
            "name": "Industry policy query",
            "query": 'sicDescription:"software" AND formType:"10-K" AND text:"revenue recognition"',
            "question": "How do software companies describe their revenue recognition in 10-K filings?",
            "size": 3,
            "expected_pattern": "industry_policy"
        },
        # 5. Company comparison pattern
        {
            "name": "Company comparison query",
            "query": '(ticker:AAPL OR ticker:MSFT) AND formType:"10-K"',
            "question": "Compare Apple and Microsoft's latest 10-K filings",
            "size": 4,
            "expected_pattern": "company_comparison"
        },
        # 6. Filing status pattern
        {
            "name": "Filing status query",
            "query": 'filerCategory:"Large Accelerated Filer" AND formType:"10-K"',
            "question": "Find 10-K filings from large accelerated filers",
            "size": 3,
            "expected_pattern": "filing_status"
        },
        # 7. Error handling - Authentication error
        {
            "name": "Authentication error",
            "query": 'ticker:AAPL',
            "question": "Find Apple's filings",
            "size": 3,
            "expect_error": True,
            "error_type": "authentication",
            "use_invalid_key": True
        },
        # 8. Error handling - Syntax error
        {
            "name": "Syntax error",
            "query": 'invalid:query',
            "question": "Find invalid query results",
            "size": 3,
            "expect_error": True,
            "error_type": "syntax"
        }
    ]
    
    # Test each case
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print(f"Question: {test_case['question']}")
        print(f"Query: {test_case['query']}")
        
        # Get API response
        if test_case.get('use_invalid_key', False):
            # Use invalid key to test authentication error
            temp_tool = SECQueryTool(sec_api_key="invalid_key")
            api_response = temp_tool._run(test_case['query'], size=test_case['size'])
        else:
            api_response = query_tool._run(test_case['query'], size=test_case['size'])
        
        # Add query_info for pattern detection testing
        if "query_info" not in api_response:
            api_response["query_info"] = {"original_query": test_case['query']}
        
        # Check if we expect an error
        if test_case.get('expect_error', False):
            if "error" in api_response:
                print(f"✅ Expected error received: {api_response['error']}")
            else:
                print("❌ Expected error but got successful response")
        
        # Format the response
        formatted = formatter.format(test_case['question'], api_response)
        print("\nFormatted response:")
        print(formatted[:500] + "..." if len(formatted) > 500 else formatted)  # Truncate long responses
        
        # Verify preprocessing adds metadata
        enhanced = formatter._preprocess_response(api_response)
        if "metadata" in enhanced:
            print("✅ Preprocessing added metadata")
            
            # Check for query pattern detection
            if "expected_pattern" in test_case and not test_case.get('expect_error', False):
                detected_pattern = enhanced["metadata"].get("query_pattern", "unknown")
                if detected_pattern == test_case["expected_pattern"]:
                    print(f"✅ Correctly detected query pattern: {detected_pattern}")
                else:
                    print(f"❌ Incorrect pattern detection. Expected: {test_case['expected_pattern']}, Got: {detected_pattern}")
            
            # Check for error type detection
            if test_case.get('expect_error', False) and "error_type" in test_case:
                if "error_type" in enhanced:
                    if enhanced["error_type"] == test_case["error_type"]:
                        print(f"✅ Correctly detected error type: {enhanced['error_type']}")
                    else:
                        print(f"❌ Incorrect error type. Expected: {test_case['error_type']}, Got: {enhanced['error_type']}")
                else:
                    print("❌ Error type not detected")
        else:
            print("⚠️ No preprocessing metadata added")

def test_complete_agent():
    """Test the complete SEC Query Agent with comprehensive query patterns."""
    print("\n=== Testing Complete SEC Query Agent ===")
    
    # Create the agent
    agent = create_sec_query_agent(llm=llm, sec_api_key=SEC_API_KEY)
    
    # Test questions covering different query patterns
    test_questions = [
        # Basic queries
        "Find Apple's latest 10-K filing",
        "What are the recent 8-K filings for Tesla?",
        "Show me Microsoft's quarterly reports from 2023",
        
        # Advanced pattern queries
        "What did large tech companies report about AI in their latest 10-K filings?",
        "Compare the risk factors between Apple and Microsoft in their most recent annual reports",
        "Which software companies mentioned 'cybersecurity' in their 10-K filings from 2023?",
        "How many large accelerated filers submitted 10-K reports in the last 3 months?"
    ]
    
    # Test each question
    for question in test_questions:
        print(f"\nQuestion: {question}")
        
        # Process the question
        result = agent.process_question(question)
        
        # Check for errors
        if "error" in result and result["error"]:
            print(f"❌ Error: {result['error']}")
        else:
            print("✅ Success!")
            print("\nTranslated query: {}".format(result.get("query", "N/A")))
            print("\nFormatted response:")
            print(result["formatted_response"])
            
            # Verify all components worked together
            if "query" in result and "api_response" in result and "formatted_response" in result:
                print("✅ All agent components functioned properly")
            else:
                print("⚠️ Some agent components may not have functioned properly")
        
        print("\n" + "-"*50)
        
    # Test error handling
    print("\nTesting agent error handling")
    # Create agent with invalid API key
    invalid_agent = create_sec_query_agent(llm=llm, sec_api_key="invalid_key")
    error_result = invalid_agent.process_question("Find Apple's latest 10-K")
    if "error" in error_result and error_result["error"]:
        print(f"✅ Agent error handling works: {error_result['error']}")
    else:
        print("❌ Agent error handling failed")

def test_real_world_scenarios():
    """Test real-world scenarios with actual SEC API data."""
    print("\n=== Testing Real-World Scenarios ===")
    
    # Create the agent
    agent = create_sec_query_agent(llm=llm, sec_api_key=SEC_API_KEY)
    
    # Real-world test scenarios
    scenarios = [
        {
            "name": "Industry + Policy Pattern",
            "question": "What did the largest companies in the tech industry report as their revenue recognition policy in their 2023 10-K?"
        },
        {
            "name": "Single Company + Section Pattern",
            "question": "What did Microsoft list as the recent accounting guidance on their last 10K?"
        },
        {
            "name": "Industry + Financial Item Pattern",
            "question": "Which large software public companies have inventory on their last 10-k?"
        },
        {
            "name": "Industry + Event Pattern",
            "question": "List all public beverage companies that had acquisition in the last 10-K"
        },
        {
            "name": "Filing Status + Time Pattern",
            "question": "How many large accelerated filers have filed a 10k in the last 30 days?"
        }
    ]
    
    # Test each scenario
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Question: {scenario['question']}")
        
        try:
            # Process the question
            result = agent.process_question(scenario['question'])
            
            # Check for errors
            if isinstance(result, dict) and "error" in result and result["error"]:
                print(f"❌ Error: {result['error']}")
            else:
                print("✅ Success!")
                if isinstance(result, dict):
                    print("\nTranslated query: {}".format(result.get("query", "N/A")))
                    print("\nFormatted response excerpt:")
                    # Print just the first 500 characters to keep output manageable
                    if "formatted_response" in result:
                        response = result["formatted_response"]
                        print(response[:500] + "..." if len(response) > 500 else response)
                    else:
                        print("No formatted response available")
                else:
                    print(f"\nUnexpected result type: {type(result)}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("\n" + "-"*50)

if __name__ == "__main__":
    print("Starting SEC Query Agent tests...")
    
    # Run individual component tests
    test_sec_query_tool()
    test_query_translator()
    test_response_formatter()
    
    # Run complete agent test
    test_complete_agent()
    
    # Run real-world scenario tests
    test_real_world_scenarios()
    
    print("\nAll tests completed!")
