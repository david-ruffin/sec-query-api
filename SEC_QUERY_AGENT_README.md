# SEC Query Agent

## Overview

The SEC Query Agent is a specialized Langchain agent designed to interact with the SEC Query API. It translates natural language questions about SEC filings into structured API queries, executes them, and formats the responses in a user-friendly manner. This agent is part of a larger system of specialized agents that collaborate to analyze SEC filings.

**IMPORTANT: DON'T OVER-ENGINEER!** For every implementation decision, ask: "Is this aligned with the end goal?" Use the simplest solution that meets the requirements. We're using widely adopted frameworks (Langchain) and well-documented APIs (SEC API), so we don't need to reinvent the wheel.

## First Principles Approach

The SEC Query Agent implementation follows a first principles approach, avoiding hard-coded rules and leveraging AI where it makes sense. This means:

1. Use AI for natural language understanding and generation rather than rule-based approaches
2. Avoid hard-coding specific query patterns or response formats
3. Design components to be flexible and adaptable to new query types
4. Let the LLM handle the complexity of understanding user intent
5. Use dynamic pattern detection rather than fixed categorization

The implementation is guided by these principles:
- Simplicity over complexity
- Dynamic over static
- AI-driven over rule-based
- Adaptability over rigid structure

This approach ensures the agent can handle a wide range of queries without requiring constant updates to hard-coded rules.

## No Fallbacks Policy

**NEVER EVER implement fallbacks in this codebase.** We have comprehensive documentation for the SEC API and should rely on that instead of implementing fallback mechanisms. This applies to all components of the agent:

1. **SECQueryTool**: Must use the SEC API directly with proper error handling. No fallbacks to alternative data sources or simplified queries.

2. **QueryTranslator**: Must translate queries based on SEC API capabilities. No fallbacks to simpler queries or alternative translation methods.

3. **ResponseFormatter**: Must format responses based on actual API results. No fallbacks to generic responses or templates.

All error handling should be explicit and informative, directing users to check their query parameters or API credentials rather than silently falling back to alternative implementations. This ensures consistency and reliability in our agent's behavior.

## Implementation Details

### Temperature Setting
All language model components must use temperature=0 to ensure consistent, deterministic outputs. This has been implemented in:
- QueryTranslator class
- ResponseFormatter class
- SECQueryAgent class

This ensures that when the same question is asked multiple times, the agent will always produce the same query translation and response formatting.

### Tool Descriptions
The SECQueryTool description is kept concise (under 1024 characters) to comply with OpenAI's API limitations while still providing all essential information for query construction. This ensures compatibility while maintaining full functionality.

## Supported Query Types

The SEC Query Agent is designed to handle the following types of queries:

### 1. Company-Specific Queries
- "Find Apple's latest 10-K filing"
- "What did Microsoft report about cybersecurity risks in their most recent 10-Q?"
- "Show me Tesla's 8-K filings from 2023"
- "When did Amazon last file their annual report?"

### 2. Industry/Sector Queries
- "Which software companies filed 10-K reports in the last quarter?"
- "How many healthcare companies mentioned 'COVID-19' in their recent filings?"
- "List all energy companies that are large accelerated filers"
- "Find tech companies that reported revenue over $1 billion in 2023"

### 3. Policy/Disclosure Queries
- "How do large banks describe their loan loss provisions?"
- "What revenue recognition policies do software companies use?"
- "Find companies that disclosed climate change risks in their 10-K"
- "How do pharmaceutical companies report R&D expenses?"

### 4. Comparative Queries
- "Compare the risk factors between Apple and Microsoft"
- "How do different airlines report on fuel hedging strategies?"
- "Compare executive compensation disclosures between top 5 banks"
- "What differences exist in how tech companies report stock-based compensation?"

### 5. Time-Based Queries
- "Show all 10-K filings submitted in the last 30 days"
- "Which companies filed late Form 4 reports this year?"
- "Find companies that changed auditors between 2022 and 2023"
- "List all IPO filings from the previous quarter"

## Unsupported Query Types

The SEC Query Agent is NOT designed to handle the following types of queries:

### 1. Financial Analysis Beyond Filing Content
- "Which stock will perform better next quarter, Apple or Microsoft?"
- "Predict the future earnings of Tesla based on their filings"
- "What's the best tech stock to invest in right now?"
- "Calculate the intrinsic value of Amazon based on their financials"

### 2. Non-SEC Data Queries
- "What is the current stock price of Apple?"
- "Show me the trading volume for Microsoft yesterday"
- "What's the market capitalization of the S&P 500 companies?"
- "Give me real-time forex rates for USD/EUR"

### 3. Legal Advice
- "Is this company's accounting practice legal?"
- "Did this company violate SEC regulations in their filing?"
- "Should I report this disclosure as potentially fraudulent?"
- "What legal action might the SEC take against this company?"

### 4. Document Retrieval Beyond SEC Filings
- "Find the company's press releases from last month"
- "Show me the transcript of the last earnings call"
- "Get me the company's sustainability report"
- "Find analyst reports about this company"

### 5. Complex Financial Modeling
- "Build a DCF model for this company based on their filings"
- "Calculate the WACC for this company"
- "Project five-year revenue growth based on historical filings"
- "Determine the optimal capital structure for this company"

## Current Implementation Status

### Completed Components

1. **SECQueryTool**
   - ✅ Direct integration with SEC API
   - ✅ Comprehensive parameter support (pagination, sorting)
   - ✅ Error handling for various scenarios
   - ✅ Query metadata addition

2. **QueryTranslator**
   - ✅ Translation of natural language to SEC API queries
   - ✅ Support for all query patterns (company, industry, time-based, etc.)
   - ✅ Preprocessing hints for query optimization

3. **ResponseFormatter**
   - ✅ Context-aware formatting based on query type
   - ✅ Enhanced error message formatting
   - ✅ Metadata enrichment for better context
   - ✅ Query pattern detection

4. **SECQueryAgent**
   - ✅ Integration of all components
   - ✅ Tool-calling agent implementation
   - ✅ Error handling and recovery

### Test Results

The implementation has been tested with various query patterns and scenarios. Key findings:

1. **Query Translation**: Successfully translates all test query patterns into valid SEC API queries.

2. **API Integration**: Properly connects to the SEC API and handles responses, though there's an issue with the sort parameter format that needs to be fixed.

3. **Response Formatting**: Enhanced formatter correctly detects query patterns and provides context-aware formatting.

4. **Error Handling**: Properly identifies and formats different error types (authentication, syntax, etc.).

5. **Agent Integration**: There's an issue with the tool description length exceeding OpenAI's limit (1024 characters), which needs to be fixed.

### Next Steps

1. Fix the tool description length issue in the SECQueryTool class
2. Resolve the sort parameter format issue in the API calls
3. Complete end-to-end testing with all query patterns
4. Optimize performance for large result sets
5. Finalize documentation with usage examples

## Overall System Architecture

### System Components

1. **Supervisor/Orchestrator Agent** (implemented in `sec_analyzer.py`):
   - Acts as the central coordinator for all specialized agents
   - Directly interfaces with the end user
   - Routes queries to appropriate specialized agents
   - Synthesizes responses from multiple agents
   - Maintains conversation context across interactions
   - Presents unified responses to the user

2. **Specialized Agents** (including SEC Query Agent):
   - Each agent focuses on a specific SEC API functionality
   - Operates independently with well-defined interfaces
   - Communicates results back to the supervisor agent

### SEC Query Agent Components

1. **SEC Query Tool**: A Langchain tool that wraps the SEC Query API
   - Handles authentication
   - Translates natural language to query parameters
   - Executes queries against the SEC API
   - Processes and formats responses

2. **Agent Implementation**: Using Langchain's Tool-Calling Agent pattern
   - Determines when to use the SEC Query Tool
   - Handles conversation context
   - Formats responses for the user

3. **Integration with Supervisor**: 
   - Exposes functionality to the supervisor agent (`sec_analyzer.py`)
   - Maintains modularity for independent testing and development

## Implementation Plan

### 1. Define SEC Query Tool

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class SECQuerySchema(BaseModel):
    query: str = Field(..., description="The query string in Lucene syntax, e.g., 'ticker:AAPL AND formType:\"10-K\"'")
    from_param: int = Field(0, description="Starting position for pagination")
    size: int = Field(10, description="Number of results to return")
    sort: Optional[List[Dict]] = Field(None, description="Sorting parameters")

class SECQueryTool(BaseTool):
    name = "sec_query_tool"
    description = "Search for SEC filings using the SEC Query API. Useful for finding filings by company, form type, date range, etc."
    args_schema = SECQuerySchema
    
    def __init__(self, sec_api_key: str):
        super().__init__()
        self.sec_api_key = sec_api_key
        self.base_url = "https://api.sec-api.io"
    
    def _run(self, query: str, from_param: int = 0, size: int = 10, sort: Optional[List[Dict]] = None) -> Dict[str, Any]:
        # Implementation of API call and response processing
        pass
        
    def _arun(self, query: str, from_param: int = 0, size: int = 10, sort: Optional[List[Dict]] = None) -> Dict[str, Any]:
        # Async implementation
        pass
```

### 2. Create Natural Language to Query Translator

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLLM

class QueryTranslator:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            Translate the following natural language question about SEC filings into a Lucene query for the SEC API:
            
            Question: {question}
            
            The query should follow the SEC API Lucene syntax, such as:
            - ticker:AAPL AND formType:"10-K"
            - formType:"10-Q" AND filedAt:[2023-01-01 TO 2023-12-31]
            
            Return only the query string without any explanation.
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def translate(self, question: str) -> str:
        return self.chain.run(question=question).strip()
```

### 3. Implement Response Formatter

```python
class ResponseFormatter:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["question", "api_response"],
            template="""
            You are an expert in SEC filings. Based on the following API response to the question, 
            create a clear, concise summary that answers the question directly.
            
            Question: {question}
            
            API Response: {api_response}
            
            Format your response to be informative and easy to understand. Include the most relevant 
            details from the API response, such as company names, filing dates, form types, and key information.
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def format(self, question: str, api_response: Dict) -> str:
        return self.chain.run(question=question, api_response=str(api_response)).strip()
```

### 4. Create SEC Query Agent

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

class SECQueryAgent:
    def __init__(self, llm: BaseLLM, sec_api_key: str):
        # Initialize components
        self.llm = llm
        self.query_tool = SECQueryTool(sec_api_key=sec_api_key)
        self.query_translator = QueryTranslator(llm=llm)
        self.response_formatter = ResponseFormatter(llm=llm)
        
        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in SEC filings and financial documents. 
            Your job is to help users find information about SEC filings by querying the SEC database.
            Use the sec_query_tool to search for relevant filings based on the user's question."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Create agent
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=[self.query_tool],
            prompt=self.prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=[self.query_tool],
            verbose=True
        )
    
    def process_question(self, question: str) -> Dict[str, Any]:
        # Process the question and return formatted results
        result = self.agent_executor.invoke({"input": question})
        
        # Additional formatting if needed
        formatted_response = self.response_formatter.format(
            question=question,
            api_response=result.get("output", "")
        )
        
        return {
            "raw_result": result,
            "formatted_response": formatted_response
        }
```

## Testing Plan

### 1. Unit Tests

Create unit tests for each component:

```python
# test_sec_query_tool.py
import unittest
from unittest.mock import patch, MagicMock
from sec_query_agent import SECQueryTool

class TestSECQueryTool(unittest.TestCase):
    def setUp(self):
        self.tool = SECQueryTool(sec_api_key="test_key")
    
    @patch('requests.post')
    def test_query_execution(self, mock_post):
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"filings": [{"companyName": "Apple Inc.", "formType": "10-K"}]}
        mock_post.return_value = mock_response
        
        # Execute query
        result = self.tool._run(query="ticker:AAPL AND formType:\"10-K\"")
        
        # Assertions
        self.assertIn("filings", result)
        self.assertEqual(1, len(result["filings"]))
        self.assertEqual("Apple Inc.", result["filings"][0]["companyName"])
```

### 2. Integration Tests

Test the complete agent with mock API responses:

```python
# test_sec_query_agent.py
import unittest
from unittest.mock import patch, MagicMock
from langchain_openai import ChatOpenAI
from sec_query_agent import SECQueryAgent

class TestSECQueryAgent(unittest.TestCase):
    def setUp(self):
        self.llm = ChatOpenAI(model_name="gpt-4o")
        self.agent = SECQueryAgent(llm=self.llm, sec_api_key="test_key")
    
    @patch('sec_query_agent.SECQueryTool._run')
    def test_process_question(self, mock_run):
        # Mock API response
        mock_run.return_value = {"filings": [{"companyName": "Apple Inc.", "formType": "10-K", "filedAt": "2023-11-03"}]}
        
        # Process question
        result = self.agent.process_question("Find Apple's latest 10-K filing")
        
        # Assertions
        self.assertIn("formatted_response", result)
        self.assertIn("Apple", result["formatted_response"])
```

### 3. End-to-End Tests

Test with real API calls (requires API key):

```python
# test_query_agent_e2e.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sec_query_agent import SECQueryAgent

# Load environment variables
load_dotenv()

SEC_API_KEY = os.getenv("SEC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def test_query_agent():
    # Initialize LLM
    llm = ChatOpenAI(model_name="gpt-4o")
    
    # Initialize agent
    agent = SECQueryAgent(llm=llm, sec_api_key=SEC_API_KEY)
    
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
            result = agent.process_question(question)
            print(f"\nFormatted Response:\n{result['formatted_response']}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_query_agent()
```

## Integration with Main System

### 1. Export Agent for Use in `sec_analyzer.py`

```python
# sec_query_agent.py
# ... (implementation as above)

# Export function for use in sec_analyzer.py
def create_sec_query_agent(llm, sec_api_key):
    """
    Create and return a SEC Query Agent instance.
    
    Args:
        llm: A Langchain LLM instance
        sec_api_key: SEC API key for authentication
        
    Returns:
        An initialized SECQueryAgent instance
    """
    return SECQueryAgent(llm=llm, sec_api_key=sec_api_key)
```

### 2. Usage in `sec_analyzer.py`

```python
# sec_analyzer.py
from langchain_openai import ChatOpenAI
from sec_query_agent import create_sec_query_agent

def initialize_agents(llm_model="gpt-4o"):
    """Initialize all specialized SEC agents"""
    llm = ChatOpenAI(model_name=llm_model)
    
    # Initialize SEC Query Agent
    sec_query_agent = create_sec_query_agent(
        llm=llm,
        sec_api_key=os.getenv("SEC_API_KEY")
    )
    
    # Initialize other agents...
    
    return {
        "query_agent": sec_query_agent,
        # Other agents...
    }

def analyze_sec_filing(query, **kwargs):
    """
    Analyze SEC filings based on the provided query.
    
    Args:
        query: Natural language query about SEC filings
        **kwargs: Additional parameters
        
    Returns:
        Analysis results
    """
    # Initialize agents
    agents = initialize_agents()
    
    # Process query with appropriate agent(s)
    # For now, just use the query agent
    result = agents["query_agent"].process_question(query)
    
    return result["formatted_response"]
```

## Detailed Implementation Steps

### Step 1: Set Up Project Structure and Environment
- Create virtual environment
- Install required dependencies
- Set up `.env` file for API keys
- **Testing Checkpoint**: Verify environment is correctly set up with real API access

### Step 2: Implement Comprehensive SEC Query Tool
- Create `SECQueryTool` class that fully implements the SEC API capabilities
- Include all query parameters and options from the SEC API documentation
- Implement specialized methods for common query patterns
- Add robust error handling, rate limiting, and retry logic
- **Testing Checkpoint**: Test with real SEC API calls across different query patterns

### Step 3: Implement Advanced Query Translator
- Create `QueryTranslator` class using Langchain's LLM
- Train on diverse examples covering all query patterns
- Add validation to ensure generated queries follow proper syntax
- Include specialized handling for industry, company, time-based, and event-based queries
- **Testing Checkpoint**: Test with real-world questions across all query patterns

### Step 4: Implement Detailed Response Formatter
- Create `ResponseFormatter` class with context-aware formatting
- Implement specialized formatting for different query types and filing categories
- Add support for extracting and highlighting relevant sections from filings
- Include links to original documents and related filings
- **Testing Checkpoint**: Test with real API responses for various filing types

### Step 5: Create Comprehensive SEC Query Agent
- Implement `SECQueryAgent` class using Langchain's Tool-Calling Agent
- Provide detailed tool descriptions that include all SEC API capabilities
- Add conversation memory with context tracking
- Implement fallback strategies for handling complex or ambiguous queries
- **Testing Checkpoint**: Test with real user questions from all test patterns

### Step 6: Create Flexible Export Function
- Implement factory function with configurable parameters
- Add support for different LLM models and configurations
- Include options for memory persistence and conversation history
- **Testing Checkpoint**: Test agent creation with various configurations

### Step 7: Comprehensive Real Data Testing

#### Real-World Testing Approach
We will use **ONLY real SEC API calls and real data** for all testing - no mocks or simulated responses. This ensures our agent works correctly with actual API behavior and real-world data patterns.

#### Test Categories

1. **Basic Functionality Tests**
   - Test basic company queries with real companies (Apple, Microsoft, Tesla)
   - Test date range queries with different time periods
   - Test form type filtering with various SEC form types (10-K, 10-Q, 8-K)
   - Test pagination and result size limits

2. **Query Pattern Tests**
   - **Industry + Policy Pattern**: "What did the largest companies in the farming industry report as their revenue recognition policy in their 2023 10-K?"
   - **Single Company + Section Pattern**: "What did Microsoft list as the recent accounting guidance on their last 10K?"
   - **Industry + Financial Item Pattern**: "Which large software public companies have inventory on their last 10-k?"
   - **Industry + Event Pattern**: "List all public beverage companies that had acquisition in the last 10-K"
   - **Filing Status + Time Pattern**: "How many large accelerated filers have filed a 10k in the last 30 days?"

3. **Edge Case and Error Handling Tests**
   - Test with ambiguous queries that require clarification
   - Test with non-existent companies or invalid tickers
   - Test with extremely broad queries that return large result sets
   - Test with queries containing typos or common user errors
   - Test API rate limit handling and retry mechanisms

4. **Performance Tests**
   - Measure response times for different query types
   - Test handling of large result sets (100+ filings)
   - Test memory usage during complex operations
   - Test concurrent query handling

5. **Multi-Turn Conversation Tests**
   - Test follow-up questions that reference previous results
   - Test clarification requests and refinements
   - Test conversation memory with context from multiple turns

#### Implementation

```python
# test_real_data.py
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sec_query_agent import SECQueryAgent

# Load environment variables
load_dotenv()

SEC_API_KEY = os.getenv("SEC_API_KEY")

def test_query_patterns():
    """Test all identified query patterns with real SEC API calls"""
    # Initialize LLM and agent
    llm = ChatOpenAI(model_name="gpt-4o")
    agent = SECQueryAgent(llm=llm, sec_api_key=SEC_API_KEY)
    
    # Define test patterns
    test_patterns = [
        # Industry + Policy Pattern
        "What did the largest companies in the farming industry report as their revenue recognition policy in their 2023 10-K?",
        
        # Single Company + Section Pattern
        "What did Microsoft list as the recent accounting guidance on their last 10K?",
        
        # Industry + Financial Item Pattern
        "Which large software public companies have inventory on their last 10-k?",
        
        # Industry + Event Pattern
        "List all public beverage companies that had acquisition in the last 10-K",
        
        # Filing Status + Time Pattern
        "How many large accelerated filers have filed a 10k in the last 30 days?"
    ]
    
    # Run tests
    results = {}
    for i, question in enumerate(test_patterns):
        print(f"\nTesting Pattern {i+1}: {question}")
        start_time = time.time()
        
        try:
            result = agent.process_question(question)
            end_time = time.time()
            
            # Store results
            results[f"pattern_{i+1}"] = {
                "question": question,
                "response": result["formatted_response"],
                "raw_results": result["raw_result"],
                "time_taken": end_time - start_time,
                "success": True
            }
            
            print(f"Success! Time taken: {end_time - start_time:.2f} seconds")
            print(f"Response: {result['formatted_response'][:200]}...")
            
        except Exception as e:
            results[f"pattern_{i+1}"] = {
                "question": question,
                "error": str(e),
                "success": False
            }
            print(f"Error: {str(e)}")
    
    # Analyze results
    successes = sum(1 for r in results.values() if r["success"])
    print(f"\nTest Summary: {successes}/{len(test_patterns)} patterns successful")
    
    return results

if __name__ == "__main__":
    test_query_patterns()
```

#### Success Criteria
- All query patterns successfully return relevant results
- Response times are reasonable (under 30 seconds per query)
- Error handling gracefully manages edge cases
- Results contain accurate and relevant information from SEC filings

**Testing Checkpoint**: Verify all test patterns work with real data before proceeding

### Step 8: Documentation and Optimization
- Add comprehensive docstrings and comments
- Create usage examples for each query pattern
- Optimize performance for common query patterns
- Implement caching for frequently accessed data
- **Testing Checkpoint**: Final review with real-world usage scenarios

## Development Principles

1. **Simplicity First**: Always prefer the simplest solution that meets requirements
2. **Test-Driven Development**: Test each component before moving to the next
3. **Modular Design**: Keep components independent and focused
4. **Documentation**: Maintain clear documentation of design decisions
5. **Alignment Check**: Regularly verify alignment with the overall project goals

**REMEMBER: DON'T OVER-ENGINEER!** For every feature or pattern, ask if it's truly necessary for the end goal.

## Requirements

- Python 3.8+
- Langchain
- OpenAI API key (or other LLM provider)
- SEC API key
- Python-dotenv for environment variable management

## Documentation References

### SEC API Documentation

#### Query API (https://api.sec-api.io)

**Core Query Parameters:**
- `query`: Lucene syntax query string
- `from`: Starting position for pagination (default: 0)
- `size`: Number of results to return (default: 10, max: 100)
- `sort`: Sorting criteria (e.g., `{"filedAt": {"order": "desc"}}`)

**Query Syntax Examples:**
- Basic company search: `ticker:AAPL`
- Form type filter: `formType:"10-K"`
- Date range: `filedAt:[2023-01-01 TO 2023-12-31]`
- Text search: `text:"revenue recognition"`
- Industry search: `sicDescription:"software"`
- Company size: `companyName:"Inc" AND marketCapitalization:[1000000000 TO *]`
- Filing status: `filerCategory:"Large Accelerated Filer"`

**Advanced Query Patterns:**
1. **Industry + Policy Pattern**:
   - `sicDescription:"farming" AND text:"revenue recognition policy" AND formType:"10-K" AND filedAt:[2023-01-01 TO 2024-12-31]`

2. **Single Company + Section Pattern**:
   - `ticker:MSFT AND formType:"10-K" AND text:"recent accounting guidance"`

3. **Industry + Financial Item Pattern**:
   - `sicDescription:"software" AND text:"inventory" AND formType:"10-K" AND marketCapitalization:[10000000000 TO *]`

4. **Industry + Event Pattern**:
   - `sicDescription:"beverage" AND text:"acquisition" AND formType:"10-K"`

5. **Filing Status + Time Pattern**:
   - `filerCategory:"Large Accelerated Filer" AND formType:"10-K" AND filedAt:[NOW-30DAYS TO NOW]`

**Response Structure:**
```json
{
  "total": 123,
  "filings": [
    {
      "id": "...",
      "accessionNo": "0000320193-24-000123",
      "cik": "320193",
      "ticker": "AAPL",
      "companyName": "Apple Inc.",
      "formType": "10-K",
      "filedAt": "2024-11-01T06:01:36-04:00",
      "items": [...],
      "documentFormatFiles": [...],
      "dataFiles": [...],
      "seriesAndClassesContractsData": {...},
      "linkToFilingDetails": "https://www.sec.gov/...",
      "linkToTxt": "https://www.sec.gov/...",
      "linkToHtml": "https://www.sec.gov/...",
      "linkToXbrl": "https://www.sec.gov/...",
      "primaryDocument": "...",
      "primaryDocDescription": "..."
    },
    ...
  ]
}
```

- **Full Documentation**: https://sec-api.io/docs
  - Authentication methods
  - Complete parameter reference
  - Advanced query techniques
  - Rate limiting information

### Langchain Documentation

- **Agents**: https://python.langchain.com/docs/modules/agents/
  - Agent types and selection guidelines
  - Tool integration patterns
  - Memory and state management

- **Tool-Calling Agent**: https://python.langchain.com/docs/modules/agents/agent_types/tool_calling/
  - Implementation details
  - Prompt templates
  - Agent executors

- **Multi-Agent Collaboration**: https://python.langchain.com/docs/use_cases/multi_agent_collaboration/
  - Supervisor/orchestrator patterns
  - Agent communication protocols
  - Workflow management

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install langchain langchain_openai pydantic python-dotenv requests

# Create .env file
cat > .env << EOL
SEC_API_KEY=your_sec_api_key
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
EOL
```

## Usage Example

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sec_query_agent import SECQueryAgent

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o")

# Initialize agent
agent = SECQueryAgent(llm=llm, sec_api_key=os.getenv("SEC_API_KEY"))

# Process a question
result = agent.process_question("Find Apple's latest 10-K filing")
print(result["formatted_response"])
```
