"""
Langchain SEC Query Agent

This module implements a Langchain-based agent for querying the SEC API.
It translates natural language questions into SEC API queries, executes them,
and formats the responses in a user-friendly manner.
"""

import os
import json
import re
import datetime
from typing import Dict, List, Optional, Any, Type
from pydantic import BaseModel, Field, field_validator
from dateutil.relativedelta import relativedelta

# dateparser is required for handling relative date references
try:
    import dateparser
except ImportError:
    print("Warning: dateparser is not installed. Please install it using: pip install dateparser")

from langchain.tools import BaseTool
from langchain.chains import LLMChain
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.agents import AgentAction
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel

from sec_api import QueryApi


# We'll use dateparser instead of a custom DateCalculator class
# dateparser is the standard library for handling relative date references in natural language
# It supports phrases like 'last year', 'last month', 'last 3 months', etc.

def get_date_range(time_period: str) -> str:
    """Convert a relative time period to an explicit date range for SEC API.
    
    Uses direct pattern matching and date calculations to handle various date formats.
    
    Args:
        time_period: A string like 'last month', 'last 3 months', 'Q1 2024', etc.
        
    Returns:
        A string in the format 'filedAt:[YYYY-MM-DD TO YYYY-MM-DD]'
    """
    # Print debug info about the time period we're parsing
    print(f"\nDEBUG: Parsing time period: '{time_period}'")
        
    # Get current date as the end date
    end_date = datetime.datetime.now().date()
    print(f"DEBUG: Current date (end_date): {end_date.isoformat()}")
    
    # Handle quarter notation (Q1 2024, etc.)
    quarter_match = re.search(r'q([1-4])\s+(\d{4})', time_period.lower())
    if quarter_match:
        quarter = int(quarter_match.group(1))
        year = int(quarter_match.group(2))
        
        # Map quarter to month ranges
        quarter_start_months = {1: 1, 2: 4, 3: 7, 4: 10}
        quarter_end_months = {1: 3, 2: 6, 3: 9, 4: 12}
        
        start_month = quarter_start_months[quarter]
        end_month = quarter_end_months[quarter]
        
        start_date = datetime.date(year, start_month, 1)
        # Last day of the end month
        if end_month == 12:
            end_date = datetime.date(year, 12, 31)
        else:
            end_date = datetime.date(year, end_month + 1, 1) - datetime.timedelta(days=1)
            
        date_range = f"filedAt:[{start_date.isoformat()} TO {end_date.isoformat()}]"
        print(f"DEBUG: Quarter match - date range: {date_range}")
        return date_range
    
    # Special handling for common phrases
    if time_period.lower() == 'last year':
        # Last year means the previous calendar year from today
        today = datetime.datetime.now().date()
        last_year = today.year - 1
        start_date = datetime.date(last_year, 1, 1)  # January 1st of last year
        end_date = datetime.date(last_year, 12, 31)  # December 31st of last year
        date_range = f"filedAt:[{start_date.isoformat()} TO {end_date.isoformat()}]"
        print(f"DEBUG: Special 'last year' handling - date range: {date_range}")
        return date_range
    
    # Handle 'last X years/months/weeks/days' patterns directly
    number_match = re.search(r'last\s+(\d+)\s+(year|month|week|day)s?', time_period.lower())
    if number_match:
        number = int(number_match.group(1))
        unit = number_match.group(2)
        
        # Calculate start date based on unit
        if unit == 'year':
            start_date = end_date - relativedelta(years=number)
        elif unit == 'month':
            start_date = end_date - relativedelta(months=number)
        elif unit == 'week':
            start_date = end_date - relativedelta(weeks=number)
        elif unit == 'day':
            start_date = end_date - relativedelta(days=number)
        
        date_range = f"filedAt:[{start_date.isoformat()} TO {end_date.isoformat()}]"
        print(f"DEBUG: Pattern match for 'last X {unit}s' - date range: {date_range}")
        return date_range
    
    # Handle 'last month', 'last week', etc.
    if time_period.lower() == 'last month':
        start_date = end_date - relativedelta(months=1)
        date_range = f"filedAt:[{start_date.isoformat()} TO {end_date.isoformat()}]"
        print(f"DEBUG: Special 'last month' handling - date range: {date_range}")
        return date_range
    
    if time_period.lower() == 'last quarter':
        start_date = end_date - relativedelta(months=3)
        date_range = f"filedAt:[{start_date.isoformat()} TO {end_date.isoformat()}]"
        print(f"DEBUG: Special 'last quarter' handling - date range: {date_range}")
        return date_range
    
    if time_period.lower() == 'last week':
        start_date = end_date - relativedelta(weeks=1)
        date_range = f"filedAt:[{start_date.isoformat()} TO {end_date.isoformat()}]"
        print(f"DEBUG: Special 'last week' handling - date range: {date_range}")
        return date_range
    
    # For any other time period, raise an exception - no fallbacks
    raise ValueError(f"Unable to parse time period: '{time_period}'. Please use a specific format like 'last year', 'last 2 months', 'Q1 2024', etc.")

class SECQuerySchema(BaseModel):
    """Schema for SEC Query API parameters with comprehensive support for all SEC API capabilities."""
    query: str = Field(..., description="The query string in Lucene syntax, e.g., 'ticker:AAPL AND formType:\"10-K\"'")
    from_param: int = Field(0, description="Starting position for pagination")
    size: int = Field(10, description="Number of results to return (max 100)")
    sort: Optional[List[Dict[str, Dict[str, str]]]] = Field(
        None, 
        description="Sorting parameters, e.g., [{\"filedAt\": {\"order\": \"desc\"}}]"
    )
    
    @field_validator('size')
    def validate_size(cls, v):
        """Validate that size is within allowed range."""
        if v > 100:
            raise ValueError("Size cannot exceed 100 (SEC API limit)")
        if v < 1:
            raise ValueError("Size must be at least 1")
        return v

class SECQueryTool(BaseTool):
    """Tool for querying the SEC API using Lucene syntax with comprehensive support for all query patterns."""
    
    name: str = "sec_query_tool"
    description: str = """Search for SEC filings using the SEC API. Useful for finding filings by company, form type, date range, and text content.
Examples:
- Company: ticker:AAPL
- Form: formType:"10-K"
- Date: filedAt:[2023-01-01 TO 2023-12-31]
- Text: text:"revenue recognition"
- Industry: sicDescription:"software"
- Combinations: ticker:MSFT AND formType:"10-K" AND text:"risk factors"
- Comparisons: (ticker:AAPL OR ticker:MSFT) AND formType:"10-K"
Supports parameters: from (starting index), size (result count), sort (ordering)."""
    
    args_schema: Type[BaseModel] = SECQuerySchema
    
    # Define Pydantic fields
    sec_api_key: str = Field(..., description="SEC API key for authentication")
    
    def __init__(self, sec_api_key: str):
        """Initialize the SEC Query Tool with an API key."""
        super().__init__(sec_api_key=sec_api_key)
        # Create the API client outside of Pydantic's control
        self._sec_api = QueryApi(api_key=sec_api_key)
    
    def _run(self, query: str, from_param: int = 0, size: int = 10, sort: Optional[List[Dict[str, Dict[str, str]]]] = None) -> Dict[str, Any]:
        """Execute a search against the SEC API with comprehensive parameter support."""
        # Only apply default sort if none provided and we need to sort
        # This simplifies the query construction
            
        # Build search parameters
        search_params = {
            "query": query,
            "from": str(from_param),
            "size": str(size)
        }
        
        # Add sort parameter according to SEC API documentation
        # The API expects sort as a JSON object, not a string
        if sort is not None and len(sort) > 0:
            search_params["sort"] = sort
        
        try:
            # Execute the query
            response = self._sec_api.get_filings(search_params)
            
            # Add metadata to help with response formatting
            if "filings" in response:
                # Get total count - expect a specific structure
                total = response.get("total")
                if total is None:
                    raise ValueError("Missing 'total' field in API response")
                    
                if not isinstance(total, dict) or "value" not in total:
                    raise ValueError(f"Unexpected 'total' structure in API response: {total}")
                    
                total_count = total["value"]
                
                response["query_info"] = {
                    "original_query": query,
                    "result_count": len(response["filings"]),
                    "total_count": total_count,
                    "has_more": total_count > (from_param + len(response["filings"]))
                }
            
            return response
        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error messages
            if "401" in error_msg:
                error_msg = "Authentication error: Invalid SEC API key"
            elif "429" in error_msg:
                error_msg = "Rate limit exceeded: Too many requests to SEC API"
                
            return {
                "error": error_msg,
                "query": query,
                "search_params": search_params
            }
    
    async def _arun(self, query: str, from_param: int = 0, size: int = 10, sort: Optional[List[Dict[str, Dict[str, str]]]] = None) -> Dict[str, Any]:
        """Async implementation of the SEC API search."""
        # For now, just call the synchronous version
        return self._run(query, from_param, size, sort)

class QueryTranslator:
    """Translates natural language questions into SEC API queries with support for all query patterns."""
    
    def __init__(self, llm: BaseLanguageModel):
        """Initialize with a language model."""
        self.llm = llm
        # Set temperature to 0 for consistent outputs
        if hasattr(self.llm, 'temperature'):
            self.llm.temperature = 0
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert in SEC filings and the SEC API. Translate the user's question into a proper SEC API query using Lucene syntax.
            
            Basic Examples:
            - "Find Apple's latest 10-K" → ticker:AAPL AND formType:"10-K"
            - "Show me Tesla's 8-K filings from last year" → ticker:TSLA AND formType:"8-K"
            - "Get Microsoft's quarterly reports from 2023" → ticker:MSFT AND formType:"10-Q" AND filedAt:[2023-01-01 TO 2023-12-31]
            - "What filings has the company with CIK 1318605 made in 2023?" → cik:1318605 AND filedAt:[2023-01-01 TO 2023-12-31]
            - "Find 10-K filings from computer programming companies in Q4 2023" → formType:"10-K" AND entities.sic:7370 AND filedAt:[2023-10-01 TO 2023-12-31]
            
            IMPORTANT: For relative date references like "last year", "last month", "last quarter", etc., DO NOT include any date range in your query. Our system will automatically handle these relative date references. Only include explicit date ranges for specific years or quarters like "2023" or "Q1 2024".
            
            Advanced Query Pattern Examples:
            1. Industry + Policy Pattern:
               - "What did the largest companies in the farming industry report as their revenue recognition policy in their 2023 10-K?" 
               → sicDescription:"farming" AND text:"revenue recognition policy" AND formType:"10-K" AND filedAt:[2023-01-01 TO 2023-12-31]
            
            2. Single Company + Section Pattern:
               - "What did Microsoft list as the recent accounting guidance on their last 10K?" 
               → ticker:MSFT AND formType:"10-K" AND text:"recent accounting guidance" AND filedAt:DATE_RANGE_PLACEHOLDER
            
            3. Industry + Financial Item Pattern:
               - "Which large software public companies have inventory on their last 10-k?" 
               → sicDescription:"software" AND text:"inventory" AND formType:"10-K" AND marketCapitalization:[1000000000 TO *] AND filedAt:DATE_RANGE_PLACEHOLDER
            
            4. Industry + Event Pattern:
               - "List all public beverage companies that had acquisition in the last 10-K" 
               → sicDescription:"beverage" AND text:"acquisition" AND formType:"10-K" AND filedAt:DATE_RANGE_PLACEHOLDER
            
            5. Filing Status + Time Pattern:
               - "How many large accelerated filers have filed a 10k in the last year?" 
               → filerCategory:"Large Accelerated Filer" AND formType:"10-K" AND filedAt:DATE_RANGE_PLACEHOLDER
            
            6. Company Comparison Pattern:
               - "Compare the risk factors of Apple and Microsoft in their latest 10-K filings"
               → (ticker:AAPL OR ticker:MSFT) AND formType:"10-K" AND text:"risk factors"
            
            7. Financial Metric Pattern:
               - "Find companies with revenue over 1 billion in their 2023 10-K"
               → text:"revenue" AND formType:"10-K" AND filedAt:[2023-01-01 TO 2023-12-31]
                
            8. Exhibit Search Pattern:
               - "Which 10-K filings include Exhibit 21 (Subsidiaries of the Registrant)?"
               → formType:"10-K" AND documentFormatFiles.type:"EX-21"
               
            9. Document Type Search Pattern:
               - "Find 10-K filings with Management's Discussion and Analysis section"
               → formType:"10-K" AND documentFormatFiles.description:"Management's Discussion"
            
            Important Rules:
            - ALWAYS use double quotes around form types: formType:"10-K"
            - For date ranges, use the format: filedAt:[2023-01-01 TO 2023-12-31]
            - For recent time periods, use NOW: filedAt:[NOW-30DAYS TO NOW]
            - For open-ended ranges, use asterisk: marketCapitalization:[1000000000 TO *]
            - For industry searches, use sicDescription:"industry name"
            - For text searches within filings, use text:"search terms"
            - For company comparisons, use parentheses and OR: (ticker:AAPL OR ticker:MSFT)
            
            Return ONLY the query string, nothing else. No explanations.
            """),
            ("human", "{question}")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        
    def _preprocess_question(self, question: str) -> str:
        """Preprocess the question to identify key entities and query patterns."""
        # Convert to lowercase for easier pattern matching
        question_lower = question.lower()
        
        # Identify common patterns and add hints
        hints = []
        
        # Check for industry pattern
        industry_keywords = ["industry", "sector", "companies in", "firms in"]
        if any(keyword in question_lower for keyword in industry_keywords):
            hints.append("This appears to be an industry-related query. Consider using sicDescription.")
            
        # Check for time-based pattern
        time_keywords = ["last year", "recent", "latest", "in 2023", "in 2022", "last quarter", "last month", "last 30 days"]
        if any(keyword in question_lower for keyword in time_keywords):
            hints.append("This appears to be a time-based query. Consider using filedAt with appropriate date ranges.")
            
        # Check for company size pattern
        size_keywords = ["large", "small", "biggest", "largest", "market cap", "capitalization"]
        if any(keyword in question_lower for keyword in size_keywords):
            hints.append("This appears to reference company size. Consider using marketCapitalization ranges.")
            
        # Check for comparison pattern
        comparison_keywords = ["compare", "comparison", "versus", "vs", "and", "both", "difference"]
        company_names = ["apple", "microsoft", "google", "amazon", "tesla", "facebook", "meta", "netflix"]
        if any(keyword in question_lower for keyword in comparison_keywords) and sum(name in question_lower for name in company_names) > 1:
            hints.append("This appears to be a comparison query. Consider using (ticker:X OR ticker:Y) syntax.")
            
        # Check for financial metric pattern
        financial_keywords = ["revenue", "profit", "income", "earnings", "ebitda", "assets", "liabilities", "debt", "cash flow"]
        if any(keyword in question_lower for keyword in financial_keywords):
            hints.append("This appears to reference financial metrics. Consider using text: to search for specific metrics.")
        
        # If we have hints, add them to the question
        if hints:
            return f"{question}\n\nHints: {' '.join(hints)}"
        
        return question
        
    def translate(self, question: str) -> str:
        """Translate a natural language question to a SEC API query with pattern recognition."""
        # Preprocess the question to identify patterns
        processed_question = self._preprocess_question(question)
        
        # Get the query from the LLM
        query = self.chain.run(question=processed_question).strip()
        print(f"\nDEBUG: Original LLM query: {query}")
        
        # Look for time period references that might need calculation
        time_period_patterns = [
            r'last\s+month',
            r'last\s+([0-9]+)\s+months?',
            r'last\s+([0-9]+)\s+years?',
            r'last\s+year',
            r'last\s+quarter',
            r'Q[1-4]\s+\d{4}'
        ]
        
        # Check if any time period patterns are in the original question
        for pattern in time_period_patterns:
            matches = re.findall(pattern, question.lower())
            if matches or re.search(pattern, question.lower()):
                # Extract the time period from the question
                time_period_match = re.search(pattern, question.lower())
                if time_period_match:
                    time_period = time_period_match.group(0)
                    print(f"DEBUG: Found time period in question: '{time_period}'")
                    
                    # Calculate the date range using the new function
                    date_range = get_date_range(time_period)
                    print(f"DEBUG: Calculated date range: {date_range}")
                    
                    # If the query already has a filedAt clause or placeholder, replace it
                    if 'filedAt:' in query:
                        old_date_range = re.search(r'filedAt:\[[^\]]+\]', query)
                        if old_date_range:
                            print(f"DEBUG: Replacing existing date range: {old_date_range.group(0)}")
                            query = re.sub(r'filedAt:\[[^\]]+\]', date_range, query)
                        elif 'filedAt:DATE_RANGE_PLACEHOLDER' in query:
                            print(f"DEBUG: Replacing DATE_RANGE_PLACEHOLDER with: {date_range}")
                            query = query.replace('filedAt:DATE_RANGE_PLACEHOLDER', date_range)
                    # Otherwise, add the date range to the query if it makes sense
                    elif any(term in question.lower() for term in ['recent', 'latest', 'last', 'this year', 'this quarter']):
                        print(f"DEBUG: Adding date range to query")
                        query += f" AND {date_range}"
        
        # Post-process the query to ensure proper formatting
        # Ensure form types are in quotes
        if "formType:" in query and not "formType:\"" in query:
            query = query.replace("formType:", "formType:\"")
            if " AND " in query:
                query = query.replace(" AND ", "\" AND ")
            else:
                query = query + "\""
        
        # Ensure SIC descriptions are in quotes
        if "sicDescription:" in query and not "sicDescription:\"" in query:
            parts = query.split("sicDescription:")
            for i in range(1, len(parts)):
                if " AND " in parts[i]:
                    end_idx = parts[i].find(" AND ")
                    parts[i] = f"\"{parts[i][:end_idx]}\"" + parts[i][end_idx:]
                else:
                    parts[i] = f"\"{parts[i]}\""
            query = "sicDescription:".join(parts)
        
        # Ensure filerCategory is in quotes
        if "filerCategory:" in query and not "filerCategory:\"" in query:
            parts = query.split("filerCategory:")
            for i in range(1, len(parts)):
                if " AND " in parts[i]:
                    end_idx = parts[i].find(" AND ")
                    parts[i] = f"\"{parts[i][:end_idx]}\"" + parts[i][end_idx:]
                else:
                    parts[i] = f"\"{parts[i]}\""
            query = "filerCategory:".join(parts)
                
        return query

class ResponseFormatter:
    """Formats SEC API responses into user-friendly text with context-aware formatting for all query patterns."""
    
    def __init__(self, llm: BaseLanguageModel):
        """Initialize with a language model."""
        self.llm = llm
        # Set temperature to 0 for consistent outputs
        if hasattr(self.llm, 'temperature'):
            self.llm.temperature = 0
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert in SEC filings. Summarize the results from an SEC API query in a clear, user-friendly way.
            
            Adapt your response format based on the query type and results:
            
            1. For company-specific queries:
               - Highlight the company name, ticker, and CIK
               - List filing types, dates, and descriptions
               - Include direct links to the filings when available
               - For 10-K and 10-Q filings, mention key financial highlights if present
            
            2. For industry or sector queries:
               - Group results by industry/sector
               - Highlight trends or patterns across companies
               - Compare companies within the same industry
               - Summarize the most relevant findings
            
            3. For policy or text search queries:
               - Extract and highlight the relevant text sections
               - Provide context around the matches
               - Compare approaches across different companies if applicable
               - Identify common themes or unique approaches
            
            4. For time-based queries:
               - Organize results chronologically
               - Highlight changes or trends over time
               - Summarize filing frequency or patterns
               - Note any unusual timing patterns
            
            5. For filing status queries:
               - Provide counts and statistics
               - Highlight notable filers
               - Summarize trends in filing behavior
               - Compare with historical patterns if relevant
            
            6. For company comparison queries:
               - Directly compare the companies mentioned
               - Highlight similarities and differences
               - Organize by topic for easy comparison
               - Provide a balanced assessment
            
            7. For financial metric queries:
               - Focus on the specific metrics mentioned
               - Provide context for the numbers
               - Compare across companies when possible
               - Note any outliers or trends
            
            Always include:
            - Total number of results found
            - Date range of the filings
            - Any limitations or suggestions for refining the search
            - Relevant SEC filing links when available
            
            If the API returned an error, explain it clearly and suggest fixes based on the error type:
            - For authentication errors, suggest checking the API key
            - For rate limit errors, suggest waiting and trying again
            - For syntax errors, suggest corrections to the query format
            - For empty results, suggest broadening the search parameters
            
            Keep your response concise but informative. Focus on answering the original question directly.
            """),
            ("human", "Question: {question}\n\nSEC API Results: {api_response}")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _preprocess_response(self, api_response: Dict) -> Dict:
        """Preprocess the API response to extract key information and add context."""
        # If there's an error, just return the original response with enhanced error context
        if "error" in api_response:
            enhanced_error = api_response.copy()
            error_msg = api_response["error"]
            
            # Add more context based on error type
            if "Authentication" in error_msg or "401" in error_msg:
                enhanced_error["error_type"] = "authentication"
                enhanced_error["error_suggestion"] = "Check your SEC API key and ensure it is valid."
            elif "Rate limit" in error_msg or "429" in error_msg:
                enhanced_error["error_type"] = "rate_limit"
                enhanced_error["error_suggestion"] = "You've exceeded the API rate limits. Wait a moment and try again."
            elif "syntax" in error_msg.lower() or "invalid" in error_msg.lower():
                enhanced_error["error_type"] = "syntax"
                enhanced_error["error_suggestion"] = "Check your query syntax. Ensure field names and operators are correct."
            else:
                enhanced_error["error_type"] = "general"
                enhanced_error["error_suggestion"] = "An unexpected error occurred. Try simplifying your query."
                
            return enhanced_error
            
        # Extract filing statistics
        # According to SEC API docs, total is an object with value and relation fields
        # Get total results - expect a specific structure without fallbacks
        total_obj = api_response.get("total", {"value": 0, "relation": "eq"})
        if not isinstance(total_obj, dict):
            # If the structure is unexpected, raise an exception
            raise ValueError(f"Unexpected 'total' structure in API response: {total_obj}")
            
        total_results = total_obj.get("value", 0)
            
        filings = api_response.get("filings", [])
        num_filings = len(filings)
        
        # Extract date range if filings exist
        filing_dates = []
        if filings:
            for filing in filings:
                if "filedAt" in filing:
                    # Extract just the date part (not time) - expect ISO format
                    if "T" not in filing["filedAt"]:
                        raise ValueError(f"Expected ISO format date with 'T' separator, got: {filing['filedAt']}")
                    date_str = filing["filedAt"].split("T")[0]
                    filing_dates.append(date_str)
        
        # Sort dates and get range
        date_range = {}
        if filing_dates:
            filing_dates.sort()
            date_range = {
                "earliest": filing_dates[0],
                "latest": filing_dates[-1],
                "span_days": (datetime.datetime.fromisoformat(filing_dates[-1]) - 
                              datetime.datetime.fromisoformat(filing_dates[0])).days
            }
        
        # Extract company information for grouping
        companies = {}
        form_types = {}
        industries = {}
        
        for filing in filings:
            # Track companies
            cik = filing.get("cik")
            if cik:
                if cik not in companies:
                    companies[cik] = {
                        "name": filing.get("companyName", "Unknown"),
                        "ticker": filing.get("ticker", "Unknown"),
                        "filings": []
                    }
                companies[cik]["filings"].append(filing)
            
            # Track form types
            form_type = filing.get("formType")
            if form_type:
                if form_type not in form_types:
                    form_types[form_type] = 0
                form_types[form_type] += 1
            
            # Track industries
            industry = filing.get("sicDescription")
            if industry:
                if industry not in industries:
                    industries[industry] = 0
                industries[industry] += 1
        
        # Add metadata to the response
        enhanced_response = api_response.copy()
        enhanced_response["metadata"] = {
            "total_results": total_results,
            "returned_results": num_filings,
            "has_more": total_results > num_filings,
            "date_range": date_range,  # Will be empty string if no dates found
            "companies": len(companies),
            "form_types": form_types,
            "industries": industries
        }
        
        # Add query pattern detection
        query_info = api_response.get("query_info", {})
        original_query = query_info.get("original_query", "")
        
        # Detect query pattern
        pattern = "unknown"
        if "ticker:" in original_query and "text:" in original_query:
            pattern = "company_section"
        elif "sicDescription:" in original_query and "text:" in original_query:
            pattern = "industry_policy"
        elif "sicDescription:" in original_query:
            pattern = "industry"
        elif "ticker:" in original_query and "OR" in original_query:
            pattern = "company_comparison"
        elif "filerCategory:" in original_query:
            pattern = "filing_status"
        elif "ticker:" in original_query:
            pattern = "company_specific"
        
        enhanced_response["metadata"]["query_pattern"] = pattern
        
        return enhanced_response
    
    def format(self, question: str, api_response: Dict) -> str:
        """Format API response into user-friendly text with context-aware formatting."""
        # Preprocess the response to add context
        enhanced_response = self._preprocess_response(api_response)
        
        # Add the original question for context
        enhanced_response["original_question"] = question
        
        # Format the response using the LLM
        return self.chain.run(
            question=question, 
            api_response=json.dumps(enhanced_response, indent=2)
        ).strip()

class SECQueryAgent:
    """Agent for answering questions about SEC filings using the SEC API."""
    
    def __init__(self, llm: BaseLanguageModel, sec_api_key: str):
        """Initialize the SEC Query Agent."""
        # Initialize components
        self.llm = llm
        # Set temperature to 0 for consistent outputs
        if hasattr(self.llm, 'temperature'):
            self.llm.temperature = 0
        self.query_tool = SECQueryTool(sec_api_key=sec_api_key)
        self.query_translator = QueryTranslator(llm=llm)
        self.response_formatter = ResponseFormatter(llm=llm)
        
        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in SEC filings and financial documents. 
            Your job is to help users find information about SEC filings by querying the SEC database.
            Use the sec_query_tool to search for relevant filings based on the user's question.
            
            When searching for specific document types or exhibits:
            - Use documentFormatFiles.type for finding specific exhibits (e.g., "EX-21" for Exhibit 21)
            - Use documentFormatFiles.description for finding specific sections
            
            IMPORTANT QUERY GUIDELINES:
            1. Always use explicit date ranges with YYYY-MM-DD format. Do NOT use NOW, NOW-1YEAR, or other relative date formats.
               - CORRECT: filedAt:[2023-01-01 TO 2023-12-31]
               - INCORRECT: filedAt:[NOW-1YEAR TO NOW]
            
            2. For company searches, you can use either ticker symbol or CIK number:
               - By ticker: ticker:AAPL
               - By CIK: cik:1318605 (Tesla's CIK)
            
            Examples of effective queries:
            - For subsidiaries information: formType:"10-K" AND documentFormatFiles.type:"EX-21"
            - For graphical content: formType:* AND documentFormatFiles.type:"GRAPHIC" AND filedAt:[2023-01-01 TO 2023-12-31]
            - For management discussion: formType:"10-K" AND documentFormatFiles.description:"Management's Discussion"
            - For earnings results: formType:"8-K" AND description:"Item 2.02"
            - For Tesla's 2023 filings: cik:1318605 AND filedAt:[2023-01-01 TO 2023-12-31]
            - For industry-specific filings: formType:"10-K" AND entities.sic:7370 AND filedAt:[2023-10-01 TO 2023-12-31]
            
            Always prioritize field-specific searches over general text searches for better results."""),
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
        """Process a user question end-to-end."""
        try:
            # Check if the question contains time-related patterns that need special handling
            time_pattern = re.search(r'last\s+(\d+\s+)?(year|month|quarter|week|day)s?|last\s+year|Q[1-4]\s+\d{4}', question.lower())
            
            if time_pattern:
                # If we have a time pattern, use the query translator to handle it properly
                print(f"\nDEBUG: Detected time pattern in question: '{time_pattern.group(0)}'")
                translated_query = self.query_translator.translate(question)
                print(f"DEBUG: Translated query: {translated_query}")
                
                # Use the translated query directly with the SEC query tool
                response = self.query_tool._run(translated_query)
                
                # Add the query to the response for debugging
                if isinstance(response, dict):
                    response['query'] = translated_query
                
                result = {
                    "output": f"I searched for {translated_query} and found results.",
                    "intermediate_steps": [
                        (AgentAction(tool="sec_query_tool", tool_input={"query": translated_query}, log=""), response)
                    ],
                    "query": translated_query  # Store the query for later reference
                }
            else:
                # For questions without time patterns, use the agent executor
                result = self.agent_executor.invoke({"input": question})
            
            # Extract the output text from the agent result
            output_text = result.get("output", "")
            
            # Get the SEC API response from the intermediate steps
            sec_api_response = None
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    # Each step is a tuple of (action, action_output)
                    if len(step) >= 2 and step[0].tool == "sec_query_tool":
                        # We found the SEC API response
                        sec_api_response = step[1]
                        break
            
            # If we found a SEC API response, use it
            if sec_api_response and isinstance(sec_api_response, dict) and "filings" in sec_api_response:
                # Get total count and sample filings
                total_count = sec_api_response.get("total", {}).get("value", 0)
                filings = sec_api_response.get("filings", [])
                
                if filings:
                    # Limit to first 3 filings to avoid context length issues
                    sample_filings = filings[:3]
                    filing_details = "\n".join([f"- {f.get('companyName', 'Unknown')} ({f.get('filedAt', '').split('T')[0]}): {f.get('description', '')}" for f in sample_filings])
                    
                    # Create a concise formatted response
                    formatted_response = f"Found {total_count} matching filings. Here are some examples:\n\n{filing_details}"
                    
                    # Add a link to the first filing
                    if sample_filings[0].get('linkToFilingDetails'):
                        formatted_response += f"\n\nView details: {sample_filings[0].get('linkToFilingDetails')}"
                else:
                    formatted_response = "No matching filings found."
            else:
                # Use the agent's output text
                formatted_response = output_text
            
            # Include the query in the response if it was generated by our translator
            query = result.get("query", None)
            response = {
                "raw_result": result,
                "formatted_response": formatted_response
            }
            if query:
                response["query"] = query
            
            return response
            
        except Exception as e:
            # Log the error and return it to the user
            error_message = f"Error processing question: {str(e)}"
            print(f"ERROR: {error_message}")
            
            return {
                "error": error_message,
                "formatted_response": f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question or check your API credentials."
            }

def create_sec_query_agent(llm: BaseLanguageModel, sec_api_key: str) -> SECQueryAgent:
    """
    Create and return a SEC Query Agent instance.
    
    Args:
        llm: A Langchain LLM instance
        sec_api_key: SEC API key for authentication
        
    Returns:
        An initialized SECQueryAgent instance
    """
    return SECQueryAgent(llm=llm, sec_api_key=sec_api_key)
