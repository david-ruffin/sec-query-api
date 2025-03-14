# sec_query_api.py
from sec_api import QueryApi
import openai
import os
import json

class SecQueryTool:
    """Simple wrapper for the SEC Query API."""
    
    def __init__(self, sec_api_key, openai_api_key, openai_model="gpt-4o"):
        self.sec_api = QueryApi(api_key=sec_api_key)
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        openai.api_key = openai_api_key
    
    def translate_query(self, user_question):
        """Translate a natural language question into a SEC API query."""
        
        system_prompt = """
        You are an expert in SEC filings and the SEC API. Translate the user's question into a proper SEC API query using Lucene syntax.
        
        Some examples:
        - "Find Apple's latest 10-K" → ticker:AAPL AND formType:"10-K"
        - "Show me Tesla's 8-K filings from last year" → ticker:TSLA AND formType:"8-K" AND filedAt:[2022-01-01 TO 2022-12-31]
        - "Get Microsoft's quarterly reports from 2023" → ticker:MSFT AND formType:"10-Q" AND filedAt:[2023-01-01 TO 2023-12-31]
        
        Return ONLY the query string, nothing else.
        ALWAYS use double quotes around form types: formType:"10-K"
        """
        
        response = openai.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=0
        )
        
        # Extract the query
        query = response.choices[0].message.content.strip()
        print(f"Translated query: {query}")
        
        return query
    
    def search_filings(self, query, from_param=0, size=10):
        """Execute a search against the SEC API."""
        search_params = {
            "query": query,
            "from": str(from_param),
            "size": str(size),
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        try:
            response = self.sec_api.get_filings(search_params)
            return response
        except Exception as e:
            return {"error": str(e)}
    
    def process_user_question(self, question):
        """Process a user question end-to-end."""
        
        # Step 1: Translate to SEC API query
        query = self.translate_query(question)
        
        # Step 2: Execute the query
        raw_results = self.search_filings(query)
        
        # Step 3: Format the results into a readable response
        system_prompt = """
        You are an expert in SEC filings. Summarize the results from an SEC API query in a clear, user-friendly way.
        Focus on the most important information:
        - Company names and tickers
        - Filing types and dates
        - Brief description of each filing
        
        Keep your response concise but informative.
        """
        
        response = openai.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nSEC API Results: {json.dumps(raw_results, indent=2)}"}
            ],
            temperature=0
        )
        
        return {
            "query": query,
            "raw_results": raw_results,
            "formatted_response": response.choices[0].message.content
        }