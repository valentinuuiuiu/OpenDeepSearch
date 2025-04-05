import os
import requests
from typing import List, Dict, Any, Optional, Union
import litellm
from smolagents import Tool

class OpenDeepSearchTool(Tool):
    """
    A lightweight yet powerful search tool designed for seamless integration with AI agents.
    It enables deep web search and retrieval, optimized for use with SmolAgents ecosystem.
    """
    
    name = "web_search"
    description = "Searches the web for information about the query"
    # Define input schema
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to find information about"
        }
    }#, "additionalProperties": false # Removed due to syntax error
    # Add outputs schema
    outputs = {
        "query": {
            "type": "string",
            "description": "The original search query"
        },
        "results": {
            "type": "array",
            "description": "List of search results ranked by relevance",
            "items": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the search result"
                    },
                    "snippet": {
                        "type": "string",
                        "description": "Text snippet or summary from the search result"
                    },
                    "url": {
                        "type": "string",
                        "description": "URL of the search result"
                    }
                }
            }
        }
    },
    # Change output_type to "json"
    output_type = "object"
    
    def __init__(
        self,
        model_name="openrouter/google/gemini-2.0-flash-001",
        reranker="jina",
        search_provider="serper",
        serper_api_key=None,
        searxng_instance_url=None,
        searxng_api_key=None,
        mode="default",  # 'default' or 'pro'
        **kwargs         # accept additional unused kwargs
    ):
        """
        print("OpenDeepSearchTool.__init__ is being executed")
        Initialize the OpenDeepSearchTool.
        
        Args:
            model_name: The LLM model to use for search-related operations
            reranker: The reranking engine to use ('jina' or 'infinity')
            serper_api_key: API key for Serper (defaults to SERPER_API_KEY env var)
            searxng_instance_url: URL for SearXNG instance (required if search_provider is searxng)
            searxng_api_key: API key for SearXNG (optional)
            mode: Search mode - 'default' for quick results or 'pro' for in-depth search
        """
        # Initialize the base Tool class
        super().__init__()
        
        self.model_name = model_name
        self.reranker = reranker
        self.mode = mode
        
        # Set up Serper API key
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        
        # Validate configuration
        if not self.serper_api_key:
            raise ValueError("Serper API key is required. Please provide serper_api_key or set SERPER_API_KEY environment variable.")
    
    def call(self, query: str) -> Dict[str, Any]:
        """
        Execute the tool with the given input.
        This method is required by the Tool base class.
        """
        # If query is a dict with a 'query' key, extract the query string
        if isinstance(query, dict) and 'query' in query:
            query = query['query']
            
        return self.__call__(query)
    
    def forward(self, query: str) -> Dict[str, Any]:
        """
        Execute a search query and return ranked results.
        This is the main method to use when calling the tool directly.
        """
        return self.__call__(query)
    
    def __call__(self, query: str) -> Dict[str, Any]:
        """
        Execute a search query and return ranked results.
        """
        # Get raw search results
        raw_results = self._search_with_serper(query)
        
        # Rerank the results
        ranked_results = self._rerank_results(query, raw_results)
        
        return {
            "query": query,
            "results": ranked_results
        }
    
    def _search_with_serper(self, query: str) -> List[Dict[str, str]]:
        """
        Search using Serper API.
        """
        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }
        data = {
            'q': query,
            'gl': 'us',
            'hl': 'en'
        }
        
        response = requests.post(
            'https://google.serper.dev/search',
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"Serper API error: {response.status_code}, {response.text}")
        
        results = []
        response_data = response.json()
        
        # Process organic results
        if 'organic' in response_data:
            for item in response_data['organic']:
                results.append({
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'url': item.get('link', '')
                })
                
        return results
    
    def _rerank_results(self, query: str, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Rerank results using the specified reranker.
        """
        if not results:
            return []
            
        if self.reranker == "jina":
            return self._rerank_with_jina(query, results)
        elif self.reranker == "infinity":
            return self._rerank_with_infinity(query, results)
        else:
            return results  # Return unranked if reranker not recognized
    
    def _rerank_with_jina(self, query: str, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Rerank using LLM-based reranking approach.
        """
        # Use the model to process and rerank results
        content_to_rank = [f"Title: {r['title']}\nSnippet: {r['snippet']}\nURL: {r['url']}" for r in results]
        
        prompt = f"""Rank the following search results based on relevance to the query: "{query}"
        
Search results:
{chr(10).join([f"{i+1}. {content}" for i, content in enumerate(content_to_rank)])}

Return the indices of the results in order of relevance, most relevant first.
Only return the indices as a comma-separated list, like: 3,1,2,4
"""
        
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse the ranking
            try:
                ranking = [int(idx.strip()) for idx in content.split(',') if idx.strip().isdigit()]
                # Adjust for 1-based indexing
                ranking = [idx - 1 for idx in ranking if 0 < idx <= len(results)]
                
                # Add any missing indices at the end
                all_indices = set(range(len(results)))
                existing = set(ranking)
                ranking.extend(list(all_indices - existing))
                
                # Return results in ranked order
                return [results[idx] for idx in ranking if idx < len(results)]
                
            except Exception:
                # If parsing fails, return original results
                return results
                
        except Exception as e:
            print(f"Reranking error: {str(e)}")
            return results
            
    def _rerank_with_infinity(self, query: str, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Rerank results using the Infinity Embeddings model.
        This is a fallback to Jina for now.
        """
        return self._rerank_with_jina(query, results)
