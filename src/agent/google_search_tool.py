import os
import requests
from src.utils.logger import get_logger

logger = get_logger()

class GoogleCustomSearch:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
        if not self.api_key or not self.cse_id:
            logger.error("Google Custom Search API key or CSE ID not set in environment variables.")
            raise ValueError("Missing Google Custom Search API key or CSE ID.")

    def search(self, query, num_results=5):
        logger.info(f"Calling Google Custom Search for query: '{query}' (num_results={num_results})")
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": num_results
        }
        try:
            response = requests.get(url, params=params)
            logger.debug(f"Google API request URL: {response.url}")
            response.raise_for_status()
            results = response.json().get("items", [])
            logger.info(f"Google Custom Search returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error during Google Custom Search: {str(e)}")
            return []
