from .google_search_tool import GoogleCustomSearch

class InternetSearchAgent:
    """Agent that uses Google Custom Search to fill information gaps."""
    def __init__(self):
        self.search_tool = GoogleCustomSearch()

    def search_if_gap(self, reasoning: str, query: str, min_gap_keywords=2):
        """
        If reasoning indicates information gaps, perform a Google search.
        Args:
            reasoning: The agent's reasoning output
            query: The original user query
            min_gap_keywords: Minimum number of keywords indicating a gap
        Returns:
            List of search results (dicts)
        """
        gap_keywords = [
            'insufficient', 'unknown', 'not found', 'uncertain', 'missing',
            'no information', 'cannot answer', 'lack of data', 'unavailable',
            'not enough', 'not sufficient', 'not provided', 'not covered',
        ]
        gap_count = sum(kw in reasoning.lower() for kw in gap_keywords)
        if gap_count >= min_gap_keywords:
            from src.utils.logger import get_logger
            logger = get_logger()
            logger.info(f"Reasoning indicates information gap ({gap_count} keywords). Triggering Google search.")
            return self.search_tool.search(query)
        return []
