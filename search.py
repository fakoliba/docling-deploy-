import os
import re
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np
import openai
import lancedb
import streamlit as st
import logging
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from utils.streamlit_utils import get_api_key

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

@st.cache_resource
def init_db():
    """
    Initialize the LanceDB connection and create or open the document table.
    
    Returns:
        LanceDB Table
    """
    try:
        # Create a folder for the database if it doesn't exist
        os.makedirs("data/lancedb", exist_ok=True)
        
        # Connect to the database
        uri = "data/lancedb"
        db = lancedb.connect(uri)
        
        # Define schema for the documents table
        import pyarrow as pa
        
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("chunk", pa.int32()),
            pa.field("embedding", pa.list_(pa.float32(), 1536)),
            pa.field("timestamp", pa.string()),
            pa.field("metadata", pa.string())
        ])
        
        # Create or open the table
        try:
            table = db.open_table("docling")
            logger.info("Opened existing table 'docling'")
        except Exception as e:
            logger.info(f"Creating new table 'docling': {str(e)}")
            table = db.create_table("docling", schema=schema)
        
        return table
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        st.error(f"Error initializing database: {str(e)}")
        # Return a dummy table object for graceful degradation
        return None

def get_embedding(text: str) -> List[float]:
    """
    Get embedding for a text using OpenAI's embedding model.
    
    Args:
        text: Text to generate embedding for
        
    Returns:
        List[float]: The embedding vector
    """
    try:
        # Truncate text to avoid token limits (roughly 8191 tokens)
        truncated_text = text[:25000]
        
        # Get embedding from OpenAI
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=truncated_text
        )
        
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        # Return a vector of zeros as fallback
        return [0.0] * 1536  # OpenAI's ada-002 embeddings are 1536 dimensions

def add_documents_to_db(chunks: List[str], source: str, table, metadata: Dict[str, Any] = None):
    """
    Add processed document chunks to the vector database.
    
    Args:
        chunks: List of text chunks
        source: Source file name
        table: LanceDB table
        metadata: Additional metadata
    """
    if metadata is None:
        metadata = {}
    
    # Prepare batch of documents for insertion
    data = []
    timestamp = datetime.now().isoformat()
    
    for i, chunk in enumerate(chunks):
        # Generate unique ID
        chunk_id = f"{source}_{i}_{timestamp}"
        
        # Get embedding for chunk
        embedding = get_embedding(chunk)
        
        # Serialize metadata to JSON string
        metadata_json = json.dumps(metadata)
        
        # Prepare document
        doc = {
            "id": chunk_id,
            "text": chunk,
            "source": source,
            "page": metadata.get("page", 0),
            "chunk": i,
            "embedding": embedding,
            "timestamp": timestamp,
            "metadata": metadata_json
        }
        
        data.append(doc)
    
    # Add documents to database
    table.add(data)

# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def parse_boolean_query(query: str) -> Dict:
    """
    Parse a query string with boolean operators (AND, OR, NOT).
    
    Args:
        query: User query with possible boolean operators
        
    Returns:
        Dict with 'must', 'should', and 'must_not' terms
    """
    # Convert to lowercase for easier processing, but preserve original for display
    query_lower = query.lower()
    
    # Initialize result structure
    parsed = {
        "original": query,
        "must": [],      # Terms that MUST be present (AND)
        "should": [],    # Terms that SHOULD be present (OR)
        "must_not": []   # Terms that MUST NOT be present (NOT)
    }
    
    # Extract NOT terms first (to avoid conflicts with AND, OR)
    not_parts = re.findall(r'not\s+(\w+)', query_lower)
    for part in not_parts:
        parsed["must_not"].append(part)
        # Remove the NOT term from query to avoid processing it again
        query_lower = query_lower.replace(f"not {part}", "")
    
    # Extract AND terms
    if " and " in query_lower:
        and_parts = [part.strip() for part in query_lower.split(" and ")]
        for part in and_parts:
            if part and part not in parsed["must"] and part not in parsed["must_not"]:
                parsed["must"].append(part)
        return parsed  # If AND is present, we prioritize it
        
    # If no AND terms, look for OR terms
    elif " or " in query_lower:
        or_parts = [part.strip() for part in query_lower.split(" or ")]
        for part in or_parts:
            if part and part not in parsed["should"] and part not in parsed["must_not"]:
                parsed["should"].append(part)
        return parsed
        
    # If no boolean operators, treat as a single term that must match
    else:
        clean_query = query_lower.strip()
        if clean_query and clean_query not in parsed["must_not"]:
            parsed["must"].append(clean_query)
            
    return parsed

def keyword_search(table, terms: List[str], column: str = "text") -> pd.DataFrame:
    """
    Perform keyword-based search on a table.
    
    Args:
        table: LanceDB table
        terms: List of terms to search for
        column: Column to search in
        
    Returns:
        DataFrame with matching rows
    """
    if not terms:
        return pd.DataFrame()
        
    # Get all data
    all_data = table.to_pandas()
    
    # Filter rows that contain any of the terms
    mask = all_data[column].str.lower().apply(
        lambda text: any(term.lower() in text.lower() for term in terms)
    )
    
    return all_data[mask]

def filter_by_metadata(df, filters: Dict) -> pd.DataFrame:
    """
    Filter search results by metadata.
    
    Args:
        df: DataFrame with search results
        filters: Dictionary with filter criteria
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Filter by document type
    if "doc_type" in filters and filters["doc_type"]:
        doc_types = filters["doc_type"]
        if isinstance(doc_types, str):
            doc_types = [doc_types]
        filtered_df = filtered_df[filtered_df["source"].str.endswith(tuple(doc_types))]
    
    # Filter by date range
    if "date_after" in filters and filters["date_after"]:
        filtered_df = filtered_df[filtered_df["timestamp"] >= filters["date_after"]]
        
    if "date_before" in filters and filters["date_before"]:
        filtered_df = filtered_df[filtered_df["timestamp"] <= filters["date_before"]]
    
    return filtered_df

def hybrid_search(query: str, table, filters: Dict = None, num_results: int = 5) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform hybrid search combining vector similarity and keyword matching.
    
    Args:
        query: User's question
        table: LanceDB table object
        filters: Dictionary with filter criteria
        num_results: Number of results to return
        
    Returns:
        Tuple of (DataFrame with results, Dictionary with search metadata)
    """
    # Initialize metadata
    metadata = {
        "original_query": query,
        "vector_results_count": 0,
        "keyword_results_count": 0,
        "boolean_query": None,
        "applied_filters": filters or {}
    }
    
    # Parse boolean operators if present
    boolean_query = parse_boolean_query(query)
    metadata["boolean_query"] = boolean_query
    
    # Get search preferences from filters
    use_vector = filters.get("use_vector", True) if filters else True
    use_keyword = filters.get("use_keyword", True) if filters else True
    
    # Initialize results DataFrames
    vector_results = pd.DataFrame()
    keyword_results = pd.DataFrame()
    final_results = pd.DataFrame()
    search_metadata = {
        'vector_count': 0,
        'keyword_count': 0,
        'total_count': 0,
        'search_type': 'hybrid'
    }
    
    # Vector search
    if use_vector:
        try:
            # Get query embedding
            query_embedding = get_embedding(query)
            
            # Search the vector database
            vector_results = table.search(query_embedding).limit(num_results * 2).to_pandas()
            metadata["vector_results_count"] = len(vector_results)
            
            # Apply metadata filters if any
            if filters:
                vector_results = filter_by_metadata(vector_results, filters)
                
            # Update search metadata
            search_metadata['vector_count'] = len(vector_results)
            
        except Exception as e:
            logger.error(f"Vector search error: {str(e)}")
            # Continue with keyword search
    
    # Keyword search
    if use_keyword and boolean_query:
        try:
            # Use must terms if any, otherwise use should terms
            search_terms = boolean_query["must"] if boolean_query["must"] else boolean_query["should"]
            
            if search_terms:
                # Perform keyword search
                keyword_results = keyword_search(table, search_terms)
                metadata["keyword_results_count"] = len(keyword_results)
                
                # Apply metadata filters if any
                if filters:
                    keyword_results = filter_by_metadata(keyword_results, filters)
                
                # Apply NOT terms if any
                if boolean_query["must_not"]:
                    # Remove rows that contain any of the must_not terms
                    for term in boolean_query["must_not"]:
                        keyword_results = keyword_results[~keyword_results["text"].str.lower().str.contains(term.lower())]
                
                # Update search metadata
                search_metadata['keyword_count'] = len(keyword_results)
        
        except Exception as e:
            logger.error(f"Keyword search error: {str(e)}")
    
    # Combine results
    if not vector_results.empty and not keyword_results.empty:
        # Combine results from both methods
        combined = pd.concat([vector_results, keyword_results]).drop_duplicates(subset=["id"])
        
        # Sort by relevance (can be improved with a scoring function)
        # For now, prioritize vector search results
        final_results = combined.iloc[:num_results]
        
    elif not vector_results.empty:
        final_results = vector_results.iloc[:num_results]
        
    elif not keyword_results.empty:
        final_results = keyword_results.iloc[:num_results]
    
    # Final count
    search_metadata['total_count'] = len(final_results)
    metadata["final_count"] = len(final_results)
    
    # If no search yielded results, return empty DataFrame
    if final_results.empty:
        return final_results, metadata
    
    # Return results and metadata
    return final_results, metadata

def format_citations(results_df: pd.DataFrame) -> str:
    """
    Format search results with detailed citations.
    
    Args:
        results_df: DataFrame with search results
        
    Returns:
        Formatted context string with citations
    """
    if results_df.empty:
        return "No relevant information found."
    
    # Initialize context string
    context_parts = []
    
    # Function to extract metadata
    def extract_metadata(metadata_json):
        try:
            return json.loads(metadata_json)
        except:
            return {}
    
    # Process each result
    for i, row in results_df.iterrows():
        # Get source filename (remove path if any)
        source = os.path.basename(row.get("source", "unknown"))
        
        # Get page number if available
        page_info = f"Page {row.get('page', 0) + 1}" if "page" in row and row.get("page", 0) >= 0 else ""
        
        # Extract metadata
        metadata = extract_metadata(row.get("metadata", "{}"))
        
        # Format language info if available
        language_info = f"Language: {metadata.get('language', 'unknown').upper()}" if "language" in metadata else ""
        
        # Combine citation info
        citation = f"Source: {source}"
        if page_info:
            citation += f" | {page_info}"
        if language_info:
            citation += f" | {language_info}"
        
        # Add text with citation
        context_parts.append(f"\n--- [{i+1}] {citation} ---\n{row.get('text', '')}\n")
    
    # Combine all parts
    return "\n".join(context_parts)

def perform_internet_search(query: str) -> List[Dict]:
    """
    Perform an internet search for the given query using Tavily API.
    
    Args:
        query: Search query
        
    Returns:
        List of search results as dictionaries with title, content, and url
    """
    try:
        # Get Tavily API key using our utility function that handles both local and cloud deployment
        tavily_api_key = get_api_key("TAVILY_API_KEY")
        
        if not tavily_api_key:
            logger.error("Tavily API key not found in environment variables")
            return []
        
        try:
            # Dynamically import to handle missing package gracefully
            from tavily import TavilyClient
        except ImportError:
            logger.error("Tavily package not installed")
            return []
        
        # Initialize client
        client = TavilyClient(api_key=tavily_api_key)
        
        # Perform search
        search_result = client.search(
            query=query,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False,
            max_results=5
        )
        
        # Extract and format the results as a list of dictionaries
        results = []
        
        # Add sources
        if "results" in search_result and search_result["results"]:
            for result in search_result["results"]:
                title = result.get("title", "No Title")
                url = result.get("url", "No URL")
                content = result.get("content", "No Content")
                
                results.append({
                    "title": title,
                    "url": url,
                    "content": content
                })
        
        # If Tavily provided a summary answer, add it as the first result
        if "answer" in search_result and search_result["answer"]:
            results.insert(0, {
                "title": "Tavily AI Summary",
                "url": "",
                "content": search_result["answer"]
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Internet search error: {str(e)}")
        return []

def get_context(query: str, table, filters: Dict = None, num_results: int = 3) -> Tuple[str, Dict]:
    """
    Search the database for relevant context using hybrid search.

    Args:
        query: User's question
        table: LanceDB table object
        filters: Dictionary with filter criteria
        num_results: Number of results to return

    Returns:
        Tuple of (Context string, Search metadata)
    """
    # Perform hybrid search
    results_df, metadata = hybrid_search(query, table, filters, num_results)
    
    # Format citations
    context = format_citations(results_df)
    
    return context, metadata
