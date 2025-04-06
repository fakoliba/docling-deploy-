"""
Document and Embedding Caching System for Document Q&A Application

This module provides caching functionality for:
1. Document embeddings to reduce API calls
2. Processed documents to speed up repeat access
3. Document metadata for quick retrieval
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

# Cache directory structure
CACHE_DIR = Path("data/cache")
EMBEDDING_CACHE_DIR = CACHE_DIR / "embeddings"
DOCUMENT_CACHE_DIR = CACHE_DIR / "documents"
METADATA_CACHE_DIR = CACHE_DIR / "metadata"


def init_cache_dirs():
    """Initialize cache directory structure"""
    for directory in [EMBEDDING_CACHE_DIR, DOCUMENT_CACHE_DIR, METADATA_CACHE_DIR]:
        os.makedirs(directory, exist_ok=True)
    logger.info("Cache directories initialized")


def get_cache_key(content: str, prefix: str = "") -> str:
    """
    Generate a unique cache key for content
    
    Args:
        content: Content to generate key for
        prefix: Optional prefix for the key
        
    Returns:
        Unique cache key
    """
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return f"{prefix}_{content_hash}" if prefix else content_hash


def cache_embedding(text: str, embedding: List[float]):
    """
    Cache an embedding vector for text
    
    Args:
        text: Text that was embedded
        embedding: The embedding vector
    """
    cache_key = get_cache_key(text, "emb")
    cache_path = EMBEDDING_CACHE_DIR / f"{cache_key}.pkl"
    
    with open(cache_path, 'wb') as f:
        pickle.dump(embedding, f)
    
    # Store metadata about this embedding for tracking
    metadata = {
        "text_length": len(text),
        "created": datetime.now().isoformat(),
        "model": "text-embedding-ada-002"  # Update this if model changes
    }
    
    metadata_path = EMBEDDING_CACHE_DIR / f"{cache_key}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    logger.debug(f"Cached embedding for {cache_key}")


def get_cached_embedding(text: str) -> Optional[List[float]]:
    """
    Retrieve cached embedding for text if available
    
    Args:
        text: Text to get embedding for
        
    Returns:
        Embedding vector if cached, None otherwise
    """
    cache_key = get_cache_key(text, "emb")
    cache_path = EMBEDDING_CACHE_DIR / f"{cache_key}.pkl"
    
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                embedding = pickle.load(f)
            logger.debug(f"Retrieved cached embedding for {cache_key}")
            return embedding
        except Exception as e:
            logger.error(f"Error loading cached embedding: {str(e)}")
    
    return None


def cache_processed_document(file_path: str, chunks: List[str], metadata: Dict[str, Any]):
    """
    Cache processed document chunks and metadata
    
    Args:
        file_path: Original file path or identifier
        chunks: Processed text chunks
        metadata: Document metadata
    """
    cache_key = get_cache_key(file_path)
    
    # Save chunks
    chunks_path = DOCUMENT_CACHE_DIR / f"{cache_key}_chunks.pkl"
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    
    # Save metadata
    metadata['cached_time'] = datetime.now().isoformat()
    metadata_path = METADATA_CACHE_DIR / f"{cache_key}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    logger.info(f"Cached document {file_path} with {len(chunks)} chunks")


def get_cached_document(file_path: str) -> Optional[Tuple[List[str], Dict[str, Any]]]:
    """
    Retrieve cached document if available
    
    Args:
        file_path: Original file path or identifier
        
    Returns:
        Tuple of (chunks, metadata) if cached, None otherwise
    """
    cache_key = get_cache_key(file_path)
    chunks_path = DOCUMENT_CACHE_DIR / f"{cache_key}_chunks.pkl"
    metadata_path = METADATA_CACHE_DIR / f"{cache_key}.json"
    
    if chunks_path.exists() and metadata_path.exists():
        try:
            # Load chunks
            with open(chunks_path, 'rb') as f:
                chunks = pickle.load(f)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Retrieved cached document {file_path}")
            return chunks, metadata
        except Exception as e:
            logger.error(f"Error loading cached document: {str(e)}")
    
    return None


def cache_document_annotations(file_path: str, annotations: List[Dict]):
    """
    Cache annotations for a document
    
    Args:
        file_path: Original file path or identifier
        annotations: List of annotation dictionaries
    """
    cache_key = get_cache_key(file_path)
    annotations_path = DOCUMENT_CACHE_DIR / f"{cache_key}_annotations.json"
    
    # Add timestamp to each annotation
    for annotation in annotations:
        if 'created' not in annotation:
            annotation['created'] = datetime.now().isoformat()
    
    # Save annotations
    with open(annotations_path, 'w') as f:
        json.dump(annotations, f)
    
    logger.info(f"Cached {len(annotations)} annotations for document {file_path}")


def get_cached_annotations(file_path: str) -> List[Dict]:
    """
    Retrieve cached annotations for a document
    
    Args:
        file_path: Original file path or identifier
        
    Returns:
        List of annotation dictionaries, empty list if none found
    """
    cache_key = get_cache_key(file_path)
    annotations_path = DOCUMENT_CACHE_DIR / f"{cache_key}_annotations.json"
    
    if annotations_path.exists():
        try:
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)
            logger.info(f"Retrieved {len(annotations)} cached annotations for {file_path}")
            return annotations
        except Exception as e:
            logger.error(f"Error loading cached annotations: {str(e)}")
    
    return []


def clear_cache(cache_type: str = "all"):
    """
    Clear specified type of cache
    
    Args:
        cache_type: Type of cache to clear ("embeddings", "documents", "metadata", or "all")
    """
    if cache_type == "embeddings" or cache_type == "all":
        for file in EMBEDDING_CACHE_DIR.glob("*"):
            file.unlink()
        logger.info("Cleared embeddings cache")
    
    if cache_type == "documents" or cache_type == "all":
        for file in DOCUMENT_CACHE_DIR.glob("*"):
            file.unlink()
        logger.info("Cleared documents cache")
    
    if cache_type == "metadata" or cache_type == "all":
        for file in METADATA_CACHE_DIR.glob("*"):
            file.unlink()
        logger.info("Cleared metadata cache")

# Initialize cache directories on module import
init_cache_dirs()
