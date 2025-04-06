"""
Streamlit-specific utility functions for cloud deployment
"""

import os
import streamlit as st
from typing import Optional

def get_api_key(key_name: str) -> Optional[str]:
    """
    Get API key from environment variables or Streamlit secrets
    
    This function prioritizes:
    1. Environment variables (for local development)
    2. Streamlit secrets (for cloud deployment)
    
    Args:
        key_name: The name of the API key to retrieve
        
    Returns:
        The API key or None if not found
    """
    # Try environment variables first (for local development)
    api_key = os.getenv(key_name)
    
    # If not in environment, try Streamlit secrets
    if not api_key and hasattr(st, 'secrets'):
        # Check direct key existence
        if key_name in st.secrets:
            api_key = st.secrets[key_name]
        # Check in api_keys section
        elif 'api_keys' in st.secrets and key_name in st.secrets['api_keys']:
            api_key = st.secrets['api_keys'][key_name]
    
    return api_key
