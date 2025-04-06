#!/usr/bin/env python
"""
Environment and API Configuration Checker for Document Q&A
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_api_keys():
    """Check if required API keys are available"""
    print("Checking API keys...")
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✅ OpenAI API key found")
        print(f"   Length: {len(openai_key)} characters")
    else:
        print("❌ OpenAI API key not found")
        print("   Make sure OPENAI_API_KEY is defined in your .env file")
    
    # Check Tavily API key
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        print("✅ Tavily API key found")
        print(f"   Length: {len(tavily_key)} characters")
    else:
        print("❌ Tavily API key not found")
        print("   Make sure TAVILY_API_KEY is defined in your .env file")
        print("   Internet search will not work without this key")
    
    return bool(openai_key), bool(tavily_key)

def check_tavily_package():
    """Check if Tavily package is installed"""
    print("\nChecking Tavily package...")
    try:
        import tavily
        from tavily import TavilyClient
        print(f"✅ Tavily package installed (version: {tavily.__version__})")
        return True
    except ImportError:
        print("❌ Tavily package not installed")
        print("   Install it with: pip install tavily-python")
        return False

def test_tavily_search(query="What is the capital of France?"):
    """Test Tavily search functionality"""
    print(f"\nTesting Tavily search with query: '{query}'")
    
    # Check if Tavily key exists
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        print("❌ Cannot test search: Tavily API key not found")
        return False
    
    # Check if package is installed
    try:
        from tavily import TavilyClient
    except ImportError:
        print("❌ Cannot test search: Tavily package not installed")
        return False
    
    # Try performing a search
    try:
        client = TavilyClient(api_key=tavily_key)
        print("✅ Successfully initialized Tavily client")
        
        # Perform search
        print("   Performing search...")
        result = client.search(
            query=query,
            search_depth="basic",
            include_answer=True,
            max_results=2
        )
        
        # Check results
        if result and isinstance(result, dict):
            print("✅ Successfully received search results")
            
            if "results" in result and result["results"]:
                print(f"   Found {len(result['results'])} results")
                for i, res in enumerate(result["results"], 1):
                    print(f"   Result {i}: {res.get('title', 'No title')}")
            else:
                print("   No results found in response")
            
            if "answer" in result and result["answer"]:
                print(f"   Answer summary: {result['answer'][:100]}...")
            
            return True
        else:
            print(f"❌ Unexpected response format: {type(result)}")
            return False
            
    except Exception as e:
        print(f"❌ Error during search test: {str(e)}")
        return False

if __name__ == "__main__":
    print("Document Q&A Environment Check")
    print("=" * 40)
    
    # Check for .env file
    env_path = Path(".env")
    if env_path.exists():
        print(f"✅ .env file found at {env_path.absolute()}")
    else:
        print(f"❌ .env file not found at {env_path.absolute()}")
        print("   Create a .env file with your API keys")
    
    # Check API keys
    has_openai, has_tavily = check_api_keys()
    
    # Check Tavily package
    has_tavily_pkg = check_tavily_package()
    
    # Test search if we have the key and package
    if has_tavily and has_tavily_pkg:
        test_tavily_search()
    
    print("\nSummary:")
    print(f"OpenAI API: {'Available' if has_openai else 'Missing'}")
    print(f"Tavily API: {'Available' if has_tavily else 'Missing'}")
    print(f"Tavily Package: {'Installed' if has_tavily_pkg else 'Not installed'}")
    print(f"Internet Search: {'Should work' if (has_tavily and has_tavily_pkg) else 'Will not work'}")
