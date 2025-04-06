"""
Document Q&A Application with Multilingual Support and Internet Search

This application allows users to upload documents, ask questions, and get AI-powered answers.
Features include:
- Document upload and processing (PDF, TXT, DOCX, Images)
- Table extraction and visualization
- Mathematical formula support
- Natural language question answering
- Context retrieval from documents
- Internet search integration
- Multilingual support with auto-detection
"""

import os
import re
import io
import uuid
import json
import base64
import logging
import requests
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import time

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from dotenv import load_dotenv
from utils.streamlit_utils import get_api_key
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Import document processing libraries
from PyPDF2 import PdfReader
import docx2txt
import pytesseract
import camelot
from langdetect import detect, DetectorFactory
from sympy import parse_expr
import sympy

# Import AI and database libraries
import openai
import lancedb
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import our new modules

# Import our new modules
from cache import init_cache_dirs, cache_embedding, get_cached_embedding, cache_processed_document, get_cached_document
from annotation import render_pdf_annotation_ui
from comparison import render_document_comparison_ui, DocumentComparer
from metadata import extract_and_store_document_metadata, render_metadata_ui

# Import search module
import search

# Set page config to remove the top white bar - must be first Streamlit command
st.set_page_config(
    #page_title="Chat  with Sily",
    #page_icon="🐘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import styles and formatting functions
from styles import (
    apply_base_styles,
    render_app_title,
    initialize_chat_container,
    close_chat_messages_container,
    initialize_chat_input,
    close_chat_input_container,
    close_chat_container,
    USER_AVATAR,
    ASSISTANT_AVATAR
)

# Import all search-related functions
from search import (
    init_db,
    get_embedding,
    hybrid_search,
    perform_internet_search,
    get_context,
    parse_boolean_query
)

# Set seed for language detection for consistent results
DetectorFactory.seed = 0

# Define key functions needed early
def detect_language(text: str) -> str:
    """
    Detect the language of a text string.
    
    Args:
        text: The text to analyze
        
    Returns:
        str: Two-letter language code (e.g. 'en', 'fr', etc.)
    """
    try:
        # Use a minimum length to avoid errors with very short texts
        if len(text) < 20:
            return "en"  # Default to English for very short texts
            
        # Detect language
        lang = detect(text)
        return lang
    except Exception as e:
        logger.error(f"Language detection error: {str(e)}")
        return "en"  # Default to English on error

def detect_document_language(content: str) -> str:
    """
    Detect the language of a document based on its content.
    More robust than simple detection for multi-lingual documents.
    
    Args:
        content: The document content to analyze
        
    Returns:
        str: Two-letter language code (e.g. 'en', 'fr', etc.)
    """
    # For very short content, return English as default
    if len(content) < 100:
        return "en"
    
    # Take samples from the document to improve detection reliability
    samples = []
    
    # Start of document (first 1000 characters)
    if len(content) > 1000:
        samples.append(content[:1000])
    else:
        samples.append(content)
    
    # Middle of document if long enough
    if len(content) > 3000:
        middle_start = len(content) // 2 - 500
        samples.append(content[middle_start:middle_start + 1000])
    
    # End of document if long enough
    if len(content) > 2000:
        samples.append(content[-1000:])
    
    # Detect language for each sample
    langs = []
    for sample in samples:
        lang = detect_language(sample)
        langs.append(lang)
    
    # Count occurrences of each language
    lang_counts = {}
    for lang in langs:
        if lang in lang_counts:
            lang_counts[lang] += 1
        else:
            lang_counts[lang] = 1
    
    # Return the most common language
    most_common_lang = max(lang_counts, key=lang_counts.get)
    return most_common_lang

def translate_text(text: str, target_language: str) -> str:
    """
    Translate text to the target language using OpenAI.
    
    Args:
        text: Text to translate
        target_language: Target language code (e.g. 'en', 'fr', etc.)
        
    Returns:
        str: Translated text
    """
    try:
        # For very short texts or if we're already in the target language
        detected_lang = detect_language(text)
        if detected_lang == target_language or len(text) < 10:
            return text
        
        # Use OpenAI to translate
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a professional translator. Translate the following text to {target_language}. Preserve all formatting."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text  # Return original text on error

def translate_document_chunks(chunks: List[str], source_language: str, target_language: str) -> List[str]:
    """
    Translate a list of document chunks to the target language.
    
    Args:
        chunks: List of text chunks to translate
        source_language: Source language code
        target_language: Target language code
        
    Returns:
        List[str]: List of translated chunks
    """
    # If languages are the same, return original chunks
    if source_language == target_language:
        return chunks
    
    translated_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Skip empty chunks
        if not chunk.strip():
            translated_chunks.append(chunk)
            continue
        
        # Translate the chunk
        translated = translate_text(chunk, target_language)
        translated_chunks.append(translated)
    
    return translated_chunks

def extract_tables_from_pdf(pdf_file):
    """
    Extract tables from a PDF file using camelot.
    
    Args:
        pdf_file: PDF file object
        
    Returns:
        List of dataframes containing tables
    """
    try:
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(pdf_file.getvalue())
            temp_pdf_path = temp_pdf.name
        
        # Use camelot to extract tables
        tables = camelot.read_pdf(temp_pdf_path, pages='all', flavor='stream')
        
        # Cleanup temp file
        os.unlink(temp_pdf_path)
        
        # Process results
        extracted_tables = []
        for i, table in enumerate(tables):
            df = table.df
            # Generate a unique ID for this table
            table_id = f"table_{uuid.uuid4()}"
            
            # Store the table info
            extracted_tables.append({
                "id": table_id,
                "index": i,
                "dataframe": df,
                "page": table.page,
                "accuracy": table.accuracy,
                "whitespace": table.whitespace,
                "shape": df.shape
            })
        
        return extracted_tables
    except Exception as e:
        logger.error(f"Table extraction error: {str(e)}")
        return []

def perform_ocr(image_file):
    """
    Perform OCR on an image file using Tesseract.
    
    Args:
        image_file: Image file object
        
    Returns:
        Extracted text from the image
    """
    try:
        # Open image using PIL
        image = Image.open(image_file)
        
        # Perform OCR
        text = pytesseract.image_to_string(image)
        
        return text
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return f"Image processing error: {str(e)}"

def process_uploaded_file(uploaded_file, table):
    """
    Process an uploaded file to extract text and add it to the database.
    
    Args:
        uploaded_file: The uploaded file
        table: LanceDB table for document storage
    
    Returns:
        Tuple of (success, message, document_name)
    """
    start_time = time.time()
    st.write(f"Processing file: {uploaded_file.name}")
    
    # Initialize session state for documents if not exists
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = {}
    
    # Generate a unique ID for this document
    doc_id = f"{uploaded_file.name}_{uuid.uuid4()}"
    
    # Check if file is already in cache
    cached_doc = get_cached_document(uploaded_file.name)
    if cached_doc:
        chunks, cached_metadata = cached_doc
        st.success(f"Retrieved cached version of {uploaded_file.name}")
        
        # Store in session state
        st.session_state.uploaded_docs[doc_id] = {
            "name": uploaded_file.name,
            "chunks": chunks,
            "metadata": cached_metadata,
            "processed_time": datetime.now().isoformat()
        }
        
        # Add to database (will use cached embeddings where available)
        with st.spinner("Adding document to database..."):
            add_chunks_to_db(chunks, uploaded_file.name, table, cached_metadata)
        
        return True, f"Successfully processed cached file {uploaded_file.name}", doc_id
    
    # Extract detailed metadata
    with st.spinner("Extracting metadata..."):
        metadata = extract_and_store_document_metadata(uploaded_file, uploaded_file.name)
        
        # Add basic file properties to metadata
        metadata.update({
            "filename": uploaded_file.name,
            "file_type": uploaded_file.type,
            "file_size": uploaded_file.size,
            "extracted_tables": [],
            "has_formulas": False
        })
    
    # Process by file type
    if uploaded_file.type == "application/pdf":
        # Extract text from PDF
        pdf_reader = PdfReader(uploaded_file)
        total_pages = len(pdf_reader.pages)
        metadata["num_pages"] = total_pages
        
        # Batch process for large PDFs
        with st.spinner(f"Extracting text from PDF ({total_pages} pages)..."):
            # Use batch processing for large documents
            if total_pages > 20:
                # Process in batches of 20 pages
                batch_size = 20
                all_text = []
                
                progress_bar = st.progress(0)
                for batch_start in range(0, total_pages, batch_size):
                    batch_end = min(batch_start + batch_size, total_pages)
                    batch_pages = pdf_reader.pages[batch_start:batch_end]
                    batch_text = "\n".join([page.extract_text() or "" for page in batch_pages])
                    all_text.append(batch_text)
                    
                    # Update progress
                    progress = int((batch_end / total_pages) * 100)
                    progress_bar.progress(progress)
                    
                # Combine all batches
                text = "\n".join(all_text)
                progress_bar.empty()
            else:
                # For smaller documents, process all at once
                text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
        
        # Extract tables from PDF
        with st.spinner("Extracting tables from PDF..."):
            tables = extract_tables_from_pdf(uploaded_file)
            if tables:
                metadata["extracted_tables"] = [t["id"] for t in tables]
                metadata["table_count"] = len(tables)
                # Store tables in session state for later use
                if "pdf_tables" not in st.session_state:
                    st.session_state.pdf_tables = {}
                st.session_state.pdf_tables[uploaded_file.name] = tables
    
    elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
        # Perform OCR on image files
        with st.spinner("Performing OCR on image..."):
            text = perform_ocr(uploaded_file)
        metadata["image_type"] = uploaded_file.type
        metadata["ocr_processed"] = True
    
    else:
        # Default text extraction for other file types
        text = uploaded_file.getvalue().decode("utf-8", errors="replace")
    
    # Check for mathematical formulas
    if "$$" in text:
        metadata["has_formulas"] = True
    
    # Detect document language
    doc_language = detect_document_language(text)
    metadata["language"] = doc_language
    
    # Set up text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    
    # Split document into chunks
    chunks = text_splitter.split_text(text)
    
    # Translate if needed and translation is enabled
    if st.session_state.translate_documents and doc_language != st.session_state.translation_target_language:
        chunks = translate_document_chunks(
            chunks, 
            source_language=doc_language, 
            target_language=st.session_state.translation_target_language
        )
        metadata["translated"] = True
        metadata["original_language"] = doc_language
        metadata["target_language"] = st.session_state.translation_target_language
    
    # Cache processed document
    cache_processed_document(uploaded_file.name, chunks, metadata)
    
    # Store in session state
    st.session_state.uploaded_docs[doc_id] = {
        "name": uploaded_file.name,
        "chunks": chunks,
        "metadata": metadata,
        "processed_time": datetime.now().isoformat()
    }
    
    # Add to database
    with st.spinner("Adding document to database..."):
        add_chunks_to_db(chunks, uploaded_file.name, table, metadata)
    
    processing_time = time.time() - start_time
    st.success(f"Successfully processed {uploaded_file.name} in {processing_time:.2f} seconds")
    
    return True, f"Successfully processed {uploaded_file.name}", doc_id

def process_document_task(file_id, uploaded_file):
    """
    Process a single document and update its status.
    This function doesn't rely on accessing session_state directly.
    
    Args:
        file_id: Unique identifier for the file
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        dict: Processing result with status information
    """
    result = {
        "file_id": file_id,
        "filename": uploaded_file.name,
        "status": "processing",
        "progress": 0,
        "error": None
    }
    
    try:
        # Process the file
        success, message, doc_id = process_uploaded_file(uploaded_file, init_db())
        result["progress"] = 50
        
        # Mark as completed
        result["status"] = "completed"
        result["progress"] = 100
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result

# Try to import pytesseract, but provide helpful error message if not installed

    # Create a dummy function to prevent errors
def pytesseract_image_to_string(*args, **kwargs):
        return "OCR functionality not available. Please install pytesseract."
    
    # Create a dummy pytesseract module
class DummyPytesseract:
        def image_to_string(self, *args, **kwargs):
            return pytesseract_image_to_string(*args, **kwargs)
    
pytesseract = DummyPytesseract()

# Try to import camelot for table extraction

    # Create a dummy module
class DummyCamelot:
        class read_pdf:
            def __init__(self, *args, **kwargs):
                pass
            
            def __iter__(self):
                return iter([])
            
            def __getitem__(self, key):
                return None
    
camelot = DummyCamelot()

# Try to import sympy for formula parsing
try:
    import sympy
except ImportError:
    st.error("""
    SymPy is not properly installed. This is needed for mathematical formula rendering.
    
    Please install SymPy:
    `pip install sympy`
    
    Then restart the application.
    """)
    
    # Create a dummy sympy module
    class DummySympy:
        def sympify(self, *args, **kwargs):
            return "Formula rendering not available. Please install sympy."
    
    sympy = DummySympy()

# Try to import docx for Word document processing
try:
    import docx2txt
except ImportError:
    st.error("""
    docx2txt is not properly installed. This is needed for processing Word documents.
    
    Please install docx2txt:
    `pip install docx2txt`
    
    Then restart the application.
    """)
    
    def dummy_process(*args, **kwargs):
        return "Word document processing not available. Please install docx2txt."
    
    # Create a dummy module
    docx2txt = dummy_process

# Try to import plotly for visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    st.error("""
    Plotly is not installed. This is needed for interactive visualizations.
    
    Please install it with:
    ```
    pip install plotly
    ```
    Then restart the application.
    """)
    
    class DummyPlotly:
        def __init__(self, *args, **kwargs):
            pass

    class DummyFigure:
        def update_layout(self, *args, **kwargs):
            return self
            
    class DummyGo:
        def Figure(self, *args, **kwargs):
            return DummyFigure()
            
        def Table(self, *args, **kwargs):
            return None
            
    class DummyPx:
        def line(self, *args, **kwargs):
            return DummyFigure()
    
    px = DummyPx()
    go = DummyGo()

# Load environment variables
load_dotenv()
openai.api_key = get_api_key("OPENAI_API_KEY")

# Function definitions that might need to be moved higher
def add_chunks_to_db(chunks, source, table, metadata=None):
    """
    Add document chunks to the database with embeddings.
    
    Args:
        chunks: List of text chunks
        source: Source file name
        table: LanceDB table
        metadata: Optional metadata dictionary
    """
    if not table:
        st.error("Database connection failed. Cannot store document.")
        return
    
    data = []
    
    for i, chunk in enumerate(chunks):
        # Check for cached embedding first
        embedding = get_cached_embedding(chunk)
        
        if embedding is None:
            # No cached embedding, generate new one
            embedding = search.get_embedding(chunk)
            # Cache the embedding for future use
            cache_embedding(chunk, embedding)
        
        # Create doc_type (can be determined from file extension)
        doc_type = "txt"  # Default
        if "." in source:
            doc_type = source.split(".")[-1].lower()
        
        # Convert metadata to string
        metadata_str = json.dumps(metadata) if metadata else "{}"
        
        # Create a document matching the schema
        timestamp_seconds = int(datetime.now().timestamp())  # Convert to seconds precision
        
        doc = {
            "id": f"{source}_{i}_{uuid.uuid4()}",  # Generate a unique ID for each chunk
            "embedding": embedding,  # Changed from "vector" to "embedding" to match schema
            "text": chunk,
            "source": source,
            "page": 0,  # Default page number
            "chunk": i,  # Use chunk index
            "timestamp": timestamp_seconds,  # Use seconds precision
            "metadata": metadata_str
        }
        data.append(doc)
    
    # Add to database
    if data:
        table.add(data)
        logger.info(f"Added {len(data)} chunks to database from {source}")

# Apply base styles and render title
apply_base_styles()
render_app_title()

# Initialize session state variables
def initialize_session_state():
    """Initialize session state variables"""
    # User interface language (default to English)
    if "user_language" not in st.session_state:
        st.session_state.user_language = "en"
    
    # Chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # OpenAI model selection
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-3.5-turbo"
    
    # Document translation settings
    if "translate_documents" not in st.session_state:
        st.session_state.translate_documents = False
    
    if "translation_target_language" not in st.session_state:
        st.session_state.translation_target_language = "en"
    
    # Web search setting
    if "use_internet" not in st.session_state:
        st.session_state.use_internet = False
    
    # Document processing queue
    if "processing_queue" not in st.session_state:
        st.session_state.processing_queue = []
        
    # Document processing status tracking
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = {}
    
    # Document languages tracking
    if "document_languages" not in st.session_state:
        st.session_state.document_languages = {}
    
    # Store uploaded files for later use (needed for PDF annotation)
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}
        
    # Store tables from PDFs
    if "pdf_tables" not in st.session_state:
        st.session_state.pdf_tables = {}
    
    # Store uploaded documents info
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = {}
    
    # Toggle for showing search context
    if "show_context" not in st.session_state:
        st.session_state.show_context = False

# Initialize database connection
table = init_db()

# Initialize session state before using it
initialize_session_state()

# The sidebar content has been moved to the main() function
# to avoid duplicate elements

def visualize_table(df):
    """
    Create an interactive visualization of a table using Plotly.
    
    Args:
        df: Pandas DataFrame containing table data
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='#4CAF50',  # Green header
            align='center',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='#f9f9f9',  # Light gray
            align='left'
        )
    )])
    
    fig.update_layout(
        title="Table Visualization",
        margin=dict(l=0, r=0, t=30, b=0),
        height=400
    )
    
    return fig

def parse_and_render_formula(formula_text):
    """
    Previously parsed and rendered a mathematical formula using SymPy.
    Now just returns the original formula text as this feature has been removed.
    
    Args:
        formula_text: String containing a mathematical formula
        
    Returns:
        The original formula text
    """
    return formula_text

def detect_and_render_formulas(text):
    """
    Previously detected and rendered LaTeX mathematical formulas.
    Now just returns the original text as this feature has been removed.
    """
    # Simply return the original text without any MathJax rendering
    return text

def main():
    # Add custom CSS for the chat file upload
    st.markdown("""
    <style>
        /* Styling for the upload box */
        .upload-container {
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
        }
        
        .upload-box {
            border: 2px dashed #4A90E2;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            background-color: rgba(74, 144, 226, 0.05);
            transition: all 0.3s;
            width: 100%;
        }
        
        .upload-box:hover {
            background-color: rgba(74, 144, 226, 0.1);
            border-color: #2171cd;
        }
        
        /* Style for the + button */
        .upload-button button {
            background-color: white;
            color: #4A90E2;
            border: 1px solid #4A90E2;
            border-radius: 50%;
            width: 36px;
            height: 36px;
            font-size: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0;
            margin: 4px 0;
            transition: all 0.2s;
        }
        
        .upload-button button:hover {
            background-color: #4A90E2;
            color: white;
        }
        
        /* Enhance chat input for drag and drop */
        .stChatInput {
            border: 1px solid #e0e0e0;
            transition: all 0.3s;
        }
        
        .stChatInput:hover {
            border-color: #4A90E2;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables
    initialize_session_state()
    
    # Initialize cache directories
    init_cache_dirs()
    
    # Display dependency status in the sidebar
    with st.sidebar:
        # Check for missing packages
        missing_packages = []
        
        # Check PyMuPDF
        try:
            import fitz
        except ImportError:
            missing_packages.append("PyMuPDF")
        
        # Check scikit-learn
        try:
            from sklearn import __version__ 
        except ImportError:
            missing_packages.append("scikit-learn")
        
        # Check python-dateutil
        try:
            import dateutil
        except ImportError:
            missing_packages.append("python-dateutil")
        
        # Show installation instructions if packages are missing
        if missing_packages:
            st.warning("⚠️ Some advanced features are not available due to missing packages")
            with st.expander("Installation Instructions"):
                package_list = " ".join(missing_packages)
                st.code(f"pip install {package_list}", language="bash")
                st.write("After installing, restart the application.")
                
                # Also show conda command
                st.write("Or with conda:")
                conda_packages = " ".join(missing_packages).replace("PyMuPDF", "pymupdf")
                st.code(f"conda install {conda_packages}", language="bash")
        
        # Continue with the rest of the sidebar...
        st.title("Document Q&A")
            
        # Tab selection
        selected_tab = st.radio(
            "Navigation", 
            ["Chat", "Document Tools", "Settings"],
            horizontal=True
        )
            
        if selected_tab == "Chat":
            # Main query interface already shown
                
            # Web search settings
            st.subheader(
                "Search Settings" 
                if st.session_state.user_language == "en" 
                else "Paramètres de Recherche"
            )
                
            # Web search toggle
            st.session_state.use_internet = st.checkbox(
                "Enable web search" if st.session_state.user_language == "en" else "Activer la recherche web",
                value=st.session_state.use_internet
            )
                
            # Context display toggle
            st.session_state.show_context = st.checkbox(
                "Show retrieved context" if st.session_state.user_language == "en" else "Afficher le contexte récupéré",
                value=st.session_state.show_context
            )
        
        elif selected_tab == "Settings":
            st.subheader(
                "Language Settings" 
                if st.session_state.user_language == "en" 
                else "Paramètres de Langue"
            )
            
            # Language selector
            lang_options = {
                "en": "English",
                "fr": "Français",
                # "es": "Español",
                #"de": "Deutsch"
            }
            
            selected_lang = st.selectbox(
                "Interface Language" if st.session_state.user_language == "en" else "Langue de l'Interface",
                options=list(lang_options.keys()),
                format_func=lambda x: lang_options[x],
                index=list(lang_options.keys()).index(st.session_state.user_language),
                key="language_selector"
            )
            
            if selected_lang != st.session_state.user_language:
                st.session_state.user_language = selected_lang
                st.rerun()
            
            # Translation options
            with st.expander(
                "Document Translation" if st.session_state.user_language == "en" else "Traduction de Document",
                expanded=False
            ):
                st.session_state.translate_documents = st.checkbox(
                    "Translate documents" if st.session_state.user_language == "en" else "Traduire les documents",
                    value=st.session_state.translate_documents
                )
                
                if st.session_state.translate_documents:
                    target_lang_options = {
                        "en": "English",
                        "fr": "French",
                        "es": "Spanish",
                        "de": "German"
                    }
                    
                    st.session_state.translation_target_language = st.selectbox(
                        "Target language" if st.session_state.user_language == "en" else "Langue cible",
                        options=list(target_lang_options.keys()),
                        format_func=lambda x: target_lang_options[x],
                        index=list(target_lang_options.keys()).index(st.session_state.translation_target_language),
                        key="translation_target_lang"
                    )
            
            # API keys section removed as requested
            
        # Document Management Section
        st.subheader(
            "Document Management" 
            if st.session_state.user_language == "en" 
            else "Gestion des Documents"
        )
        
        # Information about uploading documents via chat
        st.info(
            "💡 Upload documents using the '+' button next to the chat input below." 
            if st.session_state.user_language == "en" 
            else "💡 Téléchargez des documents en utilisant le bouton '+' à côté de la zone de chat ci-dessous."
        )
        
        # Tool tabs for different document features
        if "uploaded_docs" in st.session_state and st.session_state.uploaded_docs:
            st.subheader(
                "Document Tools" 
                if st.session_state.user_language == "en" 
                else "Outils de Document"
            )
            
            tool_tabs = st.tabs([
                "Documents" if st.session_state.user_language == "en" else "Documents",
                "Compare" if st.session_state.user_language == "en" else "Comparer",
                "Annotate" if st.session_state.user_language == "en" else "Annoter"
            ])
            
            with tool_tabs[0]:
                # Display list of uploaded documents
                st.write(
                    "Uploaded Documents:" 
                    if st.session_state.user_language == "en" 
                    else "Documents Téléchargés:"
                )
                
                for doc_id, doc_info in st.session_state.uploaded_docs.items():
                    doc_name = doc_info.get("name", "Unknown document")
                    doc_time = doc_info.get("processed_time", "")
                    
                    # Format document time if available
                    time_str = ""
                    if doc_time:
                        try:
                            time_obj = datetime.fromisoformat(doc_time)
                            time_str = time_obj.strftime("%Y-%m-%d %H:%M")
                        except:
                            time_str = doc_time
                    
                    st.write(f"**{doc_name}** - {time_str}")
                    
                    # Document metadata
                    if "metadata" in doc_info:
                        render_metadata_ui(doc_info["metadata"])
                
                # Option to clear all documents
                if st.button(
                    "Clear All Documents" 
                    if st.session_state.user_language == "en" 
                    else "Effacer Tous les Documents"
                ):
                    st.session_state.uploaded_docs = {}
                    st.success("All documents cleared")
                    st.rerun()
            
            with tool_tabs[1]:
                # Render document comparison UI
                render_document_comparison_ui()
            
            with tool_tabs[2]:
                # PDF annotation UI
                st.write(
                    "Select a PDF to annotate:" 
                    if st.session_state.user_language == "en" 
                    else "Sélectionner un PDF à annoter:"
                )
                
                # Get list of PDF documents
                pdf_docs = []
                for doc_id, doc_info in st.session_state.uploaded_docs.items():
                    doc_name = doc_info.get("name", "")
                    if doc_name.lower().endswith('.pdf'):
                        pdf_docs.append((doc_id, doc_name))
                
                if not pdf_docs:
                    st.info(
                        "Upload a PDF document to use the annotation feature." 
                        if st.session_state.user_language == "en" 
                        else "Téléchargez un document PDF pour utiliser la fonction d'annotation."
                    )
                else:
                    selected_pdf = st.selectbox(
                        "Choose a PDF" if st.session_state.user_language == "en" else "Choisir un PDF",
                        options=[name for _, name in pdf_docs],
                        key="pdf_annotation_selector"
                    )
                    
                    # Get the selected PDF file for annotation
                    for doc_id, doc_name in pdf_docs:
                        if doc_name == selected_pdf:
                            # Create a temporary file for the PDF
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                # Get file from original upload
                                uploaded_file = st.session_state.uploaded_files.get(doc_name)
                                if uploaded_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    pdf_path = tmp_file.name
                                    
                                    # Render the PDF annotation UI
                                    render_pdf_annotation_ui(pdf_path, uploaded_file.getvalue())
                                else:
                                    st.error(f"Could not locate the original PDF file: {doc_name}")
        
        # Display extracted tables if present
        if "pdf_tables" in st.session_state and st.session_state.pdf_tables:
            st.subheader(
                "Extracted Tables" 
                if st.session_state.user_language == "en" 
                else "Tableaux Extraits"
            )
            
            for doc_name, tables in st.session_state.pdf_tables.items():
                if tables:
                    with st.expander(f"{doc_name} - {len(tables)} tables"):
                        # Create tabs for each table
                        if len(tables) > 0:
                            table_tabs = st.tabs([f"Table {i+1}" for i in range(len(tables))])
                            
                            for i, (tab, table) in enumerate(zip(table_tabs, tables)):
                                with tab:
                                    st.write(f"**Accuracy Score:** {table['accuracy']:.2f}")
                                    
                                    # Display the table
                                    st.dataframe(table['dataframe'])
                                    
                                    # Add visualization option
                                    if st.button(f"Visualize Table {i+1}", key=f"viz_{doc_name}_{i}"):
                                        fig = visualize_table(table['dataframe'])
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                    # Add download option
                                    csv = table['dataframe'].to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="Download CSV",
                                        data=csv,
                                        file_name=f"{doc_name}_table_{i+1}.csv",
                                        mime="text/csv",
                                        key=f"download_{doc_name}_{i}"
                                    )

    # Main chat container
    initialize_chat_container()
    
    # Messages container
    messages_container = st.container()
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with messages_container.chat_message(message["role"], avatar=USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR):
            if "tables" in message:
                # Special handling for messages with tables
                content = message["content"]
                tables = message["tables"]
                
                # Display regular content first
                st.markdown(content)
                
                # Display tables
                for i, table in enumerate(tables):
                    st.write(f"**Table {i+1}**")
                    st.dataframe(table)
            else:
                # Process and display mathematical formulas
                rendered_content = detect_and_render_formulas(message["content"])
                st.markdown(rendered_content, unsafe_allow_html=True)
    
    # Chat input container with file upload
    chat_input_cols = st.columns([0.95, 0.05])
    
    # File upload button (plus sign)
    with chat_input_cols[1]:
        st.markdown('<div class="upload-button">', unsafe_allow_html=True)
        upload_clicked = st.button("➕", key="upload_button")
        st.markdown('</div>', unsafe_allow_html=True)
        if upload_clicked:
            st.session_state.show_uploader = True
    
    # Initialize show_uploader state if not exists
    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = False
    
    # Show file uploader if button was clicked
    if st.session_state.show_uploader:
        st.markdown('<div class="upload-container"><div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a document or drag & drop files here",
            type=["pdf", "docx", "txt", "csv", "jpg", "jpeg", "png"],
            key="chat_file_uploader"
        )
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Add a cancel button
        col1, col2, col3 = st.columns([0.4, 0.2, 0.4])
        with col2:
            if st.button("Cancel", key="cancel_upload"):
                st.session_state.show_uploader = False
                st.rerun()
        
        # Process uploaded file
        if uploaded_file:
            # Initialize database connection
            table = init_db()
            
            # Process the uploaded file
            with st.spinner("Processing document..."):
                success, message, doc_id = process_uploaded_file(uploaded_file, table)
                
                if success:
                    # Add a system message about the document upload
                    upload_message = f"📄 Document uploaded: **{uploaded_file.name}**"
                    st.session_state.messages.append({"role": "system", "content": upload_message})
                    with messages_container.chat_message("system"):
                        st.markdown(upload_message)
                    
                    # Store in document_id for reference
                    st.session_state.last_uploaded_doc_id = doc_id
                    st.session_state.show_uploader = False  # Hide uploader after successful upload
                else:
                    st.error(message)
    
    # Input for user question
    if prompt := st.chat_input("Ask a question or drag & drop a document here..."):
        thinking_placeholder = st.empty()
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with messages_container.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with messages_container.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            # Connect to database
            table = init_db()
            
            # Placeholder for context
            context = ""
            
            # Generate response based on document database and hybrid search
            thinking_placeholder.markdown("*Thinking...*")
            
            # Use hybrid search to get context
            search_results = []
            search_metadata = {}
            if table:
                # Perform vector search with hybrid ranking
                search_results, search_metadata = hybrid_search(prompt, table, num_results=10)
                
                # Format search results as context
                if not search_results.empty:
                    context += "DOCUMENT SEARCH RESULTS:\n\n"
                    for i, (_, result) in enumerate(search_results.iterrows()):
                        # Extract metadata if available
                        metadata = {}
                        try:
                            metadata = json.loads(result["metadata"])
                        except:
                            pass
                        
                        # Format search result
                        source = result["source"] if "source" in result else "Unknown source"
                        text = result["text"] if "text" in result else "No text available"
                        
                        # Handle score - LanceDB doesn't always return "score" but might have "_distance"
                        # If neither is available, provide a default
                        score = 0.0
                        if "score" in result:
                            score_val = result["score"]
                            score = float(score_val)
                        elif "_distance" in result:
                            # Lower distance means higher similarity, so convert to a score
                            # Assuming distances are normalized between 0-1
                            distance_val = result["_distance"]
                            distance = float(distance_val)
                            score = 1.0 - distance
                        
                        context += f"{i+1}. {source} (Score: {score:.2f}):\n{text}\n\n"
                    
                    # Track if we found good document matches
                    has_good_document_matches = any(float(result.get("score", 0)) > 0.7 if "score" in result else 
                                                  (1.0 - float(result.get("_distance", 1.0))) > 0.7 if "_distance" in result else False 
                                                  for _, result in search_results.iterrows())
                else:
                    context += "No relevant documents found.\n\n"
                    has_good_document_matches = False
            
            # Try to perform internet search if API key is available
            tavily_api_key = get_api_key("TAVILY_API_KEY")
            
            # Log search status to debug
            internet_search_enabled = st.session_state.get("use_internet", False)
            print(f"DEBUG: Internet search enabled: {internet_search_enabled}")
            print(f"DEBUG: Tavily API key present: {tavily_api_key is not None}")
            
            # Search internet when API key is available AND either:
            # 1. User has explicitly enabled web search OR
            # 2. No good document matches were found
            should_search_internet = (
                tavily_api_key is not None and 
                (internet_search_enabled or search_results.empty or not has_good_document_matches)
            )
            
            print(f"DEBUG: Should search internet: {should_search_internet}")
            
            if should_search_internet:
                thinking_placeholder.markdown("*Thinking...*")
                try:
                    # Call the perform_internet_search function from search.py which uses the Tavily API
                    print(f"DEBUG: Executing internet search for query: {prompt[:50]}...")
                    internet_results = perform_internet_search(prompt)
                    
                    # Print debug info about results
                    print(f"DEBUG: Internet search returned results type: {type(internet_results)}")
                    print(f"DEBUG: Internet search returned results length: {len(internet_results) if isinstance(internet_results, list) else 'N/A'}")
                    
                    # If we got valid results back, add them to the context
                    if isinstance(internet_results, list) and internet_results:
                        print(f"DEBUG: Found {len(internet_results)} internet results")
                        context += "INTERNET SEARCH RESULTS:\n\n"
                        for i, result in enumerate(internet_results):
                            title = result.get("title", "No title")
                            content = result.get("content", "No content")
                            url = result.get("url", "")
                            
                            context += f"{i+1}. [{title}]({url}):\n{content}\n\n"
                    elif isinstance(internet_results, str):
                        # If we got a string back, it's likely an error message or formatted results
                        if "Error" in internet_results:
                            print(f"DEBUG: Internet search error: {internet_results}")
                        else:
                            context += f"{internet_results}\n\n"
                    else:
                        print("DEBUG: No internet results found")
                        context += "Internet search performed but no relevant results found.\n\n"
                except Exception as e:
                    print(f"DEBUG: Error during internet search: {str(e)}")
                    context += f"Error during internet search: {str(e)}\n\n"
            else:
                # Only show this message if internet search toggle is on but still not searching
                if internet_search_enabled and tavily_api_key is None:
                    context += "Note: Internet search is not available (Tavily API key not configured).\n\n"
            
            # Determine if context should be shown
            show_context = st.session_state.get("show_context", False)
            if show_context:
                with st.expander("Retrieved Context", expanded=True):
                    st.markdown(context)
            
            # Show generating state
            thinking_placeholder.markdown("*Generating response...*")
            
            # Prepare the message for the API
            messages = [
                {"role": "system", "content": f"""You are a helpful multilingual document assistant specializing in extracting insights and answering questions based on provided context.

Current user language: {st.session_state.user_language}

IMPORTANT: You must respond in the user's language specified above ({st.session_state.user_language}). 
For example:
- If language is 'en', respond in English
- If language is 'fr', respond in French
- Always match your response language to {st.session_state.user_language}

Response Guidelines:
Use the Context Wisely

If INTERNET SEARCH RESULTS are available, use them to confidently answer current or real-time factual questions. These are up-to-date and reliable.
For document-based questions, prioritize content from the DOCUMENT SEARCH RESULTS section.

CRITICAL INSTRUCTIONS FOR RESPONSES:
- DO NOT mention or cite sources in your response
- DO NOT refer to document names or file names
- DO NOT include URL references
- DO NOT include any phrases like "According to the documents", "Based on the search results", "The provided internet search results confirm", etc.
- DO NOT include citation numbers or reference markers
- DO NOT acknowledge where information came from
- DO NOT start answers with "Based on the information provided..."
- DO NOT refer to "context", "search results", "documents", or "internet results" in any way
- SIMPLY ANSWER THE QUESTION DIRECTLY as if you inherently know the information
- JUST PROVIDE THE ANSWER - nothing about where it came from

EXAMPLES OF FORBIDDEN PHRASES:
- "The provided internet search results confirm..."
- "According to the information..."
- "Based on the documents..."
- "The search results indicate..."
- "From the information available..."

Instead, just answer directly without any reference to sources.

Formatting and Content Instructions

Use Markdown for clear formatting and readability.
Do not start responses with phrases like "Answer:" or "Here's what I found:" — simply begin with the relevant information.

Provide sufficient context in your answers:
- Include relevant background information and explanations
- Explain concepts thoroughly but avoid unnecessary verbosity
- Include specific details that demonstrate depth of knowledge
- Balance being thorough with being concise (aim for 1-3 paragraphs for most answers)
- For complex topics, provide examples where helpful

Handle Special Content

If the context includes tables, describe and discuss their contents clearly.
Critical Note on Internet Search:
When INTERNET SEARCH RESULTS are included in the context, you must not say that you lack access to real-time information or cannot browse the web.
Instead, treat the internet search results as real-time, up-to-date information that you can and should use confidently.

Current date: {datetime.now().strftime('%Y-%m-%d')}"""},
            ]
            
            # Add the chat history
            for message in st.session_state.messages[:-1]:  # Exclude the latest user message
                messages.append({"role": message["role"], "content": message["content"]})
            
            # Add the retrieved context for the current question
            messages.append(
                {"role": "user", "content": f"Question: {prompt}\n\nContext:\n{context}"}
            )
            
            # Call the OpenAI API to get a streaming response
            response = openai.chat.completions.create(
                model=st.session_state.openai_model,
                messages=messages,
                temperature=0.7,
                stream=True,
            )
            
            # Clear the thinking indicator
            thinking_placeholder.empty()
            
            # Set up a placeholder for the streaming response
            response_placeholder = st.empty()
            
            # Process the streaming response
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content_delta = chunk.choices[0].delta.content
                    full_response += content_delta
                    # Update the response placeholder with the current full response plus cursor
                    response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
            # Final update without cursor
            response_placeholder.markdown(full_response, unsafe_allow_html=True)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Run the main function when script is executed
if __name__ == "__main__":


    main()
