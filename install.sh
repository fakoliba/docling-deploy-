#!/bin/bash
# DocLing Application Installation Script

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    source venv/bin/activate
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install Homebrew first: https://brew.sh/"
        exit 1
    fi
    
    # Install system dependencies with Homebrew
    echo "Installing system dependencies..."
    brew install ghostscript tesseract
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    source venv/bin/activate
    
    # Install system dependencies with apt-get
    echo "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y ghostscript python3-tk tesseract-ocr
else
    # Windows or other
    echo "For Windows:"
    echo "1. Activate your virtual environment with: venv\\Scripts\\activate"
    echo "2. Install Ghostscript: https://ghostscript.com/releases/gsdnld.html"
    echo "3. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki"
    echo "4. Continue with: pip install -r requirements.txt"
    exit 0
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python packages one by one to better handle errors
echo "Installing core dependencies..."
pip install streamlit openai python-dotenv langchain uuid datetime numpy pandas

echo "Installing document processing libraries..."
pip install pypdf PyPDF2 docx2txt pdfplumber python-docx

echo "Installing table extraction dependencies..."
pip install "camelot-py[cv]" ghostscript opencv-python

echo "Installing OCR dependencies..."
pip install pytesseract Pillow

echo "Installing visualization libraries..."
pip install plotly matplotlib sympy

echo "Installing vector search libraries..."
pip install lancedb langchain-text-splitters pyarrow langchain_openai langchain_community tiktoken

echo "Installing language detection and web search..."
pip install langdetect tavily-python

echo ""
echo "Installation complete! You can run the application with:"
echo "streamlit run chat.py"
