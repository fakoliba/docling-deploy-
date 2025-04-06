# DocLing - Enhanced Document Q&A Application

An intelligent document Q&A application that leverages AI to extract insights from various document types. This application combines document processing capabilities with vector database storage, advanced table extraction, mathematical formula rendering, and OCR to create a comprehensive Retrieval Augmented Generation (RAG) system.

## Key Features

- **Document Processing**: Upload and process PDF, DOCX, and image files
- **Table Extraction**: Extract tables from PDF documents with accuracy scores
- **Table Visualization**: Interactive visualization of extracted tables with Plotly
- **Mathematical Formula Support**: Parse and render LaTeX-style mathematical formulas
- **OCR Integration**: Extract text from images using Tesseract OCR
- **Multilingual Support**: Interface available in multiple languages
- **Vector Search**: Find information based on meaning using semantic search
- **Internet Search Integration**: Enhance answers with web information (optional)

## Installation

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

Quick setup:

```bash
# Make the install script executable
chmod +x install.sh

# Run the install script
./install.sh
```

### Checking Dependencies

If you're having issues with missing dependencies, you can use the dependency checker:

```bash
# Install colorama for better formatting (optional)
pip install colorama

# Run the dependency checker
python check_dependencies.py
```

Or use the quick installer script to automatically install missing dependencies:

```bash
python quick_install.py
```

### 1. Clone and Set Up

```bash
# Clone the repository
git clone https://github.com/yourusername/docling.git
cd docling

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

### 2. System Dependencies

For full functionality, install the following system dependencies:

**Table Extraction (Camelot):**
- **macOS**: `brew install ghostscript`
- **Ubuntu/Debian**: `sudo apt-get install ghostscript python3-tk`
- **Windows**: Download from [Ghostscript Website](https://ghostscript.com/releases/gsdnld.html)

**OCR Support (Pytesseract):**
- **macOS**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
- **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

### 3. Environment Setup

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key  # Optional, for web search
```

**Security Note:** Keep your `.env` file private and add it to `.gitignore`.

## Usage

1. Start the application:
   ```bash
   streamlit run chat.py
   ```

2. **Upload Documents**: Use the sidebar to upload PDF, DOCX, or image files
3. **Ask Questions**: Type your questions in the chat interface
4. **View Tables**: Tables extracted from documents are displayed interactively
5. **Test Formula Rendering**: Use the formula input field to test mathematical formula display
6. **Configure Settings**: Adjust search settings and interface language in the sidebar

## Features in Detail

### Table Extraction and Visualization

The application extracts tables from PDF documents using Camelot, providing:
- Table structure preservation
- Accuracy scores for extraction quality
- Interactive table visualization with Plotly
- CSV export option for extracted tables

### Mathematical Formula Support

Mathematical formulas are rendered using MathJax:
- Support for LaTeX syntax with `$$` delimiters
- Formula parsing and rendering in chat responses
- Formula testing input field for experimentation

### OCR for Images

Image text extraction using Tesseract OCR:
- Convert image text to searchable content
- Support for various image formats
- Integration with the document database

## Troubleshooting

- **Missing Dependencies**: If you encounter module import errors, check the system dependencies section
- **OCR Issues**: Ensure Tesseract is properly installed for image processing
- **Table Extraction Problems**: Verify Ghostscript installation for Camelot functionality

## License

This project is licensed under the MIT License.

## Acknowledgments

- OpenAI for the GPT API
- Streamlit for the web framework
- Camelot for table extraction capabilities
- Tesseract for OCR functionality
- All the open-source libraries that make this project possible