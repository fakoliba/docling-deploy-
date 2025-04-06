# Installation Guide for DocLing

This guide will help you install all necessary dependencies for the DocLing application, including system dependencies and Python packages.

## Quick Installation (macOS)

For a quick installation on macOS, run:

```bash
# Make the install script executable
chmod +x install.sh

# Run the install script
./install.sh
```

## Manual Installation Steps

### 1. System Dependencies

#### macOS

Install Ghostscript (for table extraction) and Tesseract (for OCR):

```bash
brew install ghostscript tesseract
```

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install ghostscript python3-tk tesseract-ocr
```

#### Windows

1. Install Ghostscript from [Ghostscript Website](https://ghostscript.com/releases/gsdnld.html)
2. Install Tesseract OCR from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
3. Make sure both are added to your system PATH

### 2. Python Environment

It's recommended to use a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Python Dependencies

Install the required Python packages:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all packages
pip install -r requirements.txt
```

If you encounter issues with camelot-py, try installing it separately:

```bash
pip install "camelot-py[cv]" ghostscript opencv-python
```

### 4. Environment Variables

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key  # Optional, for web search
```

## Troubleshooting

### camelot-py Installation Issues

If you encounter issues installing camelot-py:

1. Make sure Ghostscript is installed correctly
2. Try installing with specific versions:
   ```bash
   pip install "camelot-py[cv]==0.11.0" opencv-python==4.8.0.74 ghostscript==0.7
   ```

### OCR (pytesseract) Issues

If OCR functionality isn't working:

1. Verify Tesseract is installed and in your PATH
2. Try reinstalling pytesseract:
   ```bash
   pip install pytesseract --force-reinstall
   ```

### Other Issues

For other installation issues, please refer to the documentation of the specific package:

- [camelot-py documentation](https://camelot-py.readthedocs.io/en/master/user/install.html)
- [pytesseract documentation](https://github.com/madmaze/pytesseract)

## Running the Application

After installation is complete, run the application:

```bash
streamlit run chat.py
```
