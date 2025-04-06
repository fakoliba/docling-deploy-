#!/usr/bin/env python3
"""
Dependency checker for DocLing application
Checks if all required dependencies are installed and provides installation instructions
"""

import importlib
import sys
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

def check_dependency(module_name, pip_package=None, system_dependency=None):
    """Check if a Python module is available"""
    if pip_package is None:
        pip_package = module_name
        
    try:
        importlib.import_module(module_name)
        print(f"{Fore.GREEN}✓ {module_name} is installed{Style.RESET_ALL}")
        return True
    except ImportError:
        print(f"{Fore.RED}✗ {module_name} is not installed{Style.RESET_ALL}")
        
        install_cmd = f"pip install {pip_package}"
        
        print(f"  {Fore.YELLOW}Install with: {install_cmd}{Style.RESET_ALL}")
        
        if system_dependency:
            print(f"  {Fore.YELLOW}Note: You may also need to install system dependency: {system_dependency}{Style.RESET_ALL}")
            
        return False

def main():
    """Check all dependencies for DocLing application"""
    print(f"{Fore.CYAN}=== DocLing Dependency Checker ==={Style.RESET_ALL}")
    print("Checking if all required dependencies are installed...\n")
    
    all_good = True
    
    # Core dependencies
    print(f"{Fore.CYAN}Core dependencies:{Style.RESET_ALL}")
    all_good &= check_dependency("streamlit")
    all_good &= check_dependency("openai")
    all_good &= check_dependency("dotenv", "python-dotenv")
    all_good &= check_dependency("pandas")
    all_good &= check_dependency("numpy")
    
    # Document processing dependencies
    print(f"\n{Fore.CYAN}Document processing dependencies:{Style.RESET_ALL}")
    all_good &= check_dependency("PyPDF2")
    all_good &= check_dependency("docx2txt")
    all_good &= check_dependency("pdfplumber")
    
    # Vector database dependencies
    print(f"\n{Fore.CYAN}Vector database dependencies:{Style.RESET_ALL}")
    all_good &= check_dependency("lancedb")
    all_good &= check_dependency("langchain")
    all_good &= check_dependency("pyarrow")
    
    # Advanced features dependencies
    print(f"\n{Fore.CYAN}Advanced features dependencies:{Style.RESET_ALL}")
    all_good &= check_dependency("plotly", "plotly>=5.18.0")
    all_good &= check_dependency("camelot", "camelot-py[cv]", "Ghostscript (brew install ghostscript)")
    all_good &= check_dependency("pytesseract", "pytesseract", "Tesseract OCR (brew install tesseract)")
    all_good &= check_dependency("sympy")
    all_good &= check_dependency("tavily", "tavily-python")
    
    print("\n" + "="*50)
    if all_good:
        print(f"{Fore.GREEN}All dependencies are installed! You're good to go.{Style.RESET_ALL}")
        print(f"Run the application with: {Fore.CYAN}streamlit run chat.py{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}Some dependencies are missing. Please install them using the commands above.{Style.RESET_ALL}")
        print(f"Alternatively, run: {Fore.CYAN}python quick_install.py{Style.RESET_ALL}")
        
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
