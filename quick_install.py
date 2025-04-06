#!/usr/bin/env python3
"""
Quick installation script for DocLing dependencies
This script checks which dependencies are missing and installs them
"""

import sys
import subprocess
import importlib.util

def is_package_installed(package_name):
    """Check if a package is installed"""
    return importlib.util.find_spec(package_name) is not None

def install_package(package_name, extra_args=""):
    """Install a package using pip"""
    print(f"Installing {package_name}...")
    cmd = f"{sys.executable} -m pip install {package_name} {extra_args}"
    subprocess.check_call(cmd, shell=True)

def main():
    """Main function to check and install dependencies"""
    print("Checking for missing dependencies...")
    
    # Core visualization dependencies
    if not is_package_installed("plotly"):
        install_package("plotly>=5.18.0")
    
    # Table extraction dependencies
    if not is_package_installed("camelot"):
        print("Installing camelot and its dependencies...")
        install_package("opencv-python==4.8.0.74")
        install_package("ghostscript==0.7")
        install_package("camelot-py[cv]==0.11.0")
    
    # OCR dependencies
    if not is_package_installed("pytesseract"):
        install_package("pytesseract==0.3.10")
    
    # Formula rendering dependencies
    if not is_package_installed("sympy"):
        install_package("sympy==1.12")
    
    # Document processing dependencies
    if not is_package_installed("docx2txt"):
        install_package("docx2txt==0.8")
    
    if not is_package_installed("pdfplumber"):
        install_package("pdfplumber==0.9.0")
        
    # Web search dependencies
    if not is_package_installed("tavily"):
        install_package("tavily-python==0.2.7")
    
    print("\nInstallation complete! You can now run:")
    print("streamlit run chat.py")

if __name__ == "__main__":
    main()
