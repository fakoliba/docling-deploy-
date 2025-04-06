"""
PDF Annotation System for Document Q&A Application

This module provides functionality for:
1. Adding annotations to PDFs
2. Storing and retrieving annotations
3. Rendering annotated PDFs in the interface
"""

import os
import json
import uuid
import base64
import logging
import streamlit as st
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Try to import PyMuPDF, but provide graceful degradation if it's not available
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    st.warning("PyMuPDF not installed. PDF annotation functionality will not be available. Install with 'pip install PyMuPDF'")

from cache import cache_document_annotations, get_cached_annotations, get_cache_key

# Setup logging
logger = logging.getLogger(__name__)

# Constants
ANNOTATION_TYPES = ["highlight", "comment", "underline", "strikeout"]
ANNOTATION_COLORS = {
    "yellow": (1, 0.8, 0),
    "green": (0, 0.8, 0),
    "blue": (0, 0.6, 1),
    "red": (1, 0, 0),
    "purple": (0.6, 0, 0.6)
}

class PDFAnnotator:
    def __init__(self, pdf_path: str):
        """
        Initialize the PDF annotator
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = pdf_path
        if PYMUPDF_AVAILABLE:
            self.document = fitz.open(pdf_path)
        else:
            self.document = None
        self.annotations = get_cached_annotations(pdf_path) or []
        
    def add_annotation(self, page_number: int, annotation_type: str, 
                       rect: Tuple[float, float, float, float], text: str, 
                       color: str = "yellow", metadata: Dict = None):
        """
        Add an annotation to the PDF
        
        Args:
            page_number: Page number (0-indexed)
            annotation_type: Type of annotation (highlight, comment, etc.)
            rect: Rectangle coordinates (x0, y0, x1, y1)
            text: Text content of annotation
            color: Color name from ANNOTATION_COLORS
            metadata: Additional metadata for the annotation
        
        Returns:
            Annotation ID
        """
        if not PYMUPDF_AVAILABLE:
            raise Exception("PyMuPDF not installed. PDF annotation functionality will not be available. Install with 'pip install PyMuPDF'")
        
        if page_number < 0 or page_number >= len(self.document):
            raise ValueError(f"Invalid page number: {page_number}")
        
        if annotation_type not in ANNOTATION_TYPES:
            raise ValueError(f"Invalid annotation type: {annotation_type}")
        
        if color not in ANNOTATION_COLORS:
            color = "yellow"  # Default color
            
        annotation_id = str(uuid.uuid4())
        
        # Create annotation object for storage
        annotation = {
            "id": annotation_id,
            "page": page_number,
            "type": annotation_type,
            "rect": rect,
            "text": text,
            "color": color,
            "created": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add to our annotations list
        self.annotations.append(annotation)
        
        # Save to cache
        cache_document_annotations(self.pdf_path, self.annotations)
        
        return annotation_id
    
    def get_annotations(self, page_number: Optional[int] = None) -> List[Dict]:
        """
        Get annotations for the PDF
        
        Args:
            page_number: If provided, only get annotations for this page
            
        Returns:
            List of annotation dictionaries
        """
        if not PYMUPDF_AVAILABLE:
            raise Exception("PyMuPDF not installed. PDF annotation functionality will not be available. Install with 'pip install PyMuPDF'")
        
        if page_number is not None:
            return [a for a in self.annotations if a["page"] == page_number]
        return self.annotations
    
    def delete_annotation(self, annotation_id: str) -> bool:
        """
        Delete an annotation
        
        Args:
            annotation_id: ID of annotation to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not PYMUPDF_AVAILABLE:
            raise Exception("PyMuPDF not installed. PDF annotation functionality will not be available. Install with 'pip install PyMuPDF'")
        
        for i, annotation in enumerate(self.annotations):
            if annotation["id"] == annotation_id:
                del self.annotations[i]
                cache_document_annotations(self.pdf_path, self.annotations)
                return True
        return False
    
    def render_pdf_with_annotations(self) -> bytes:
        """
        Render the PDF with annotations
        
        Returns:
            PDF bytes with annotations applied
        """
        if not PYMUPDF_AVAILABLE:
            raise Exception("PyMuPDF not installed. PDF annotation functionality will not be available. Install with 'pip install PyMuPDF'")
        
        # Create a copy of the document to add annotations to
        annotated_pdf = fitz.open()
        annotated_pdf.insert_pdf(self.document)
        
        # Add annotations to the PDF
        for annotation in self.annotations:
            page_number = annotation["page"]
            if page_number < 0 or page_number >= len(annotated_pdf):
                continue
                
            page = annotated_pdf[page_number]
            rect = fitz.Rect(*annotation["rect"])
            color = ANNOTATION_COLORS.get(annotation["color"], ANNOTATION_COLORS["yellow"])
            
            if annotation["type"] == "highlight":
                highlight = page.add_highlight_annot(rect)
                highlight.set_colors(stroke=color)
                highlight.update()
                
            elif annotation["type"] == "comment":
                comment = page.add_text_annot(rect.top_right, annotation["text"])
                comment.set_colors(stroke=color)
                comment.update()
                
            elif annotation["type"] == "underline":
                underline = page.add_underline_annot(rect)
                underline.set_colors(stroke=color)
                underline.update()
                
            elif annotation["type"] == "strikeout":
                strikeout = page.add_strikeout_annot(rect)
                strikeout.set_colors(stroke=color)
                strikeout.update()
        
        # Save the annotated PDF to memory
        pdf_bytes = annotated_pdf.tobytes()
        annotated_pdf.close()
        
        return pdf_bytes
    
    def close(self):
        """Close the document"""
        if self.document:
            self.document.close()


def render_pdf_annotation_ui(pdf_path: str, pdf_bytes: bytes):
    """
    Render the PDF annotation UI in Streamlit
    
    Args:
        pdf_path: Path to the PDF file
        pdf_bytes: Bytes of the PDF file
    """
    try:
        # Create tabs for viewing and annotation
        view_tab, annotate_tab = st.tabs(["View PDF", "Add Annotations"])
        
        # Initialize the annotator
        annotator = PDFAnnotator(pdf_path)
        
        # Get existing annotations
        annotations = annotator.get_annotations()
        
        with view_tab:
            # Display PDF with annotations
            if PYMUPDF_AVAILABLE:
                annotated_pdf = annotator.render_pdf_with_annotations()
                base64_pdf = base64.b64encode(annotated_pdf).decode('utf-8')
                
                # Display PDF viewer
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                
                # Display existing annotations list
                if annotations:
                    st.subheader("Existing Annotations")
                    for i, annot in enumerate(annotations):
                        with st.expander(f"Annotation {i+1}: {annot['type']} on page {annot['page']+1}"):
                            st.write(f"**Text:** {annot['text']}")
                            st.write(f"**Type:** {annot['type']}")
                            st.write(f"**Color:** {annot['color']}")
                            st.write(f"**Created:** {annot['created']}")
                            
                            if st.button(f"Delete Annotation", key=f"delete_{annot['id']}"):
                                if annotator.delete_annotation(annot['id']):
                                    st.success("Annotation deleted!")
                                    st.experimental_rerun()
            else:
                st.error("PyMuPDF not installed. PDF annotation functionality will not be available. Install with 'pip install PyMuPDF'")
        
        with annotate_tab:
            # Form for adding new annotations
            with st.form("annotation_form"):
                st.subheader("Add New Annotation")
                
                # Get PDF document info for valid page range
                if PYMUPDF_AVAILABLE:
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    page_count = len(doc)
                    doc.close()
                else:
                    page_count = 0
                
                # Form fields
                page_number = st.number_input("Page Number", min_value=1, max_value=page_count, step=1) - 1  # Convert to 0-indexed
                annotation_type = st.selectbox("Annotation Type", ANNOTATION_TYPES)
                annotation_color = st.selectbox("Color", list(ANNOTATION_COLORS.keys()))
                
                # Coordinates (simplified for user input)
                st.write("Annotation Position (values from 0 to 1, where 0 is top/left and 1 is bottom/right)")
                col1, col2 = st.columns(2)
                with col1:
                    x0 = st.slider("Left (x0)", 0.0, 1.0, 0.1, 0.01)
                    y0 = st.slider("Top (y0)", 0.0, 1.0, 0.1, 0.01)
                with col2:
                    x1 = st.slider("Right (x1)", 0.0, 1.0, 0.9, 0.01)
                    y1 = st.slider("Bottom (y1)", 0.0, 1.0, 0.2, 0.01)
                
                # Ensure x0 < x1 and y0 < y1
                x0, x1 = min(x0, x1), max(x0, x1)
                y0, y1 = min(y0, y1), max(y0, y1)
                
                # Document dimensions (will be scaled for actual annotations)
                if PYMUPDF_AVAILABLE:
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    page = doc[page_number]
                    width, height = page.rect.width, page.rect.height
                    doc.close()
                else:
                    width, height = 0, 0
                
                # Scale coordinates to actual page dimensions
                rect = (x0 * width, y0 * height, x1 * width, y1 * height)
                
                annotation_text = st.text_area("Annotation Text", "")
                
                submitted = st.form_submit_button("Add Annotation")
                
                if submitted and annotation_text:
                    try:
                        if PYMUPDF_AVAILABLE:
                            annotation_id = annotator.add_annotation(
                                page_number=page_number,
                                annotation_type=annotation_type,
                                rect=rect,
                                text=annotation_text,
                                color=annotation_color
                            )
                            st.success(f"Annotation added successfully!")
                            # Force a rerun to update the UI
                            st.experimental_rerun()
                        else:
                            st.error("PyMuPDF not installed. PDF annotation functionality will not be available. Install with 'pip install PyMuPDF'")
                    except Exception as e:
                        st.error(f"Error adding annotation: {str(e)}")
        
        # Close the annotator
        annotator.close()
        
    except Exception as e:
        st.error(f"Error in PDF annotation UI: {str(e)}")
        logger.error(f"Error in PDF annotation UI: {str(e)}")
