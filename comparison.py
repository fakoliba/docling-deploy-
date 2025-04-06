"""
Document Comparison Module for Document Q&A Application

This module provides functionality for:
1. Comparing multiple documents for similarities and differences
2. Visualizing document comparisons
3. Generating comparison reports
"""

import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
import difflib
import logging
import plotly.express as px
import plotly.graph_objects as go

# Try to import sklearn, but provide graceful degradation if it's not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn not installed. Document comparison functionality will be limited. Install with 'pip install scikit-learn'")

from cache import get_cached_document
import search

# Setup logging
logger = logging.getLogger(__name__)

class DocumentComparer:
    def __init__(self):
        """Initialize the document comparison system"""
        self.documents = {}
        self.embeddings = {}
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    def add_document(self, doc_id: str, text: str, chunks: List[str], metadata: Dict[str, Any] = None):
        """
        Add a document to the comparison system
        
        Args:
            doc_id: Document identifier
            text: Full document text
            chunks: Document chunks
            metadata: Document metadata
        """
        self.documents[doc_id] = {
            "text": text,
            "chunks": chunks,
            "metadata": metadata or {}
        }
        logger.info(f"Added document {doc_id} to comparer with {len(chunks)} chunks")
    
    def load_document_from_cache(self, doc_id: str) -> bool:
        """
        Load a document from cache
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if document was loaded, False otherwise
        """
        cached_doc = get_cached_document(doc_id)
        if cached_doc:
            chunks, metadata = cached_doc
            # Reconstruct full text from chunks
            text = " ".join(chunks)
            self.add_document(doc_id, text, chunks, metadata)
            return True
        return False
    
    def compute_embeddings(self, doc_ids: Optional[List[str]] = None):
        """
        Compute embeddings for documents
        
        Args:
            doc_ids: List of document IDs to compute embeddings for, or None for all
        """
        if doc_ids is None:
            doc_ids = list(self.documents.keys())
        
        for doc_id in doc_ids:
            if doc_id not in self.documents:
                continue
                
            # Compute embedding for full document text
            text = self.documents[doc_id]["text"]
            try:
                embedding = search.get_embedding(text)
                self.embeddings[doc_id] = embedding
                logger.info(f"Computed embedding for document {doc_id}")
            except Exception as e:
                logger.error(f"Error computing embedding for document {doc_id}: {str(e)}")
    
    def compare_documents_semantic(self, doc_id1: str, doc_id2: str) -> float:
        """
        Compare two documents semantically using embeddings
        
        Args:
            doc_id1: First document ID
            doc_id2: Second document ID
            
        Returns:
            Similarity score between 0 and 1
        """
        # Ensure both documents have embeddings
        if doc_id1 not in self.embeddings or doc_id2 not in self.embeddings:
            self.compute_embeddings([doc_id1, doc_id2])
        
        if doc_id1 not in self.embeddings or doc_id2 not in self.embeddings:
            return 0.0
        
        # Compute cosine similarity
        emb1 = np.array(self.embeddings[doc_id1]).reshape(1, -1)
        emb2 = np.array(self.embeddings[doc_id2]).reshape(1, -1)
        similarity = cosine_similarity(emb1, emb2)[0][0]
        
        return float(similarity)
    
    def compare_documents_tfidf(self, doc_id1: str, doc_id2: str) -> Dict[str, Any]:
        """
        Compare two documents using TF-IDF similarity and common terms
        
        Args:
            doc_id1: First document ID
            doc_id2: Second document ID
            
        Returns:
            Dictionary with comparison results
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not installed"}
        
        if doc_id1 not in self.documents or doc_id2 not in self.documents:
            return {"error": "One or both documents not found"}
        
        text1 = self.documents[doc_id1]["text"]
        text2 = self.documents[doc_id2]["text"]
        
        # TF-IDF similarity
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception as e:
            logger.error(f"Error computing TF-IDF similarity: {str(e)}")
            similarity = 0.0
        
        # Extract top terms from each document
        try:
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            dense = tfidf_matrix.todense()
            
            doc1_terms = {}
            doc2_terms = {}
            
            for term_idx, term in enumerate(feature_names):
                doc1_terms[term] = dense[0, term_idx]
                doc2_terms[term] = dense[1, term_idx]
            
            doc1_top_terms = sorted(doc1_terms.items(), key=lambda x: x[1], reverse=True)[:20]
            doc2_top_terms = sorted(doc2_terms.items(), key=lambda x: x[1], reverse=True)[:20]
            
            # Find common top terms
            doc1_top_terms_set = {term for term, _ in doc1_top_terms}
            doc2_top_terms_set = {term for term, _ in doc2_top_terms}
            common_terms = doc1_top_terms_set.intersection(doc2_top_terms_set)
        except Exception as e:
            logger.error(f"Error extracting top terms: {str(e)}")
            doc1_top_terms = []
            doc2_top_terms = []
            common_terms = set()
        
        return {
            "similarity": float(similarity),
            "doc1_top_terms": doc1_top_terms,
            "doc2_top_terms": doc2_top_terms,
            "common_terms": list(common_terms)
        }
    
    def generate_diff(self, doc_id1: str, doc_id2: str) -> List[Dict[str, Any]]:
        """
        Generate a diff between two documents
        
        Args:
            doc_id1: First document ID
            doc_id2: Second document ID
            
        Returns:
            List of diff chunks with metadata
        """
        if doc_id1 not in self.documents or doc_id2 not in self.documents:
            return []
        
        text1 = self.documents[doc_id1]["text"]
        text2 = self.documents[doc_id2]["text"]
        
        # Split into lines for diffing
        lines1 = text1.splitlines()
        lines2 = text2.splitlines()
        
        # Generate diff
        diff = difflib.ndiff(lines1, lines2)
        
        # Format diff results
        result = []
        for line in diff:
            if line.startswith('+ '):
                result.append({
                    "type": "added",
                    "text": line[2:],
                    "doc_id": doc_id2
                })
            elif line.startswith('- '):
                result.append({
                    "type": "removed",
                    "text": line[2:],
                    "doc_id": doc_id1
                })
            elif line.startswith('? '):
                # Skip diff control lines
                continue
            else:
                result.append({
                    "type": "unchanged",
                    "text": line[2:] if line.startswith('  ') else line
                })
        
        return result
    
    def generate_comparison_report(self, doc_ids: List[str]) -> Dict[str, Any]:
        """
        Generate a comprehensive comparison report for multiple documents
        
        Args:
            doc_ids: List of document IDs to compare
            
        Returns:
            Comparison report dictionary
        """
        if len(doc_ids) < 2:
            return {"error": "Need at least two documents to compare"}
        
        # Ensure we have embeddings for all documents
        self.compute_embeddings(doc_ids)
        
        # Compute pairwise similarities
        similarities = {}
        for i, doc_id1 in enumerate(doc_ids):
            for j, doc_id2 in enumerate(doc_ids):
                if i >= j:  # Skip self-comparisons and duplicates
                    continue
                    
                semantic_sim = self.compare_documents_semantic(doc_id1, doc_id2)
                tfidf_results = self.compare_documents_tfidf(doc_id1, doc_id2)
                
                pair_key = f"{doc_id1}_vs_{doc_id2}"
                similarities[pair_key] = {
                    "doc1": doc_id1,
                    "doc2": doc_id2,
                    "semantic_similarity": semantic_sim,
                    "tfidf_similarity": tfidf_results.get("similarity", 0.0),
                    "common_terms": tfidf_results.get("common_terms", [])
                }
        
        # Document metadata
        doc_metadata = {}
        for doc_id in doc_ids:
            if doc_id in self.documents:
                doc_metadata[doc_id] = {
                    "chunk_count": len(self.documents[doc_id]["chunks"]),
                    "text_length": len(self.documents[doc_id]["text"]),
                    "metadata": self.documents[doc_id].get("metadata", {})
                }
        
        return {
            "document_count": len(doc_ids),
            "document_ids": doc_ids,
            "document_metadata": doc_metadata,
            "pairwise_similarities": similarities
        }
    
    def visualize_similarity_matrix(self, doc_ids: List[str]) -> go.Figure:
        """
        Create a similarity matrix visualization
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            Plotly figure
        """
        # Generate document labels
        doc_labels = {}
        for i, doc_id in enumerate(doc_ids):
            if doc_id in self.documents and "metadata" in self.documents[doc_id]:
                filename = self.documents[doc_id]["metadata"].get("filename", f"Doc {i+1}")
                doc_labels[doc_id] = filename
            else:
                doc_labels[doc_id] = f"Doc {i+1}"
        
        # Create similarity matrix
        n_docs = len(doc_ids)
        similarity_matrix = np.zeros((n_docs, n_docs))
        
        for i, doc_id1 in enumerate(doc_ids):
            for j, doc_id2 in enumerate(doc_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # Self similarity
                elif i < j:
                    similarity = self.compare_documents_semantic(doc_id1, doc_id2)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity  # Matrix is symmetric
        
        # Create heatmap
        labels = [doc_labels[doc_id] for doc_id in doc_ids]
        fig = px.imshow(
            similarity_matrix,
            x=labels,
            y=labels,
            color_continuous_scale="Viridis",
            title="Document Similarity Matrix",
            labels=dict(x="Document", y="Document", color="Similarity")
        )
        
        return fig


def render_document_comparison_ui():
    """Render the document comparison UI in Streamlit"""
    st.header("Document Comparison")
    
    # Get available documents from the session state
    available_docs = []
    if "uploaded_docs" in st.session_state:
        available_docs = list(st.session_state.uploaded_docs.keys())
    
    if not available_docs:
        st.warning("You need to upload at least two documents to use the comparison feature.")
        return
    
    if len(available_docs) < 2:
        st.warning("You need one more document to compare. Please upload another document.")
        return
    
    # Select documents to compare
    selected_docs = st.multiselect(
        "Select documents to compare",
        available_docs,
        default=available_docs[:2] if len(available_docs) >= 2 else []
    )
    
    if len(selected_docs) < 2:
        st.info("Please select at least two documents to compare")
        return
    
    # Initialize document comparer
    comparer = DocumentComparer()
    
    # Load selected documents
    for doc_id in selected_docs:
        doc_info = st.session_state.uploaded_docs[doc_id]
        
        # Try to load from cache first
        if not comparer.load_document_from_cache(doc_id):
            # If not in cache, add directly from session state
            chunks = doc_info.get("chunks", [])
            text = " ".join(chunks)  # Reconstruct full text
            metadata = doc_info.get("metadata", {})
            comparer.add_document(doc_id, text, chunks, metadata)
    
    # Compute embeddings for all documents
    with st.spinner("Computing document similarities..."):
        comparer.compute_embeddings()
    
    # Create tabs for different comparison views
    matrix_tab, report_tab, diff_tab = st.tabs(["Similarity Matrix", "Comparison Report", "Document Diff"])
    
    with matrix_tab:
        st.subheader("Document Similarity Matrix")
        
        # Generate similarity matrix visualization
        fig = comparer.visualize_similarity_matrix(selected_docs)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("The similarity matrix shows how similar the documents are to each other. " +
                "A value of 1.0 indicates identical documents, while 0.0 indicates completely different documents.")
    
    with report_tab:
        st.subheader("Comparison Report")
        
        # Generate comparison report
        report = comparer.generate_comparison_report(selected_docs)
        
        # Display document metadata
        st.write("### Document Information")
        doc_info = []
        for doc_id, metadata in report["document_metadata"].items():
            doc_name = metadata.get("metadata", {}).get("filename", doc_id)
            doc_info.append({
                "Document": doc_name,
                "Chunks": metadata["chunk_count"],
                "Length (chars)": metadata["text_length"]
            })
        
        st.dataframe(pd.DataFrame(doc_info))
        
        # Display pairwise similarities
        st.write("### Pairwise Similarities")
        similarities = []
        for pair_key, data in report["pairwise_similarities"].items():
            doc1_name = report["document_metadata"][data["doc1"]].get("metadata", {}).get("filename", data["doc1"])
            doc2_name = report["document_metadata"][data["doc2"]].get("metadata", {}).get("filename", data["doc2"])
            
            similarities.append({
                "Document 1": doc1_name,
                "Document 2": doc2_name,
                "Semantic Similarity": f"{data['semantic_similarity']:.2f}",
                "TF-IDF Similarity": f"{data['tfidf_similarity']:.2f}",
                "Common Key Terms": ", ".join(data["common_terms"][:5])
            })
        
        st.dataframe(pd.DataFrame(similarities))
    
    with diff_tab:
        st.subheader("Document Differences")
        
        if len(selected_docs) == 2:
            # For exactly two documents, show detailed diff
            doc_id1, doc_id2 = selected_docs
            doc1_name = comparer.documents[doc_id1].get("metadata", {}).get("filename", doc_id1)
            doc2_name = comparer.documents[doc_id2].get("metadata", {}).get("filename", doc_id2)
            
            st.write(f"Comparing: **{doc1_name}** and **{doc2_name}**")
            
            diff_results = comparer.generate_diff(doc_id1, doc_id2)
            
            # Format and display diff
            diff_html = []
            for chunk in diff_results:
                if chunk["type"] == "added":
                    diff_html.append(f'<div style="background-color: #e6ffe6; padding: 2px;">+ {chunk["text"]}</div>')
                elif chunk["type"] == "removed":
                    diff_html.append(f'<div style="background-color: #ffe6e6; padding: 2px;">- {chunk["text"]}</div>')
                else:
                    diff_html.append(f'<div style="padding: 2px;">  {chunk["text"]}</div>')
            
            st.markdown("".join(diff_html), unsafe_allow_html=True)
        else:
            st.info("Select exactly two documents to view detailed differences")
