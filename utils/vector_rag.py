import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import json
import pickle
import os
from datetime import datetime

class VectorRAG:
    """Basic Vector RAG implementation using FAISS for similarity search"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 chunk_size: int = 500, overlap: int = 50):
        """
        Initialize Vector RAG system
        
        Args:
            embedding_model: Name of the sentence transformer model
            chunk_size: Size of text chunks for embedding
            overlap: Overlap between chunks
        """
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Initialize sentence transformer
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            # Fallback to a simple model
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_dim = 384
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        self.metadata = []
        self.is_built = False
        
    def build_index(self, hr_data: Dict[str, Any]) -> None:
        """
        Build FAISS index from HR data
        
        Args:
            hr_data: Dictionary containing HR documents and data
        """
        print("Building vector index...")
        
        # Extract and process all text content
        all_texts = []
        all_metadata = []
        
        # Process employees
        if 'employees' in hr_data:
            for emp in hr_data['employees']:
                text = self._employee_to_text(emp)
                chunks = self._chunk_text(text)
                for chunk in chunks:
                    all_texts.append(chunk)
                    all_metadata.append({
                        'type': 'employee',
                        'id': emp['id'],
                        'name': emp['name'],
                        'department': emp['department'],
                        'source': 'employee_data'
                    })
        
        # Process policies
        if 'policies' in hr_data:
            for policy in hr_data['policies']:
                text = self._policy_to_text(policy)
                chunks = self._chunk_text(text)
                for chunk in chunks:
                    all_texts.append(chunk)
                    all_metadata.append({
                        'type': 'policy',
                        'id': policy['id'],
                        'title': policy['title'],
                        'department': policy.get('department', 'All'),
                        'priority': policy.get('priority', 'Medium'),
                        'source': 'policy_document'
                    })
        
        # Process documents
        if 'documents' in hr_data:
            for doc in hr_data['documents']:
                text = self._document_to_text(doc)
                chunks = self._chunk_text(text)
                for chunk in chunks:
                    all_texts.append(chunk)
                    all_metadata.append({
                        'type': 'document',
                        'id': doc['id'],
                        'title': doc['title'],
                        'doc_type': doc.get('doc_type', 'Document'),
                        'department': doc.get('department', 'All'),
                        'source': 'hr_document'
                    })
        
        if not all_texts:
            raise ValueError("No text content found in HR data")
        
        # Generate embeddings
        print(f"Generating embeddings for {len(all_texts)} text chunks...")
        embeddings = self.embedding_model.encode(all_texts, show_progress_bar=True)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents = all_texts
        self.metadata = all_metadata
        self.is_built = True
        
        print(f"Vector index built successfully with {len(all_texts)} documents")
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector index
        
        Args:
            query_text: Query string
            top_k: Number of top results to return
            
        Returns:
            List of search results with content, metadata, and scores
        """
        if not self.is_built:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):  # Valid index
                result = {
                    'rank': i + 1,
                    'content': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score),
                    'similarity': float(score)  # Since we're using normalized vectors
                }
                results.append(result)
        
        return results
    
    def get_index_size(self) -> int:
        """Get the number of documents in the index"""
        return len(self.documents) if self.is_built else 0
    
    def save_index(self, filepath: str) -> None:
        """Save the index and metadata to disk"""
        if not self.is_built:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save metadata and documents
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'embedding_model_name': self.embedding_model_name,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str) -> None:
        """Load the index and metadata from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load metadata and documents
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.embedding_model_name = data['embedding_model_name']
            self.chunk_size = data['chunk_size']
            self.overlap = data['overlap']
            self.embedding_dim = data['embedding_dim']
        
        # Reinitialize embedding model if needed
        if not hasattr(self, 'embedding_model'):
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        self.is_built = True
        print(f"Index loaded from {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        if not self.is_built:
            return {"status": "not_built"}
        
        # Count by type
        type_counts = {}
        dept_counts = {}
        
        for meta in self.metadata:
            doc_type = meta.get('type', 'unknown')
            department = meta.get('department', 'unknown')
            
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            dept_counts[department] = dept_counts.get(department, 0) + 1
        
        return {
            "status": "built",
            "total_documents": len(self.documents),
            "embedding_dimension": self.embedding_dim,
            "model_name": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "type_distribution": type_counts,
            "department_distribution": dept_counts
        }
    
    def _employee_to_text(self, employee: Dict[str, Any]) -> str:
        """Convert employee data to searchable text"""
        text_parts = [
            f"Employee: {employee.get('name', '')}",
            f"Department: {employee.get('department', '')}",
            f"Job Title: {employee.get('job_title', '')}",
            f"Email: {employee.get('email', '')}",
            f"Location: {employee.get('location', '')}",
            f"Employment Type: {employee.get('employment_type', '')}",
            f"Hire Date: {employee.get('hire_date', '')}",
            f"Performance Rating: {employee.get('performance_rating', '')}"
        ]
        
        if employee.get('salary'):
            text_parts.append(f"Salary: ${employee['salary']:,}")
        
        return " | ".join(text_parts)
    
    def _policy_to_text(self, policy: Dict[str, Any]) -> str:
        """Convert policy data to searchable text"""
        text_parts = [
            f"Policy: {policy.get('title', '')}",
            f"Type: {policy.get('type', '')}",
            f"Department: {policy.get('department', '')}",
            f"Priority: {policy.get('priority', '')}",
            f"Content: {policy.get('content', '')}"
        ]
        
        return " | ".join(text_parts)
    
    def _document_to_text(self, document: Dict[str, Any]) -> str:
        """Convert document data to searchable text"""
        text_parts = [
            f"Document: {document.get('title', '')}",
            f"Type: {document.get('doc_type', '')}",
            f"Department: {document.get('department', '')}",
            f"Content: {document.get('content', '')}"
        ]
        
        return " | ".join(text_parts)
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Find last space before the end
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.overlap
            if start >= len(text):
                break
        
        return chunks
    
    def semantic_search(self, query: str, top_k: int = 10, 
                       filter_type: Optional[str] = None,
                       filter_department: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search with optional filtering
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_type: Filter by document type
            filter_department: Filter by department
            
        Returns:
            Filtered and ranked search results
        """
        # Get initial results with higher top_k for filtering
        initial_results = self.query(query, top_k * 3)
        
        # Apply filters
        filtered_results = []
        for result in initial_results:
            metadata = result['metadata']
            
            # Type filter
            if filter_type and metadata.get('type') != filter_type:
                continue
            
            # Department filter
            if filter_department and metadata.get('department') != filter_department:
                continue
            
            filtered_results.append(result)
            
            # Stop when we have enough results
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results[:top_k]

