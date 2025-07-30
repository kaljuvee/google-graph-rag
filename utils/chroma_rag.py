import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import json
import uuid
from datetime import datetime
import os
import tempfile

class ChromaRAG:
    """ChromaDB-based RAG implementation with enhanced filtering and metadata search"""
    
    def __init__(self, collection_name: str = "hr_documents", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB RAG system
        
        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: Name of the embedding model
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Set up persistence directory
        if persist_directory is None:
            self.persist_directory = tempfile.mkdtemp()
        else:
            self.persist_directory = persist_directory
            os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            # Fallback to in-memory client
            self.client = chromadb.Client()
        
        self.collection = None
        self.is_built = False
        
    def build_collection(self, hr_data: Dict[str, Any]) -> None:
        """
        Build ChromaDB collection from HR data
        
        Args:
            hr_data: Dictionary containing HR documents and data
        """
        print("Building ChromaDB collection...")
        
        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "HR documents and data for RAG"}
            )
        except Exception:
            # Collection might already exist
            self.collection = self.client.get_collection(name=self.collection_name)
            # Clear existing data
            self.collection.delete()
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "HR documents and data for RAG"}
            )
        
        # Prepare documents for insertion
        documents = []
        metadatas = []
        ids = []
        
        # Process employees
        if 'employees' in hr_data:
            for emp in hr_data['employees']:
                doc_text = self._employee_to_text(emp)
                documents.append(doc_text)
                metadatas.append({
                    'type': 'employee',
                    'id': emp['id'],
                    'name': emp['name'],
                    'department': emp['department'],
                    'job_title': emp['job_title'],
                    'location': emp.get('location', ''),
                    'employment_type': emp.get('employment_type', ''),
                    'source': 'employee_data',
                    'last_updated': emp.get('last_updated', datetime.now().isoformat())
                })
                ids.append(f"emp_{emp['id']}")
        
        # Process policies
        if 'policies' in hr_data:
            for policy in hr_data['policies']:
                doc_text = self._policy_to_text(policy)
                documents.append(doc_text)
                metadatas.append({
                    'type': 'policy',
                    'id': policy['id'],
                    'title': policy['title'],
                    'policy_type': policy.get('type', ''),
                    'department': policy.get('department', 'All'),
                    'priority': policy.get('priority', 'Medium'),
                    'doc_type': 'Policy',
                    'effective_date': policy.get('effective_date', ''),
                    'version': policy.get('version', ''),
                    'source': 'policy_document',
                    'last_updated': policy.get('last_updated', datetime.now().isoformat())
                })
                ids.append(f"policy_{policy['id']}")
        
        # Process documents
        if 'documents' in hr_data:
            for doc in hr_data['documents']:
                doc_text = self._document_to_text(doc)
                documents.append(doc_text)
                metadatas.append({
                    'type': 'document',
                    'id': doc['id'],
                    'title': doc['title'],
                    'doc_type': doc.get('doc_type', 'Document'),
                    'department': doc.get('department', 'All'),
                    'priority': doc.get('priority', 'Medium'),
                    'author': doc.get('author', ''),
                    'status': doc.get('status', 'Active'),
                    'source': 'hr_document',
                    'last_updated': doc.get('last_updated', datetime.now().isoformat())
                })
                ids.append(f"doc_{doc['id']}")
        
        if not documents:
            raise ValueError("No documents found in HR data")
        
        # Add documents to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
        
        self.is_built = True
        print(f"ChromaDB collection built with {len(documents)} documents")
    
    def query_with_filters(self, query_text: str, top_k: int = 5, 
                          filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query with metadata filters
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            filters: Dictionary of metadata filters
            
        Returns:
            List of search results with content, metadata, and distances
        """
        if not self.is_built:
            raise ValueError("Collection not built. Call build_collection() first.")
        
        # Build where clause for filtering
        where_clause = {}
        if filters:
            for key, value in filters.items():
                if value and value != "All":
                    where_clause[key] = value
        
        # Query the collection
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_clause if where_clause else None
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                result = {
                    'rank': i + 1,
                    'content': doc,
                    'metadata': metadata,
                    'distance': distance,
                    'similarity': 1 - distance  # Convert distance to similarity
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def find_similar_documents(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find documents similar to the query without filters"""
        return self.query_with_filters(query_text, top_k)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        if not self.is_built:
            return {"status": "not_built"}
        
        try:
            count = self.collection.count()
            return {
                "status": "built",
                "name": self.collection_name,
                "count": count,
                "embedding_model": self.embedding_model_name
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def analyze_metadata(self) -> Dict[str, Dict[str, int]]:
        """Analyze metadata distribution in the collection"""
        if not self.is_built:
            return {}
        
        # Get all documents with metadata
        all_results = self.collection.get()
        
        analysis = {}
        metadata_fields = ['type', 'department', 'doc_type', 'priority', 'employment_type']
        
        for field in metadata_fields:
            field_analysis = {}
            for metadata in all_results['metadatas']:
                value = metadata.get(field, 'Unknown')
                field_analysis[value] = field_analysis.get(value, 0) + 1
            
            if field_analysis:
                analysis[field] = field_analysis
        
        return analysis
    
    def reset_collection(self) -> None:
        """Reset the collection by deleting all documents"""
        if self.collection:
            try:
                self.client.delete_collection(name=self.collection_name)
                self.collection = None
                self.is_built = False
                print("Collection reset successfully")
            except Exception as e:
                print(f"Error resetting collection: {e}")
    
    def export_collection(self) -> str:
        """Export collection data as JSON"""
        if not self.is_built:
            return json.dumps({"error": "Collection not built"})
        
        try:
            all_data = self.collection.get()
            export_data = {
                "collection_name": self.collection_name,
                "export_timestamp": datetime.now().isoformat(),
                "document_count": len(all_data['documents']),
                "documents": []
            }
            
            for i, (doc, metadata, doc_id) in enumerate(zip(
                all_data['documents'],
                all_data['metadatas'],
                all_data['ids']
            )):
                export_data["documents"].append({
                    "id": doc_id,
                    "content": doc,
                    "metadata": metadata
                })
            
            return json.dumps(export_data, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def search_by_metadata(self, metadata_filters: Dict[str, Any], 
                          limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search documents by metadata only (no text query)
        
        Args:
            metadata_filters: Dictionary of metadata filters
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        if not self.is_built:
            raise ValueError("Collection not built. Call build_collection() first.")
        
        # Build where clause
        where_clause = {}
        for key, value in metadata_filters.items():
            if value and value != "All":
                where_clause[key] = value
        
        try:
            results = self.collection.get(
                where=where_clause if where_clause else None,
                limit=limit
            )
            
            formatted_results = []
            for i, (doc, metadata, doc_id) in enumerate(zip(
                results['documents'],
                results['metadatas'],
                results['ids']
            )):
                result = {
                    'rank': i + 1,
                    'id': doc_id,
                    'content': doc,
                    'metadata': metadata
                }
                formatted_results.append(result)
            
            return formatted_results
        except Exception as e:
            print(f"Error in metadata search: {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        if not self.is_built:
            return None
        
        try:
            results = self.collection.get(ids=[doc_id])
            if results['documents']:
                return {
                    'id': doc_id,
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
        except Exception as e:
            print(f"Error getting document {doc_id}: {e}")
        
        return None
    
    def _employee_to_text(self, employee: Dict[str, Any]) -> str:
        """Convert employee data to searchable text"""
        text_parts = [
            f"Employee: {employee.get('name', '')}",
            f"Department: {employee.get('department', '')}",
            f"Job Title: {employee.get('job_title', '')}",
            f"Email: {employee.get('email', '')}",
            f"Location: {employee.get('location', '')}",
            f"Employment Type: {employee.get('employment_type', '')}",
            f"Performance Rating: {employee.get('performance_rating', '')}"
        ]
        
        # Add contact information
        if employee.get('manager_id'):
            text_parts.append(f"Reports to: {employee['manager_id']}")
        
        return " | ".join(filter(None, text_parts))
    
    def _policy_to_text(self, policy: Dict[str, Any]) -> str:
        """Convert policy data to searchable text"""
        text_parts = [
            f"Policy: {policy.get('title', '')}",
            f"Type: {policy.get('type', '')}",
            f"Department: {policy.get('department', '')}",
            f"Priority: {policy.get('priority', '')}",
            f"Applies to: {policy.get('applies_to', '')}",
            f"Content: {policy.get('content', '')}"
        ]
        
        return " | ".join(filter(None, text_parts))
    
    def _document_to_text(self, document: Dict[str, Any]) -> str:
        """Convert document data to searchable text"""
        text_parts = [
            f"Document: {document.get('title', '')}",
            f"Type: {document.get('doc_type', '')}",
            f"Department: {document.get('department', '')}",
            f"Author: {document.get('author', '')}",
            f"Status: {document.get('status', '')}",
            f"Content: {document.get('content', '')}"
        ]
        
        return " | ".join(filter(None, text_parts))

