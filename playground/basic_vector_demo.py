#!/usr/bin/env python3
"""
Basic Vector Search Demo
Demonstrates traditional embedding-based retrieval for HR information
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.vector_rag import VectorRAG
from utils.hr_data_generator import HRDataGenerator

def main():
    print("🔍 Basic Vector Search Demo")
    print("=" * 50)
    
    # Create test-data directory if it doesn't exist
    test_data_dir = Path("../test-data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Generate sample data
    print("\n📊 Generating HR sample data...")
    generator = HRDataGenerator()
    hr_data = generator.generate_comprehensive_data(
        num_employees=20,
        num_policies=8
    )
    
    # Save sample data to test-data directory
    hr_data_file = test_data_dir / "hr_sample_data.json"
    with open(hr_data_file, 'w') as f:
        json.dump(hr_data, f, indent=2, default=str)
    print(f"✅ Sample data saved to: {hr_data_file}")
    
    # Initialize Vector RAG
    print("\n🤖 Initializing Vector RAG...")
    vector_rag = VectorRAG(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=500
    )
    vector_rag.build_index(hr_data)
    print("✅ Vector RAG initialized successfully!")
    
    # Test queries
    test_queries = [
        "What is the vacation policy?",
        "How do I submit a timesheet?",
        "Who is the HR manager for engineering?",
        "What are the health insurance benefits?",
        "How to request parental leave?",
        "What is the dress code policy?",
        "How do I report harassment?",
        "What are the remote work policies?"
    ]
    
    print("\n🔍 Running test queries...")
    results_file = test_data_dir / "vector_search_results.json"
    all_results = {}
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_rag.query(query, top_k=3)
        
        # Display results
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result['score']:.3f}")
            print(f"     Content: {result['content'][:100]}...")
            print(f"     Metadata: {result['metadata']}")
        
        all_results[query] = results
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅ Search results saved to: {results_file}")
    
    print("\n🎉 Basic Vector Search Demo completed!")
    print(f"📁 Check the test-data directory for generated files:")

if __name__ == "__main__":
    main() 