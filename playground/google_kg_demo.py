#!/usr/bin/env python3
"""
Google Knowledge Graph Integration Demo
Demonstrates hybrid search combining internal HR data with external knowledge
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

from utils.google_kg_rag import GoogleKnowledgeGraphRAG
from utils.hr_data_generator import HRDataGenerator

def main():
    print("🌐 Google Knowledge Graph Integration Demo")
    print("=" * 50)
    
    # Create test-data directory if it doesn't exist
    test_data_dir = Path("../test-data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Generate sample data
    print("\n📊 Generating HR sample data...")
    generator = HRDataGenerator()
    hr_data = generator.generate_comprehensive_data(
        num_employees=15,
        num_policies=6
    )
    
    # Save sample data to test-data directory
    hr_data_file = test_data_dir / "hr_kg_data.json"
    with open(hr_data_file, 'w') as f:
        json.dump(hr_data, f, indent=2, default=str)
    print(f"✅ Sample data saved to: {hr_data_file}")
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n⚠️  No Google API key found!")
        print("💡 Set your Google API key as an environment variable:")
        print("   export GOOGLE_API_KEY='your_api_key_here'")
        print("\n📝 This demo will run in simulation mode without external API calls")
        api_key = "demo_key"
    
    # Initialize Google Knowledge Graph RAG
    print("\n🤖 Initializing Google Knowledge Graph RAG...")
    try:
        kg_rag = GoogleKnowledgeGraphRAG(api_key=api_key, hr_data=hr_data)
        print("✅ Google Knowledge Graph RAG initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing Google Knowledge Graph RAG: {e}")
        return
    
    # Test hybrid search modes
    print("\n🔍 Running hybrid search tests...")
    search_queries = [
        "employment law",
        "workplace safety regulations",
        "employee benefits requirements",
        "discrimination policies",
        "remote work guidelines"
    ]
    
    search_modes = [
        "Internal Only",
        "External Only", 
        "Hybrid (Internal + External)"
    ]
    
    all_results = {}
    
    for query in search_queries:
        print(f"\nQuery: {query}")
        query_results = {}
        
        for mode in search_modes:
            print(f"  Mode: {mode}")
            try:
                results = kg_rag.hybrid_search(query, mode=mode)
                print(f"    Found {len(results)} results")
                
                # Display top results
                for i, result in enumerate(results[:2]):  # Show top 2
                    print(f"    {i+1}. Type: {result.get('type', 'N/A')}")
                    print(f"       Content: {result.get('content', 'N/A')[:80]}...")
                    print(f"       Score: {result.get('score', 'N/A')}")
                
                query_results[mode] = results
            except Exception as e:
                print(f"    Error: {e}")
                query_results[mode] = []
        
        all_results[query] = query_results
    
    # Test entity search
    print("\n🏢 Testing entity search...")
    entity_queries = [
        "Google",
        "Microsoft",
        "Apple Inc",
        "Amazon"
    ]
    
    entity_results = {}
    for entity in entity_queries:
        print(f"\nSearching for entity: {entity}")
        try:
            results = kg_rag.search_entities(entity)
            print(f"Found {len(results)} entities")
            
            for i, result in enumerate(results[:3]):  # Show top 3
                print(f"  {i+1}. Name: {result.get('name', 'N/A')}")
                print(f"     Type: {result.get('type', 'N/A')}")
                print(f"     Description: {result.get('description', 'N/A')[:100]}...")
            
            entity_results[entity] = results
        except Exception as e:
            print(f"  Error: {e}")
            entity_results[entity] = []
    
    # Save results
    results_file = test_data_dir / "google_kg_results.json"
    final_results = {
        "hybrid_search": all_results,
        "entity_search": entity_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\n✅ Google Knowledge Graph results saved to: {results_file}")
    
    print("\n🎉 Google Knowledge Graph Integration Demo completed!")
    print(f"📁 Check the test-data directory for generated files:")

if __name__ == "__main__":
    main() 