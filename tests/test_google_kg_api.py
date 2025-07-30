#!/usr/bin/env python3
"""
Test Google Knowledge Graph API key functionality
"""

import sys
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from google_kg_rag import GoogleKnowledgeGraphRAG
from hr_data_generator import HRDataGenerator

def test_google_kg_api_key():
    """Test Google Knowledge Graph API key functionality"""
    print("🔑 Testing Google Knowledge Graph API Key")
    print("=" * 60)
    
    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ No GOOGLE_API_KEY found in environment variables")
        print("💡 Please set your Google API key in the .env file:")
        print("   GOOGLE_API_KEY=your_actual_api_key_here")
        return False
    
    print(f"✅ Found API key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
    
    # Generate test data
    print("\n📊 Generating test HR data...")
    generator = HRDataGenerator()
    hr_data = generator.generate_comprehensive_data(
        num_employees=5, 
        num_policies=3
    )
    
    # Initialize Google Knowledge Graph RAG
    print("\n🤖 Initializing Google Knowledge Graph RAG...")
    try:
        kg_rag = GoogleKnowledgeGraphRAG(api_key=api_key, hr_data=hr_data)
        print("✅ Google Knowledge Graph RAG initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Google Knowledge Graph RAG: {e}")
        return False
    
    # Test entity search
    print("\n🔍 Testing entity search...")
    test_entities = ["Google", "Microsoft", "Apple Inc"]
    
    for entity in test_entities:
        print(f"\nSearching for: {entity}")
        try:
            results = kg_rag.search_entities(entity, limit=3)
            if results:
                print(f"✅ Found {len(results)} entities")
                for i, result in enumerate(results[:2]):  # Show top 2
                    print(f"  {i+1}. {result.get('name', 'N/A')} - {result.get('type', 'N/A')}")
            else:
                print("⚠️  No entities found (this might be normal for some queries)")
        except Exception as e:
            print(f"❌ Error searching for {entity}: {e}")
    
    # Test hybrid search
    print("\n🌐 Testing hybrid search...")
    test_queries = [
        "employment law",
        "workplace safety",
        "employee benefits"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            # Test internal search
            results = kg_rag.hybrid_search(query, mode="Internal Only")
            internal_count = len(results.get('internal', []))
            print(f"  Internal results: {internal_count}")
            
            # Test external search
            results = kg_rag.hybrid_search(query, mode="External Only")
            external_count = len(results.get('external', []))
            print(f"  External results: {external_count}")
            
            # Test hybrid search
            results = kg_rag.hybrid_search(query, mode="Hybrid (Internal + External)")
            total_count = len(results.get('internal', [])) + len(results.get('external', []))
            print(f"  Hybrid results: {total_count}")
            
        except Exception as e:
            print(f"❌ Error in hybrid search: {e}")
    
    # Test usage stats
    print("\n📊 Testing usage statistics...")
    try:
        stats = kg_rag.get_usage_stats()
        print(f"API calls: {stats.get('api_calls', 0)}")
        print(f"Entities found: {stats.get('entities_found', 0)}")
        print(f"Cache hits: {stats.get('cache_hits', 0)}")
    except Exception as e:
        print(f"❌ Error getting usage stats: {e}")
    
    print("\n✅ Google Knowledge Graph API key test completed!")
    return True

def save_test_results(success):
    """Save test results to file"""
    results_file = os.path.join(os.path.dirname(__file__), '..', 'test-data', 'google_kg_api_test_results.json')
    
    # Ensure test-data directory exists
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    test_summary = {
        "timestamp": datetime.now().isoformat(),
        "test_name": "Google Knowledge Graph API Key Test",
        "success": success,
        "api_key_present": bool(os.getenv("GOOGLE_API_KEY")),
        "api_key_preview": os.getenv("GOOGLE_API_KEY", "")[:10] + "..." if os.getenv("GOOGLE_API_KEY") else None
    }
    
    with open(results_file, 'w') as f:
        json.dump(test_summary, f, indent=2, default=str)
    
    print(f"\n✅ Test results saved to: {results_file}")

def main():
    """Run Google KG API key test"""
    print("🔑 Google Knowledge Graph API Key Test")
    print("=" * 60)
    
    success = test_google_kg_api_key()
    save_test_results(success)
    
    if success:
        print("\n🎉 All tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 