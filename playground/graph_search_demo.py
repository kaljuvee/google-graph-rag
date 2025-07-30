#!/usr/bin/env python3
"""
Graph-based Search Demo
Demonstrates Neo4j graph database integration for relationship-aware search
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

from utils.neo4j_rag import Neo4jRAG
from utils.hr_data_generator import HRDataGenerator

def main():
    print("🕸️ Graph-based Search Demo")
    print("=" * 50)
    
    # Create test-data directory if it doesn't exist
    test_data_dir = Path("../test-data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Generate sample data
    print("\n📊 Generating HR sample data...")
    generator = HRDataGenerator()
    hr_data = generator.generate_comprehensive_data(
        num_employees=25,
        num_policies=10
    )
    
    # Save sample data to test-data directory
    hr_data_file = test_data_dir / "hr_graph_data.json"
    with open(hr_data_file, 'w') as f:
        json.dump(hr_data, f, indent=2, default=str)
    print(f"✅ Sample data saved to: {hr_data_file}")
    
    # Initialize Neo4j RAG
    print("\n🤖 Initializing Neo4j RAG...")
    try:
        neo4j_rag = Neo4jRAG(embedded=True)
        neo4j_rag.build_graph(hr_data)
        print("✅ Neo4j RAG initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing Neo4j RAG: {e}")
        print("💡 Make sure Neo4j is running or use embedded mode")
        return
    
    # Test semantic search with graph context
    print("\n🔍 Running semantic search with graph context...")
    semantic_queries = [
        "engineering team",
        "management hierarchy",
        "HR policies",
        "employee benefits",
        "remote work"
    ]
    
    semantic_results = {}
    for query in semantic_queries:
        print(f"\nQuery: {query}")
        try:
            results = neo4j_rag.semantic_search(query, max_depth=2)
            print(f"Found {len(results)} results")
            
            for i, result in enumerate(results[:3]):  # Show top 3
                print(f"  {i+1}. Node: {result.get('node', 'N/A')}")
                print(f"     Score: {result.get('score', 'N/A')}")
                print(f"     Properties: {result.get('properties', {})}")
            
            semantic_results[query] = results
        except Exception as e:
            print(f"  Error: {e}")
            semantic_results[query] = []
    
    # Test relationship traversal
    print("\n🔄 Testing relationship traversal...")
    traversal_tests = [
        ("Engineering", ["WORKS_IN", "MANAGES"]),
        ("HR", ["WORKS_IN", "REPORTS_TO"]),
        ("Sales", ["WORKS_IN", "COLLABORATES_WITH"])
    ]
    
    traversal_results = {}
    for start_node, rel_types in traversal_tests:
        print(f"\nTraversing from '{start_node}' with relationships: {rel_types}")
        try:
            relationships = neo4j_rag.traverse_relationships(start_node, rel_types)
            print(f"Found {len(relationships)} relationships")
            
            for i, rel in enumerate(relationships[:5]):  # Show top 5
                print(f"  {i+1}. {rel.get('start_node', 'N/A')} -[{rel.get('type', 'N/A')}]-> {rel.get('end_node', 'N/A')}")
            
            traversal_results[f"{start_node}_{'_'.join(rel_types)}"] = relationships
        except Exception as e:
            print(f"  Error: {e}")
            traversal_results[f"{start_node}_{'_'.join(rel_types)}"] = []
    
    # Save results
    results_file = test_data_dir / "graph_search_results.json"
    all_results = {
        "semantic_search": semantic_results,
        "relationship_traversal": traversal_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅ Graph search results saved to: {results_file}")
    
    # Clean up
    try:
        neo4j_rag.close()
        print("✅ Neo4j connection closed")
    except:
        pass
    
    print("\n🎉 Graph-based Search Demo completed!")
    print(f"📁 Check the test-data directory for generated files:")

if __name__ == "__main__":
    main() 