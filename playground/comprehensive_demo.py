#!/usr/bin/env python3
"""
Comprehensive RAG Demo
Runs all three RAG approaches and compares their performance
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.vector_rag import VectorRAG
from utils.neo4j_rag import Neo4jRAG
from utils.google_kg_rag import GoogleKnowledgeGraphRAG
from utils.hr_data_generator import HRDataGenerator

def run_vector_demo(hr_data, test_data_dir):
    """Run basic vector search demo"""
    print("\n🔍 Running Basic Vector Search...")
    start_time = time.time()
    
    vector_rag = VectorRAG(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=500
    )
    vector_rag.build_index(hr_data)
    
    # Test queries
    test_queries = [
        "What is the vacation policy?",
        "How do I submit a timesheet?",
        "Who is the HR manager for engineering?",
        "What are the health insurance benefits?",
        "How to request parental leave?"
    ]
    
    results = {}
    for query in test_queries:
        query_results = vector_rag.query(query, top_k=3)
        results[query] = query_results
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Save results
    results_file = test_data_dir / "comprehensive_vector_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "results": results,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    print(f"✅ Vector search completed in {execution_time:.2f} seconds")
    return results, execution_time

def run_graph_demo(hr_data, test_data_dir):
    """Run graph-based search demo"""
    print("\n🕸️ Running Graph-based Search...")
    start_time = time.time()
    
    try:
        neo4j_rag = Neo4jRAG(embedded=True)
        neo4j_rag.build_graph(hr_data)
        
        # Test queries
        test_queries = [
            "engineering team",
            "management hierarchy",
            "HR policies",
            "employee benefits"
        ]
        
        results = {}
        for query in test_queries:
            query_results = neo4j_rag.semantic_search(query, max_depth=2)
            results[query] = query_results
        
        neo4j_rag.close()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Save results
        results_file = test_data_dir / "comprehensive_graph_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "results": results,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        print(f"✅ Graph search completed in {execution_time:.2f} seconds")
        return results, execution_time
        
    except Exception as e:
        print(f"❌ Graph search failed: {e}")
        return {}, 0

def run_kg_demo(hr_data, test_data_dir):
    """Run Google Knowledge Graph demo"""
    print("\n🌐 Running Google Knowledge Graph Search...")
    start_time = time.time()
    
    api_key = os.getenv("GOOGLE_API_KEY", "demo_key")
    
    try:
        kg_rag = GoogleKnowledgeGraphRAG(api_key=api_key, hr_data=hr_data)
        
        # Test queries
        test_queries = [
            "employment law",
            "workplace safety",
            "employee benefits",
            "remote work"
        ]
        
        results = {}
        for query in test_queries:
            query_results = kg_rag.hybrid_search(query, mode="Internal Only")
            results[query] = query_results
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Save results
        results_file = test_data_dir / "comprehensive_kg_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "results": results,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        print(f"✅ Knowledge Graph search completed in {execution_time:.2f} seconds")
        return results, execution_time
        
    except Exception as e:
        print(f"❌ Knowledge Graph search failed: {e}")
        return {}, 0

def compare_results(vector_results, graph_results, kg_results, test_data_dir):
    """Compare results from different approaches"""
    print("\n📊 Comparing Results...")
    
    comparison = {
        "vector_search": {
            "total_results": sum(len(results) for results in vector_results.values()),
            "avg_results_per_query": sum(len(results) for results in vector_results.values()) / len(vector_results) if vector_results else 0
        },
        "graph_search": {
            "total_results": sum(len(results) for results in graph_results.values()),
            "avg_results_per_query": sum(len(results) for results in graph_results.values()) / len(graph_results) if graph_results else 0
        },
        "knowledge_graph": {
            "total_results": sum(len(results) for results in kg_results.values()),
            "avg_results_per_query": sum(len(results) for results in kg_results.values()) / len(kg_results) if kg_results else 0
        }
    }
    
    # Save comparison
    comparison_file = test_data_dir / "rag_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print("✅ Comparison saved to comparison file")
    return comparison

def main():
    print("🚀 Comprehensive RAG Demo")
    print("=" * 50)
    print("This demo will test all three RAG approaches and compare their performance")
    
    # Create test-data directory
    test_data_dir = Path("../test-data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Generate comprehensive sample data
    print("\n📊 Generating comprehensive HR sample data...")
    generator = HRDataGenerator()
    hr_data = generator.generate_comprehensive_data(
        num_employees=30,
        num_policies=12
    )
    
    # Save sample data
    hr_data_file = test_data_dir / "comprehensive_hr_data.json"
    with open(hr_data_file, 'w') as f:
        json.dump(hr_data, f, indent=2, default=str)
    print(f"✅ Comprehensive sample data saved to: {hr_data_file}")
    
    # Run all demos
    vector_results, vector_time = run_vector_demo(hr_data, test_data_dir)
    graph_results, graph_time = run_graph_demo(hr_data, test_data_dir)
    kg_results, kg_time = run_kg_demo(hr_data, test_data_dir)
    
    # Compare results
    comparison = compare_results(vector_results, graph_results, kg_results, test_data_dir)
    
    # Performance summary
    print("\n📈 Performance Summary")
    print("=" * 30)
    print(f"Vector Search:     {vector_time:.2f} seconds")
    print(f"Graph Search:      {graph_time:.2f} seconds")
    print(f"Knowledge Graph:   {kg_time:.2f} seconds")
    
    print("\n📊 Results Summary")
    print("=" * 30)
    print(f"Vector Search Results:     {comparison['vector_search']['total_results']} total")
    print(f"Graph Search Results:      {comparison['graph_search']['total_results']} total")
    print(f"Knowledge Graph Results:   {comparison['knowledge_graph']['total_results']} total")
    
    # Save final summary
    summary_file = test_data_dir / "demo_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "performance": {
            "vector_search_time": vector_time,
            "graph_search_time": graph_time,
            "knowledge_graph_time": kg_time
        },
        "results_summary": comparison,
        "data_generated": {
            "employees": len(hr_data.get("employees", [])),
            "policies": len(hr_data.get("policies", [])),
            "departments": len(hr_data.get("departments", []))
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ Demo summary saved to: {summary_file}")
    print("\n🎉 Comprehensive RAG Demo completed!")
    print(f"📁 All results saved in the test-data directory")

if __name__ == "__main__":
    main() 