#!/usr/bin/env python3
"""
Test Vector RAG implementation with sample HR data
"""

import sys
import os
import json
from datetime import datetime

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from vector_rag import VectorRAG
from hr_data_generator import HRDataGenerator

def test_vector_rag_basic():
    """Test basic Vector RAG functionality"""
    print("=" * 60)
    print("Testing Vector RAG - Basic Functionality")
    print("=" * 60)
    
    # Generate test data
    print("1. Generating HR test data...")
    generator = HRDataGenerator()
    hr_data = generator.generate_comprehensive_data(num_employees=20, num_policies=5)
    print(f"   Generated {len(hr_data['employees'])} employees, {len(hr_data['policies'])} policies")
    
    # Initialize Vector RAG
    print("2. Initializing Vector RAG...")
    vector_rag = VectorRAG(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=300
    )
    
    # Build index
    print("3. Building vector index...")
    vector_rag.build_index(hr_data)
    
    # Get statistics
    stats = vector_rag.get_statistics()
    print(f"   Index built with {stats['total_documents']} documents")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    
    return vector_rag, hr_data, stats

def test_vector_rag_queries():
    """Test Vector RAG query functionality"""
    print("\n" + "=" * 60)
    print("Testing Vector RAG - Query Functionality")
    print("=" * 60)
    
    vector_rag, hr_data, stats = test_vector_rag_basic()
    
    # Test queries
    test_queries = [
        "What is the vacation policy?",
        "Who works in the Engineering department?",
        "How do I submit a timesheet?",
        "What are the health benefits?",
        "Remote work policy information"
    ]
    
    results_summary = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing query: '{query}'")
        
        try:
            results = vector_rag.query(query, top_k=3)
            print(f"   Found {len(results)} results")
            
            if results:
                best_result = results[0]
                print(f"   Best match score: {best_result['score']:.3f}")
                print(f"   Content preview: {best_result['content'][:100]}...")
                
                results_summary.append({
                    'query': query,
                    'num_results': len(results),
                    'best_score': best_result['score'],
                    'success': True
                })
            else:
                print("   No results found")
                results_summary.append({
                    'query': query,
                    'num_results': 0,
                    'best_score': 0.0,
                    'success': False
                })
                
        except Exception as e:
            print(f"   Error: {str(e)}")
            results_summary.append({
                'query': query,
                'num_results': 0,
                'best_score': 0.0,
                'success': False,
                'error': str(e)
            })
    
    return results_summary

def test_vector_rag_filtering():
    """Test Vector RAG filtering functionality"""
    print("\n" + "=" * 60)
    print("Testing Vector RAG - Filtering Functionality")
    print("=" * 60)
    
    vector_rag, hr_data, stats = test_vector_rag_basic()
    
    # Test semantic search with filters
    print("1. Testing semantic search with type filter...")
    results = vector_rag.semantic_search(
        "employee information",
        top_k=5,
        filter_type="employee"
    )
    print(f"   Found {len(results)} employee results")
    
    print("2. Testing semantic search with department filter...")
    results = vector_rag.semantic_search(
        "engineering team",
        top_k=5,
        filter_department="Engineering"
    )
    print(f"   Found {len(results)} engineering results")
    
    return True

def test_vector_rag_performance():
    """Test Vector RAG performance with larger dataset"""
    print("\n" + "=" * 60)
    print("Testing Vector RAG - Performance")
    print("=" * 60)
    
    # Generate larger dataset
    print("1. Generating larger HR dataset...")
    generator = HRDataGenerator()
    hr_data = generator.generate_comprehensive_data(num_employees=100, num_policies=20)
    
    # Initialize and build index
    print("2. Building index for performance test...")
    start_time = datetime.now()
    
    vector_rag = VectorRAG(chunk_size=400)
    vector_rag.build_index(hr_data)
    
    build_time = (datetime.now() - start_time).total_seconds()
    print(f"   Index build time: {build_time:.2f} seconds")
    
    # Test query performance
    print("3. Testing query performance...")
    query_times = []
    
    for i in range(5):
        start_time = datetime.now()
        results = vector_rag.query("employee benefits policy", top_k=10)
        query_time = (datetime.now() - start_time).total_seconds()
        query_times.append(query_time)
        print(f"   Query {i+1}: {query_time:.3f}s, {len(results)} results")
    
    avg_query_time = sum(query_times) / len(query_times)
    print(f"   Average query time: {avg_query_time:.3f} seconds")
    
    return {
        'build_time': build_time,
        'avg_query_time': avg_query_time,
        'dataset_size': len(hr_data['employees']) + len(hr_data['policies'])
    }

def save_test_results(results_summary, performance_results):
    """Save test results to file"""
    print("\n" + "=" * 60)
    print("Saving Test Results")
    print("=" * 60)
    
    # Create test results directory
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'test-results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare results data
    test_results = {
        'test_timestamp': datetime.now().isoformat(),
        'test_type': 'vector_rag',
        'query_results': results_summary,
        'performance_results': performance_results,
        'summary': {
            'total_queries': len(results_summary),
            'successful_queries': sum(1 for r in results_summary if r['success']),
            'average_score': sum(r['best_score'] for r in results_summary if r['success']) / max(1, sum(1 for r in results_summary if r['success'])),
            'build_time': performance_results['build_time'],
            'avg_query_time': performance_results['avg_query_time']
        }
    }
    
    # Save to file
    results_file = os.path.join(results_dir, 'vector_rag_test_results.json')
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"Test results saved to: {results_file}")
    
    # Print summary
    summary = test_results['summary']
    print(f"\nTest Summary:")
    print(f"  Total queries: {summary['total_queries']}")
    print(f"  Successful queries: {summary['successful_queries']}")
    print(f"  Success rate: {summary['successful_queries']/summary['total_queries']*100:.1f}%")
    print(f"  Average relevance score: {summary['average_score']:.3f}")
    print(f"  Index build time: {summary['build_time']:.2f}s")
    print(f"  Average query time: {summary['avg_query_time']:.3f}s")

def main():
    """Run all Vector RAG tests"""
    print("Starting Vector RAG Tests")
    print("=" * 60)
    
    try:
        # Run tests
        results_summary = test_vector_rag_queries()
        test_vector_rag_filtering()
        performance_results = test_vector_rag_performance()
        
        # Save results
        save_test_results(results_summary, performance_results)
        
        print("\n" + "=" * 60)
        print("Vector RAG Tests Completed Successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

