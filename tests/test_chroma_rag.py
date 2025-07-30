#!/usr/bin/env python3
"""
Test ChromaDB RAG implementation with sample HR data
"""

import sys
import os
import json
from datetime import datetime

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from chroma_rag import ChromaRAG
from hr_data_generator import HRDataGenerator

def test_chroma_rag_basic():
    """Test basic ChromaDB RAG functionality"""
    print("=" * 60)
    print("Testing ChromaDB RAG - Basic Functionality")
    print("=" * 60)
    
    # Generate test data
    print("1. Generating HR test data...")
    generator = HRDataGenerator()
    hr_data = generator.generate_comprehensive_data(num_employees=25, num_policies=8)
    print(f"   Generated {len(hr_data['employees'])} employees, {len(hr_data['policies'])} policies")
    
    # Initialize ChromaDB RAG
    print("2. Initializing ChromaDB RAG...")
    chroma_rag = ChromaRAG(
        collection_name="test_hr_collection",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Build collection
    print("3. Building ChromaDB collection...")
    chroma_rag.build_collection(hr_data)
    
    # Get collection info
    info = chroma_rag.get_collection_info()
    print(f"   Collection built with {info['count']} documents")
    print(f"   Collection name: {info['name']}")
    
    return chroma_rag, hr_data, info

def test_chroma_rag_queries():
    """Test ChromaDB RAG query functionality"""
    print("\n" + "=" * 60)
    print("Testing ChromaDB RAG - Query Functionality")
    print("=" * 60)
    
    chroma_rag, hr_data, info = test_chroma_rag_basic()
    
    # Test queries
    test_queries = [
        "What is the vacation policy?",
        "Who works in the Engineering department?",
        "Remote work guidelines",
        "Employee benefits information",
        "Performance review process"
    ]
    
    results_summary = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing query: '{query}'")
        
        try:
            results = chroma_rag.query_with_filters(query, top_k=3)
            print(f"   Found {len(results)} results")
            
            if results:
                best_result = results[0]
                print(f"   Best match distance: {best_result['distance']:.3f}")
                print(f"   Best match similarity: {best_result['similarity']:.3f}")
                print(f"   Content preview: {best_result['content'][:100]}...")
                
                results_summary.append({
                    'query': query,
                    'num_results': len(results),
                    'best_distance': best_result['distance'],
                    'best_similarity': best_result['similarity'],
                    'success': True
                })
            else:
                print("   No results found")
                results_summary.append({
                    'query': query,
                    'num_results': 0,
                    'best_distance': 1.0,
                    'best_similarity': 0.0,
                    'success': False
                })
                
        except Exception as e:
            print(f"   Error: {str(e)}")
            results_summary.append({
                'query': query,
                'num_results': 0,
                'best_distance': 1.0,
                'best_similarity': 0.0,
                'success': False,
                'error': str(e)
            })
    
    return results_summary

def test_chroma_rag_filtering():
    """Test ChromaDB RAG filtering functionality"""
    print("\n" + "=" * 60)
    print("Testing ChromaDB RAG - Advanced Filtering")
    print("=" * 60)
    
    chroma_rag, hr_data, info = test_chroma_rag_basic()
    
    # Test department filtering
    print("1. Testing department filtering...")
    results = chroma_rag.query_with_filters(
        "employee information",
        top_k=5,
        filters={"department": "Engineering"}
    )
    print(f"   Found {len(results)} Engineering department results")
    
    # Test document type filtering
    print("2. Testing document type filtering...")
    results = chroma_rag.query_with_filters(
        "policy information",
        top_k=5,
        filters={"type": "policy"}
    )
    print(f"   Found {len(results)} policy results")
    
    # Test combined filtering
    print("3. Testing combined filtering...")
    results = chroma_rag.query_with_filters(
        "work guidelines",
        top_k=5,
        filters={"department": "HR", "priority": "High"}
    )
    print(f"   Found {len(results)} HR high-priority results")
    
    # Test metadata search
    print("4. Testing metadata-only search...")
    metadata_results = chroma_rag.search_by_metadata(
        {"doc_type": "Policy"},
        limit=10
    )
    print(f"   Found {len(metadata_results)} policy documents via metadata search")
    
    return True

def test_chroma_rag_analytics():
    """Test ChromaDB RAG analytics functionality"""
    print("\n" + "=" * 60)
    print("Testing ChromaDB RAG - Analytics")
    print("=" * 60)
    
    chroma_rag, hr_data, info = test_chroma_rag_basic()
    
    # Test metadata analysis
    print("1. Analyzing collection metadata...")
    metadata_analysis = chroma_rag.analyze_metadata()
    
    for field, analysis in metadata_analysis.items():
        print(f"   {field.title()} distribution:")
        for value, count in analysis.items():
            print(f"     {value}: {count}")
    
    # Test similar document finding
    print("\n2. Testing similar document search...")
    similar_docs = chroma_rag.find_similar_documents("vacation policy", top_k=5)
    print(f"   Found {len(similar_docs)} similar documents")
    
    # Test document retrieval by ID
    print("\n3. Testing document retrieval by ID...")
    if similar_docs:
        doc_id = similar_docs[0]['metadata'].get('id', 'unknown')
        retrieved_doc = chroma_rag.get_document_by_id(f"policy_{doc_id}")
        if retrieved_doc:
            print(f"   Successfully retrieved document: {retrieved_doc['id']}")
        else:
            print("   Document retrieval failed")
    
    return metadata_analysis

def test_chroma_rag_export():
    """Test ChromaDB RAG export functionality"""
    print("\n" + "=" * 60)
    print("Testing ChromaDB RAG - Export Functionality")
    print("=" * 60)
    
    chroma_rag, hr_data, info = test_chroma_rag_basic()
    
    # Test collection export
    print("1. Testing collection export...")
    export_data = chroma_rag.export_collection()
    
    try:
        export_json = json.loads(export_data)
        print(f"   Export successful: {export_json['document_count']} documents")
        print(f"   Export timestamp: {export_json['export_timestamp']}")
        
        # Save export to file
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'test-results')
        os.makedirs(results_dir, exist_ok=True)
        
        export_file = os.path.join(results_dir, 'chroma_collection_export.json')
        with open(export_file, 'w') as f:
            f.write(export_data)
        print(f"   Export saved to: {export_file}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"   Export failed: {str(e)}")
        return False

def test_chroma_rag_performance():
    """Test ChromaDB RAG performance"""
    print("\n" + "=" * 60)
    print("Testing ChromaDB RAG - Performance")
    print("=" * 60)
    
    # Generate larger dataset
    print("1. Generating larger HR dataset...")
    generator = HRDataGenerator()
    hr_data = generator.generate_comprehensive_data(num_employees=80, num_policies=15)
    
    # Initialize and build collection
    print("2. Building collection for performance test...")
    start_time = datetime.now()
    
    chroma_rag = ChromaRAG(collection_name="performance_test_collection")
    chroma_rag.build_collection(hr_data)
    
    build_time = (datetime.now() - start_time).total_seconds()
    print(f"   Collection build time: {build_time:.2f} seconds")
    
    # Test query performance
    print("3. Testing query performance...")
    query_times = []
    
    test_queries = [
        "employee benefits",
        "vacation policy",
        "remote work",
        "performance review",
        "safety procedures"
    ]
    
    for query in test_queries:
        start_time = datetime.now()
        results = chroma_rag.query_with_filters(query, top_k=5)
        query_time = (datetime.now() - start_time).total_seconds()
        query_times.append(query_time)
        print(f"   Query '{query}': {query_time:.3f}s, {len(results)} results")
    
    avg_query_time = sum(query_times) / len(query_times)
    print(f"   Average query time: {avg_query_time:.3f} seconds")
    
    # Test filtering performance
    print("4. Testing filtering performance...")
    start_time = datetime.now()
    filtered_results = chroma_rag.query_with_filters(
        "employee information",
        top_k=10,
        filters={"department": "Engineering", "type": "employee"}
    )
    filter_time = (datetime.now() - start_time).total_seconds()
    print(f"   Filtered query time: {filter_time:.3f}s, {len(filtered_results)} results")
    
    return {
        'build_time': build_time,
        'avg_query_time': avg_query_time,
        'filter_query_time': filter_time,
        'dataset_size': len(hr_data['employees']) + len(hr_data['policies'])
    }

def save_test_results(results_summary, metadata_analysis, performance_results):
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
        'test_type': 'chroma_rag',
        'query_results': results_summary,
        'metadata_analysis': metadata_analysis,
        'performance_results': performance_results,
        'summary': {
            'total_queries': len(results_summary),
            'successful_queries': sum(1 for r in results_summary if r['success']),
            'average_similarity': sum(r['best_similarity'] for r in results_summary if r['success']) / max(1, sum(1 for r in results_summary if r['success'])),
            'build_time': performance_results['build_time'],
            'avg_query_time': performance_results['avg_query_time'],
            'filter_query_time': performance_results['filter_query_time']
        }
    }
    
    # Save to file
    results_file = os.path.join(results_dir, 'chroma_rag_test_results.json')
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"Test results saved to: {results_file}")
    
    # Print summary
    summary = test_results['summary']
    print(f"\nTest Summary:")
    print(f"  Total queries: {summary['total_queries']}")
    print(f"  Successful queries: {summary['successful_queries']}")
    print(f"  Success rate: {summary['successful_queries']/summary['total_queries']*100:.1f}%")
    print(f"  Average similarity score: {summary['average_similarity']:.3f}")
    print(f"  Collection build time: {summary['build_time']:.2f}s")
    print(f"  Average query time: {summary['avg_query_time']:.3f}s")
    print(f"  Filtered query time: {summary['filter_query_time']:.3f}s")

def main():
    """Run all ChromaDB RAG tests"""
    print("Starting ChromaDB RAG Tests")
    print("=" * 60)
    
    try:
        # Run tests
        results_summary = test_chroma_rag_queries()
        test_chroma_rag_filtering()
        metadata_analysis = test_chroma_rag_analytics()
        test_chroma_rag_export()
        performance_results = test_chroma_rag_performance()
        
        # Save results
        save_test_results(results_summary, metadata_analysis, performance_results)
        
        print("\n" + "=" * 60)
        print("ChromaDB RAG Tests Completed Successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

