#!/usr/bin/env python3
"""
Test Neo4j Graph RAG implementation with sample HR data
"""

import sys
import os
import json
from datetime import datetime

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from neo4j_rag import Neo4jRAG
from hr_data_generator import HRDataGenerator

def test_neo4j_rag_basic():
    """Test basic Neo4j Graph RAG functionality"""
    print("=" * 60)
    print("Testing Neo4j Graph RAG - Basic Functionality")
    print("=" * 60)
    
    # Generate test data
    print("1. Generating HR graph test data...")
    generator = HRDataGenerator()
    hr_data = generator.generate_graph_data(num_employees=20, num_policies=6)
    print(f"   Generated {len(hr_data['nodes']['employees'])} employees, {len(hr_data['nodes']['policies'])} policies")
    print(f"   Generated {len(hr_data['relationships'])} relationships")
    
    # Initialize Neo4j Graph RAG (embedded mode for testing)
    print("2. Initializing Neo4j Graph RAG (embedded mode)...")
    neo4j_rag = Neo4jRAG(embedded=True)
    
    # Build graph
    print("3. Building knowledge graph...")
    neo4j_rag.build_graph(hr_data)
    
    # Get graph statistics
    stats = neo4j_rag.get_graph_stats()
    print(f"   Graph built with {stats['nodes']} nodes and {stats['relationships']} relationships")
    print(f"   Node types: {stats['node_types']}, Relationship types: {stats['rel_types']}")
    
    return neo4j_rag, hr_data, stats

def test_neo4j_rag_semantic_search():
    """Test Neo4j Graph RAG semantic search"""
    print("\n" + "=" * 60)
    print("Testing Neo4j Graph RAG - Semantic Search")
    print("=" * 60)
    
    neo4j_rag, hr_data, stats = test_neo4j_rag_basic()
    
    # Test semantic search queries
    test_queries = [
        "vacation policy information",
        "engineering department employees",
        "remote work guidelines",
        "employee benefits",
        "performance management"
    ]
    
    results_summary = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing semantic search: '{query}'")
        
        try:
            results = neo4j_rag.semantic_search(query, top_k=3, max_depth=2)
            print(f"   Found {len(results)} results")
            
            if results:
                best_result = results[0]
                print(f"   Best match score: {best_result['score']:.3f}")
                print(f"   Node type: {best_result['node_type']}")
                print(f"   Content preview: {best_result['content'][:100]}...")
                
                # Check graph context
                if 'graph_context' in best_result:
                    context = best_result['graph_context']
                    print(f"   Graph context: {len(context.get('neighbors', []))} neighbors, {len(context.get('relationships', []))} relationships")
                
                results_summary.append({
                    'query': query,
                    'num_results': len(results),
                    'best_score': best_result['score'],
                    'node_type': best_result['node_type'],
                    'has_context': 'graph_context' in best_result,
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

def test_neo4j_rag_relationship_traversal():
    """Test Neo4j Graph RAG relationship traversal"""
    print("\n" + "=" * 60)
    print("Testing Neo4j Graph RAG - Relationship Traversal")
    print("=" * 60)
    
    neo4j_rag, hr_data, stats = test_neo4j_rag_basic()
    
    # Test relationship traversal
    test_entities = [
        "Engineering",
        "HR",
        "Finance"
    ]
    
    relationship_types = ["WORKS_IN", "MANAGES", "APPLIES_TO"]
    
    traversal_results = []
    
    for i, entity in enumerate(test_entities, 1):
        print(f"\n{i}. Testing relationship traversal from: '{entity}'")
        
        try:
            relationships = neo4j_rag.traverse_relationships(
                entity, relationship_types, max_depth=2
            )
            print(f"   Found {len(relationships)} relationships")
            
            if relationships:
                for rel in relationships[:3]:  # Show first 3
                    print(f"   {rel['start']} --[{rel['type']}]--> {rel['end']}")
                
                traversal_results.append({
                    'entity': entity,
                    'num_relationships': len(relationships),
                    'relationship_types': list(set(rel['type'] for rel in relationships)),
                    'success': True
                })
            else:
                print("   No relationships found")
                traversal_results.append({
                    'entity': entity,
                    'num_relationships': 0,
                    'success': False
                })
                
        except Exception as e:
            print(f"   Error: {str(e)}")
            traversal_results.append({
                'entity': entity,
                'num_relationships': 0,
                'success': False,
                'error': str(e)
            })
    
    return traversal_results

def test_neo4j_rag_path_finding():
    """Test Neo4j Graph RAG path finding"""
    print("\n" + "=" * 60)
    print("Testing Neo4j Graph RAG - Path Finding")
    print("=" * 60)
    
    neo4j_rag, hr_data, stats = test_neo4j_rag_basic()
    
    # Test path finding between entities
    test_paths = [
        ("Engineering", "HR"),
        ("Finance", "Engineering"),
        ("HR", "Finance")
    ]
    
    path_results = []
    
    for i, (entity_a, entity_b) in enumerate(test_paths, 1):
        print(f"\n{i}. Finding path from '{entity_a}' to '{entity_b}'")
        
        try:
            path = neo4j_rag.find_shortest_path(entity_a, entity_b)
            
            if path:
                print(f"   Path found with {len(path)} nodes:")
                path_str = " → ".join([node['name'] for node in path])
                print(f"   {path_str}")
                
                path_results.append({
                    'from': entity_a,
                    'to': entity_b,
                    'path_length': len(path),
                    'path_nodes': [node['name'] for node in path],
                    'success': True
                })
            else:
                print("   No path found")
                path_results.append({
                    'from': entity_a,
                    'to': entity_b,
                    'path_length': 0,
                    'success': False
                })
                
        except Exception as e:
            print(f"   Error: {str(e)}")
            path_results.append({
                'from': entity_a,
                'to': entity_b,
                'path_length': 0,
                'success': False,
                'error': str(e)
            })
    
    return path_results

def test_neo4j_rag_community_detection():
    """Test Neo4j Graph RAG community detection"""
    print("\n" + "=" * 60)
    print("Testing Neo4j Graph RAG - Community Detection")
    print("=" * 60)
    
    neo4j_rag, hr_data, stats = test_neo4j_rag_basic()
    
    # Test different community detection algorithms
    algorithms = ["Louvain", "Label Propagation", "Connected Components"]
    
    community_results = []
    
    for i, algorithm in enumerate(algorithms, 1):
        print(f"\n{i}. Testing {algorithm} community detection...")
        
        try:
            communities = neo4j_rag.detect_communities(algorithm)
            print(f"   Found {len(communities)} communities")
            
            for j, community in enumerate(communities[:3]):  # Show first 3
                print(f"   Community {j+1}: {len(community)} members - {', '.join(community[:5])}{'...' if len(community) > 5 else ''}")
            
            community_results.append({
                'algorithm': algorithm,
                'num_communities': len(communities),
                'largest_community_size': max(len(c) for c in communities) if communities else 0,
                'total_members': sum(len(c) for c in communities),
                'success': True
            })
            
        except Exception as e:
            print(f"   Error: {str(e)}")
            community_results.append({
                'algorithm': algorithm,
                'num_communities': 0,
                'success': False,
                'error': str(e)
            })
    
    return community_results

def test_neo4j_rag_analytics():
    """Test Neo4j Graph RAG analytics"""
    print("\n" + "=" * 60)
    print("Testing Neo4j Graph RAG - Analytics")
    print("=" * 60)
    
    neo4j_rag, hr_data, stats = test_neo4j_rag_basic()
    
    # Test network metrics
    print("1. Testing network metrics...")
    try:
        metrics = neo4j_rag.calculate_network_metrics()
        print(f"   Network density: {metrics.get('density', 0):.3f}")
        print(f"   Average clustering: {metrics.get('clustering', 0):.3f}")
        print(f"   Network diameter: {metrics.get('diameter', 0)}")
    except Exception as e:
        print(f"   Error calculating network metrics: {str(e)}")
        metrics = {}
    
    # Test centrality analysis
    print("\n2. Testing centrality analysis...")
    try:
        centrality = neo4j_rag.calculate_centrality()
        print(f"   Calculated centrality for {len(centrality)} nodes")
        if centrality:
            top_node = centrality[0]
            print(f"   Most central node: {top_node['node']} (score: {top_node['centrality']:.3f})")
    except Exception as e:
        print(f"   Error calculating centrality: {str(e)}")
        centrality = []
    
    # Test relationship patterns
    print("\n3. Testing relationship pattern analysis...")
    try:
        patterns = neo4j_rag.analyze_relationship_patterns()
        print(f"   Found {len(patterns)} relationship types")
        for pattern in patterns:
            print(f"   {pattern['relationship_type']}: {pattern['count']} instances")
    except Exception as e:
        print(f"   Error analyzing patterns: {str(e)}")
        patterns = []
    
    return {
        'network_metrics': metrics,
        'centrality_analysis': centrality,
        'relationship_patterns': patterns
    }

def test_neo4j_rag_visualization():
    """Test Neo4j Graph RAG visualization data"""
    print("\n" + "=" * 60)
    print("Testing Neo4j Graph RAG - Visualization Data")
    print("=" * 60)
    
    neo4j_rag, hr_data, stats = test_neo4j_rag_basic()
    
    # Test visualization data generation
    viz_types = ["Overview", "Department Focus", "Employee Network"]
    
    viz_results = []
    
    for viz_type in viz_types:
        print(f"Testing {viz_type} visualization...")
        
        try:
            viz_data = neo4j_rag.get_visualization_data(viz_type)
            print(f"   Generated {len(viz_data['nodes'])} nodes and {len(viz_data['edges'])} edges")
            
            viz_results.append({
                'type': viz_type,
                'num_nodes': len(viz_data['nodes']),
                'num_edges': len(viz_data['edges']),
                'success': True
            })
            
        except Exception as e:
            print(f"   Error: {str(e)}")
            viz_results.append({
                'type': viz_type,
                'num_nodes': 0,
                'num_edges': 0,
                'success': False,
                'error': str(e)
            })
    
    return viz_results

def save_test_results(semantic_results, traversal_results, path_results, 
                     community_results, analytics_results, viz_results):
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
        'test_type': 'neo4j_graph_rag',
        'semantic_search_results': semantic_results,
        'relationship_traversal_results': traversal_results,
        'path_finding_results': path_results,
        'community_detection_results': community_results,
        'analytics_results': analytics_results,
        'visualization_results': viz_results,
        'summary': {
            'total_semantic_queries': len(semantic_results),
            'successful_semantic_queries': sum(1 for r in semantic_results if r['success']),
            'total_traversals': len(traversal_results),
            'successful_traversals': sum(1 for r in traversal_results if r['success']),
            'total_paths': len(path_results),
            'successful_paths': sum(1 for r in path_results if r['success']),
            'total_community_algorithms': len(community_results),
            'successful_community_algorithms': sum(1 for r in community_results if r['success']),
            'average_semantic_score': sum(r['best_score'] for r in semantic_results if r['success']) / max(1, sum(1 for r in semantic_results if r['success']))
        }
    }
    
    # Save to file
    results_file = os.path.join(results_dir, 'neo4j_rag_test_results.json')
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"Test results saved to: {results_file}")
    
    # Print summary
    summary = test_results['summary']
    print(f"\nTest Summary:")
    print(f"  Semantic search success rate: {summary['successful_semantic_queries']}/{summary['total_semantic_queries']}")
    print(f"  Relationship traversal success rate: {summary['successful_traversals']}/{summary['total_traversals']}")
    print(f"  Path finding success rate: {summary['successful_paths']}/{summary['total_paths']}")
    print(f"  Community detection success rate: {summary['successful_community_algorithms']}/{summary['total_community_algorithms']}")
    print(f"  Average semantic score: {summary['average_semantic_score']:.3f}")

def main():
    """Run all Neo4j Graph RAG tests"""
    print("Starting Neo4j Graph RAG Tests")
    print("=" * 60)
    
    try:
        # Run tests
        semantic_results = test_neo4j_rag_semantic_search()
        traversal_results = test_neo4j_rag_relationship_traversal()
        path_results = test_neo4j_rag_path_finding()
        community_results = test_neo4j_rag_community_detection()
        analytics_results = test_neo4j_rag_analytics()
        viz_results = test_neo4j_rag_visualization()
        
        # Save results
        save_test_results(semantic_results, traversal_results, path_results,
                         community_results, analytics_results, viz_results)
        
        print("\n" + "=" * 60)
        print("Neo4j Graph RAG Tests Completed Successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

