import requests
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import hashlib

class GoogleKnowledgeGraphRAG:
    """Google Knowledge Graph integration for external knowledge enhancement"""
    
    def __init__(self, api_key: str, hr_data: Optional[Dict[str, Any]] = None, 
                 mock_mode: bool = False):
        """
        Initialize Google Knowledge Graph RAG
        
        Args:
            api_key: Google Knowledge Graph API key
            hr_data: Internal HR data for hybrid search
            mock_mode: Use mock responses for demonstration
        """
        self.api_key = api_key
        self.hr_data = hr_data or {}
        self.mock_mode = mock_mode
        self.base_url = "https://kgsearch.googleapis.com/v1/entities:search"
        
        # Cache for API responses
        self.cache = {}
        self.usage_stats = {
            'api_calls': 0,
            'entities_found': 0,
            'cache_hits': 0
        }
        
        # Query history
        self.query_history = []
        
        # Mock data for demonstration
        self.mock_entities = self._create_mock_entities()
        
    def hybrid_search(self, query: str, mode: str = "Hybrid (Internal + External)",
                     confidence_threshold: float = 0.7, max_external_results: int = 5,
                     entity_types: List[str] = None, languages: List[str] = None) -> Dict[str, Any]:
        """
        Perform hybrid search combining internal HR data with external knowledge
        
        Args:
            query: Search query
            mode: Search mode (Hybrid, Internal Only, External Only, Entity Enrichment)
            confidence_threshold: Minimum confidence for external results
            max_external_results: Maximum external results to return
            entity_types: Entity types to search for
            languages: Languages for search
            
        Returns:
            Combined search results from internal and external sources
        """
        results = {'internal': [], 'external': []}
        
        # Record query
        self.query_history.append({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'mode': mode
        })
        
        # Internal search
        if mode in ["Hybrid (Internal + External)", "Internal Only"]:
            results['internal'] = self._search_internal(query)
        
        # External search
        if mode in ["Hybrid (Internal + External)", "External Only", "Entity Enrichment"]:
            results['external'] = self._search_external(
                query, confidence_threshold, max_external_results, entity_types, languages
            )
        
        return results
    
    def _search_internal(self, query: str) -> List[Dict[str, Any]]:
        """Search internal HR data"""
        
        internal_results = []
        query_lower = query.lower()
        
        # Search employees
        for emp in self.hr_data.get('employees', []):
            score = 0
            content_parts = []
            
            # Check various fields
            fields_to_check = ['name', 'department', 'job_title', 'email']
            for field in fields_to_check:
                if field in emp and query_lower in str(emp[field]).lower():
                    score += 1
                    content_parts.append(f"{field}: {emp[field]}")
            
            if score > 0:
                internal_results.append({
                    'content': " | ".join(content_parts),
                    'score': score / len(fields_to_check),
                    'metadata': {
                        'type': 'employee',
                        'id': emp['id'],
                        'department': emp['department']
                    }
                })
        
        # Search policies
        for policy in self.hr_data.get('policies', []):
            score = 0
            content_parts = []
            
            fields_to_check = ['title', 'content', 'type']
            for field in fields_to_check:
                if field in policy and query_lower in str(policy[field]).lower():
                    score += 1
                    content_parts.append(f"{field}: {policy[field]}")
            
            if score > 0:
                internal_results.append({
                    'content': " | ".join(content_parts),
                    'score': score / len(fields_to_check),
                    'metadata': {
                        'type': 'policy',
                        'id': policy['id'],
                        'department': policy.get('department', 'All')
                    }
                })
        
        # Sort by score
        internal_results.sort(key=lambda x: x['score'], reverse=True)
        return internal_results[:5]
    
    def _search_external(self, query: str, confidence_threshold: float,
                        max_results: int, entity_types: List[str],
                        languages: List[str]) -> List[Dict[str, Any]]:
        """Search Google Knowledge Graph"""
        
        if self.mock_mode:
            return self._mock_external_search(query, max_results)
        
        # Check cache first
        cache_key = self._get_cache_key(query, entity_types, languages)
        if cache_key in self.cache:
            self.usage_stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        try:
            # Prepare API request
            params = {
                'query': query,
                'key': self.api_key,
                'limit': max_results,
                'indent': True
            }
            
            if entity_types:
                params['types'] = entity_types
            
            if languages:
                params['languages'] = languages
            
            # Make API request
            response = requests.get(self.base_url, params=params, timeout=10)
            self.usage_stats['api_calls'] += 1
            
            if response.status_code == 200:
                data = response.json()
                results = self._process_kg_response(data, confidence_threshold)
                
                # Cache results
                self.cache[cache_key] = results
                self.usage_stats['entities_found'] += len(results)
                
                return results
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return self._mock_external_search(query, max_results)
                
        except Exception as e:
            print(f"Error calling Google Knowledge Graph API: {e}")
            return self._mock_external_search(query, max_results)
    
    def _mock_external_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Mock external search for demonstration"""
        
        query_lower = query.lower()
        mock_results = []
        
        for entity in self.mock_entities:
            score = 0
            
            # Simple text matching
            if query_lower in entity['name'].lower():
                score += 2
            if query_lower in entity['description'].lower():
                score += 1
            
            # Check types
            for entity_type in entity['types']:
                if query_lower in entity_type.lower():
                    score += 1
            
            if score > 0:
                mock_results.append({
                    'name': entity['name'],
                    'description': entity['description'],
                    'types': entity['types'],
                    'confidence': min(score / 3.0, 1.0),
                    'url': entity.get('url', ''),
                    'id': entity['id']
                })
        
        # Sort by confidence and return top results
        mock_results.sort(key=lambda x: x['confidence'], reverse=True)
        return mock_results[:max_results]
    
    def _process_kg_response(self, data: Dict[str, Any], 
                           confidence_threshold: float) -> List[Dict[str, Any]]:
        """Process Google Knowledge Graph API response"""
        
        results = []
        
        for item in data.get('itemListElement', []):
            result = item.get('result', {})
            score = item.get('resultScore', 0)
            
            # Convert score to confidence (normalize)
            confidence = min(score / 1000.0, 1.0)  # Adjust normalization as needed
            
            if confidence >= confidence_threshold:
                entity_result = {
                    'name': result.get('name', ''),
                    'description': result.get('description', ''),
                    'types': result.get('@type', []),
                    'confidence': confidence,
                    'url': result.get('url', ''),
                    'id': result.get('@id', '')
                }
                
                # Add additional properties if available
                if 'detailedDescription' in result:
                    entity_result['detailed_description'] = result['detailedDescription'].get('articleBody', '')
                
                results.append(entity_result)
        
        return results
    
    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for entities by name"""
        
        if self.mock_mode:
            return self._mock_external_search(query, limit)
        
        try:
            params = {
                'query': query,
                'key': self.api_key,
                'limit': limit
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            self.usage_stats['api_calls'] += 1
            
            if response.status_code == 200:
                data = response.json()
                return self._process_kg_response(data, 0.0)  # No confidence threshold
            else:
                return []
                
        except Exception as e:
            print(f"Error searching entities: {e}")
            return self._mock_external_search(query, limit)
    
    def get_entity_details(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific entity"""
        
        if self.mock_mode:
            # Find mock entity
            for entity in self.mock_entities:
                if entity['id'] == entity_id:
                    return entity
            return None
        
        # In a real implementation, this would make a specific API call
        # Google Knowledge Graph API doesn't have a direct entity details endpoint
        # You would typically use the entity ID to search for more information
        return None
    
    def get_related_entities(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get entities related to the given entity"""
        
        # Mock implementation - in reality, this would require additional API calls
        # or use of other Google services like Wikidata
        return [
            {'name': 'Related Entity 1', 'description': 'A related entity'},
            {'name': 'Related Entity 2', 'description': 'Another related entity'}
        ]
    
    def create_knowledge_graph(self) -> Dict[str, Any]:
        """Create a knowledge graph from search results"""
        
        # This would typically build a graph from entity relationships
        # For now, return a simple structure
        return {
            'nodes': [
                {'id': 'hr_policies', 'label': 'HR Policies', 'type': 'concept'},
                {'id': 'employment_law', 'label': 'Employment Law', 'type': 'concept'},
                {'id': 'benefits', 'label': 'Employee Benefits', 'type': 'concept'}
            ],
            'edges': [
                {'from': 'hr_policies', 'to': 'employment_law', 'label': 'governed_by'},
                {'from': 'hr_policies', 'to': 'benefits', 'label': 'includes'}
            ]
        }
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get API usage statistics"""
        return self.usage_stats.copy()
    
    def get_knowledge_sources(self) -> Dict[str, int]:
        """Get distribution of knowledge sources"""
        
        internal_count = len(self.hr_data.get('employees', [])) + len(self.hr_data.get('policies', []))
        external_count = self.usage_stats['entities_found']
        
        return {
            'Internal HR Data': internal_count,
            'Google Knowledge Graph': external_count
        }
    
    def get_performance_metrics(self) -> List[Dict[str, Any]]:
        """Get performance metrics over time"""
        
        # Mock performance data
        metrics = []
        base_time = datetime.now()
        
        for i in range(10):
            metrics.append({
                'timestamp': (base_time.timestamp() - i * 3600) * 1000,  # Convert to milliseconds
                'response_time': 0.5 + (i * 0.1),
                'confidence_score': 0.8 - (i * 0.02)
            })
        
        return metrics
    
    def get_entity_type_distribution(self) -> Dict[str, int]:
        """Get distribution of entity types found"""
        
        type_counts = {}
        
        # Count from mock entities
        for entity in self.mock_entities:
            for entity_type in entity['types']:
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        return type_counts
    
    def export_query_history(self) -> str:
        """Export query history as JSON"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_queries': len(self.query_history),
            'queries': self.query_history
        }
        
        return json.dumps(export_data, indent=2)
    
    def export_entity_cache(self) -> str:
        """Export entity cache as JSON"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'cache_size': len(self.cache),
            'cache_entries': self.cache
        }
        
        return json.dumps(export_data, indent=2)
    
    def export_knowledge_graph(self) -> str:
        """Export knowledge graph data as JSON"""
        
        kg_data = self.create_knowledge_graph()
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'knowledge_graph': kg_data
        }
        
        return json.dumps(export_data, indent=2)
    
    def _get_cache_key(self, query: str, entity_types: List[str], 
                      languages: List[str]) -> str:
        """Generate cache key for query"""
        
        key_data = {
            'query': query,
            'entity_types': entity_types or [],
            'languages': languages or []
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _create_mock_entities(self) -> List[Dict[str, Any]]:
        """Create mock entities for demonstration"""
        
        return [
            {
                'id': 'kg_entity_1',
                'name': 'Employment Law',
                'description': 'Legal framework governing employer-employee relationships',
                'types': ['Thing', 'Concept'],
                'url': 'https://example.com/employment-law'
            },
            {
                'id': 'kg_entity_2',
                'name': 'GDPR',
                'description': 'General Data Protection Regulation for data privacy',
                'types': ['Thing', 'Regulation'],
                'url': 'https://example.com/gdpr'
            },
            {
                'id': 'kg_entity_3',
                'name': 'Remote Work',
                'description': 'Work arrangement allowing employees to work from locations outside the office',
                'types': ['Thing', 'Concept'],
                'url': 'https://example.com/remote-work'
            },
            {
                'id': 'kg_entity_4',
                'name': 'Employee Benefits',
                'description': 'Non-wage compensation provided to employees',
                'types': ['Thing', 'Concept'],
                'url': 'https://example.com/employee-benefits'
            },
            {
                'id': 'kg_entity_5',
                'name': 'Human Resources',
                'description': 'Department responsible for managing employee relations and policies',
                'types': ['Organization', 'Department'],
                'url': 'https://example.com/human-resources'
            },
            {
                'id': 'kg_entity_6',
                'name': 'Performance Management',
                'description': 'Process of ensuring employees meet organizational goals',
                'types': ['Thing', 'Process'],
                'url': 'https://example.com/performance-management'
            },
            {
                'id': 'kg_entity_7',
                'name': 'Workplace Safety',
                'description': 'Practices and policies to ensure employee safety at work',
                'types': ['Thing', 'Concept'],
                'url': 'https://example.com/workplace-safety'
            },
            {
                'id': 'kg_entity_8',
                'name': 'Diversity and Inclusion',
                'description': 'Organizational efforts to create an inclusive workplace',
                'types': ['Thing', 'Concept'],
                'url': 'https://example.com/diversity-inclusion'
            }
        ]

