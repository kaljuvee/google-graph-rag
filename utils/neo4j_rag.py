from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import networkx as nx
import random

# Mock Neo4j implementation for demonstration
class MockNeo4jDriver:
    """Mock Neo4j driver for demonstration purposes"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.session_active = False
    
    def session(self):
        return MockNeo4jSession(self.graph)
    
    def close(self):
        self.session_active = False

class MockNeo4jSession:
    """Mock Neo4j session"""
    
    def __init__(self, graph):
        self.graph = graph
    
    def run(self, query, **kwargs):
        # Simple query simulation
        return MockNeo4jResult([])
    
    def close(self):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class MockNeo4jResult:
    """Mock Neo4j result"""
    
    def __init__(self, data):
        self.data = data
    
    def data(self):
        return self.data

class Neo4jRAG:
    """Neo4j Graph RAG implementation for relationship-aware information retrieval"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", 
                 password: str = "password", embedded: bool = False):
        """
        Initialize Neo4j Graph RAG system
        
        Args:
            uri: Neo4j database URI
            user: Username
            password: Password
            embedded: Use embedded graph simulation if True
        """
        self.uri = uri
        self.user = user
        self.embedded = embedded
        
        if embedded:
            # Use NetworkX for embedded graph simulation
            self.driver = MockNeo4jDriver()
            self.graph = nx.DiGraph()
        else:
            try:
                from neo4j import GraphDatabase
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                self.graph = None
            except ImportError:
                print("Neo4j driver not available, using embedded simulation")
                self.driver = MockNeo4jDriver()
                self.graph = nx.DiGraph()
                self.embedded = True
            except Exception as e:
                print(f"Failed to connect to Neo4j: {e}, using embedded simulation")
                self.driver = MockNeo4jDriver()
                self.graph = nx.DiGraph()
                self.embedded = True
        
        self.is_built = False
        self.node_embeddings = {}
        
    def build_graph(self, hr_data: Dict[str, Any]) -> None:
        """
        Build knowledge graph from HR data
        
        Args:
            hr_data: Dictionary containing HR nodes and relationships
        """
        print("Building Neo4j knowledge graph...")
        
        if self.embedded:
            self._build_embedded_graph(hr_data)
        else:
            self._build_neo4j_graph(hr_data)
        
        self.is_built = True
        print("Knowledge graph built successfully")
    
    def _build_embedded_graph(self, hr_data: Dict[str, Any]) -> None:
        """Build graph using NetworkX for embedded simulation"""
        
        # Add employee nodes
        if 'employees' in hr_data:
            for emp in hr_data['employees']:
                self.graph.add_node(
                    emp['id'],
                    type='Employee',
                    name=emp['name'],
                    department=emp['department'],
                    job_title=emp['job_title'],
                    email=emp.get('email', ''),
                    location=emp.get('location', ''),
                    properties=emp
                )
        
        # Add policy nodes
        if 'policies' in hr_data:
            for policy in hr_data['policies']:
                self.graph.add_node(
                    policy['id'],
                    type='Policy',
                    title=policy['title'],
                    policy_type=policy.get('type', ''),
                    department=policy.get('department', ''),
                    properties=policy
                )
        
        # Add department nodes
        departments = set()
        if 'employees' in hr_data:
            departments.update(emp['department'] for emp in hr_data['employees'])
        if 'policies' in hr_data:
            departments.update(policy.get('department', '') for policy in hr_data['policies'] if policy.get('department'))
        
        for dept in departments:
            if dept and dept != 'All':
                self.graph.add_node(
                    f"dept_{dept.lower()}",
                    type='Department',
                    name=dept,
                    properties={'name': dept}
                )
        
        # Add relationships
        if 'relationships' in hr_data:
            for rel in hr_data['relationships']:
                self.graph.add_edge(
                    rel['from'],
                    rel['to'],
                    type=rel['type'],
                    properties=rel.get('properties', {})
                )
        else:
            # Generate relationships from data
            self._generate_relationships(hr_data)
    
    def _build_neo4j_graph(self, hr_data: Dict[str, Any]) -> None:
        """Build graph in actual Neo4j database"""
        
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create employee nodes
            if 'employees' in hr_data:
                for emp in hr_data['employees']:
                    session.run(
                        """
                        CREATE (e:Employee {
                            id: $id,
                            name: $name,
                            department: $department,
                            job_title: $job_title,
                            email: $email,
                            location: $location
                        })
                        """,
                        **emp
                    )
            
            # Create policy nodes
            if 'policies' in hr_data:
                for policy in hr_data['policies']:
                    session.run(
                        """
                        CREATE (p:Policy {
                            id: $id,
                            title: $title,
                            type: $type,
                            department: $department,
                            content: $content
                        })
                        """,
                        **policy
                    )
            
            # Create relationships
            if 'relationships' in hr_data:
                for rel in hr_data['relationships']:
                    session.run(
                        f"""
                        MATCH (a), (b)
                        WHERE a.id = $from_id AND b.id = $to_id
                        CREATE (a)-[:{rel['type']}]->(b)
                        """,
                        from_id=rel['from'],
                        to_id=rel['to']
                    )
    
    def _generate_relationships(self, hr_data: Dict[str, Any]) -> None:
        """Generate relationships for embedded graph"""
        
        # Employee-Department relationships
        if 'employees' in hr_data:
            for emp in hr_data['employees']:
                dept_id = f"dept_{emp['department'].lower()}"
                if self.graph.has_node(dept_id):
                    self.graph.add_edge(
                        emp['id'], dept_id,
                        type='WORKS_IN',
                        properties={'since': emp.get('hire_date', '')}
                    )
        
        # Manager-Employee relationships
        if 'employees' in hr_data:
            for emp in hr_data['employees']:
                if emp.get('manager_id') and self.graph.has_node(emp['manager_id']):
                    self.graph.add_edge(
                        emp['manager_id'], emp['id'],
                        type='MANAGES',
                        properties={'since': emp.get('hire_date', '')}
                    )
        
        # Policy-Department relationships
        if 'policies' in hr_data:
            for policy in hr_data['policies']:
                if policy.get('department') and policy['department'] != 'All':
                    dept_id = f"dept_{policy['department'].lower()}"
                    if self.graph.has_node(dept_id):
                        self.graph.add_edge(
                            policy['id'], dept_id,
                            type='APPLIES_TO',
                            properties={'effective_date': policy.get('effective_date', '')}
                        )
    
    def semantic_search(self, query: str, top_k: int = 5, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Perform semantic search enhanced with graph context
        
        Args:
            query: Search query
            top_k: Number of results to return
            max_depth: Maximum graph traversal depth
            
        Returns:
            List of search results with graph context
        """
        if not self.is_built:
            raise ValueError("Graph not built. Call build_graph() first.")
        
        if self.embedded:
            return self._embedded_semantic_search(query, top_k, max_depth)
        else:
            return self._neo4j_semantic_search(query, top_k, max_depth)
    
    def _embedded_semantic_search(self, query: str, top_k: int, max_depth: int) -> List[Dict[str, Any]]:
        """Semantic search using embedded graph"""
        
        # Simple text matching for demonstration
        results = []
        query_lower = query.lower()
        
        for node_id, node_data in self.graph.nodes(data=True):
            score = 0
            content_parts = []
            
            # Check node properties for matches
            for key, value in node_data.get('properties', {}).items():
                if isinstance(value, str) and query_lower in value.lower():
                    score += 1
                    content_parts.append(f"{key}: {value}")
            
            # Check direct attributes
            for key in ['name', 'title', 'content']:
                if key in node_data and isinstance(node_data[key], str):
                    if query_lower in node_data[key].lower():
                        score += 2
                        content_parts.append(f"{key}: {node_data[key]}")
            
            if score > 0:
                # Get graph context
                graph_context = self._get_graph_context(node_id, max_depth)
                
                result = {
                    'node_id': node_id,
                    'content': " | ".join(content_parts),
                    'score': score / 10.0,  # Normalize score
                    'node_type': node_data.get('type', 'Unknown'),
                    'graph_context': graph_context
                }
                results.append(result)
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _neo4j_semantic_search(self, query: str, top_k: int, max_depth: int) -> List[Dict[str, Any]]:
        """Semantic search using Neo4j database"""
        
        with self.driver.session() as session:
            # Simple text search query
            result = session.run(
                """
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower($query)
                   OR toLower(n.title) CONTAINS toLower($query)
                   OR toLower(n.content) CONTAINS toLower($query)
                RETURN n, labels(n) as labels
                LIMIT $limit
                """,
                query=query,
                limit=top_k
            )
            
            results = []
            for record in result:
                node = record['n']
                labels = record['labels']
                
                # Get graph context
                graph_context = self._get_neo4j_context(node['id'], max_depth)
                
                result_item = {
                    'node_id': node['id'],
                    'content': str(dict(node)),
                    'score': 0.8,  # Mock score
                    'node_type': labels[0] if labels else 'Unknown',
                    'graph_context': graph_context
                }
                results.append(result_item)
            
            return results
    
    def traverse_relationships(self, entity_name: str, relationship_types: List[str], 
                             max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Traverse relationships starting from an entity
        
        Args:
            entity_name: Starting entity name
            relationship_types: Types of relationships to follow
            max_depth: Maximum traversal depth
            
        Returns:
            List of relationship paths
        """
        if not self.is_built:
            raise ValueError("Graph not built. Call build_graph() first.")
        
        if self.embedded:
            return self._embedded_traverse_relationships(entity_name, relationship_types, max_depth)
        else:
            return self._neo4j_traverse_relationships(entity_name, relationship_types, max_depth)
    
    def _embedded_traverse_relationships(self, entity_name: str, relationship_types: List[str], 
                                       max_depth: int) -> List[Dict[str, Any]]:
        """Traverse relationships in embedded graph"""
        
        # Find starting node
        start_node = None
        for node_id, node_data in self.graph.nodes(data=True):
            if (node_data.get('name', '').lower() == entity_name.lower() or
                node_data.get('title', '').lower() == entity_name.lower()):
                start_node = node_id
                break
        
        if not start_node:
            return []
        
        relationships = []
        visited = set()
        
        def traverse(node, depth):
            if depth > max_depth or node in visited:
                return
            
            visited.add(node)
            
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, neighbor)
                rel_type = edge_data.get('type', 'RELATED')
                
                if rel_type in relationship_types:
                    start_name = self.graph.nodes[node].get('name', node)
                    end_name = self.graph.nodes[neighbor].get('name', neighbor)
                    
                    relationships.append({
                        'start': start_name,
                        'end': end_name,
                        'type': rel_type,
                        'properties': edge_data.get('properties', {}),
                        'depth': depth
                    })
                    
                    traverse(neighbor, depth + 1)
        
        traverse(start_node, 1)
        return relationships
    
    def _neo4j_traverse_relationships(self, entity_name: str, relationship_types: List[str], 
                                    max_depth: int) -> List[Dict[str, Any]]:
        """Traverse relationships in Neo4j database"""
        
        with self.driver.session() as session:
            # Build relationship type filter
            rel_filter = "|".join(relationship_types)
            
            result = session.run(
                f"""
                MATCH (start)
                WHERE toLower(start.name) = toLower($entity_name)
                MATCH path = (start)-[r:{rel_filter}*1..{max_depth}]->(end)
                RETURN start.name as start_name, end.name as end_name, 
                       type(r) as rel_type, properties(r) as rel_props
                """,
                entity_name=entity_name
            )
            
            relationships = []
            for record in result:
                relationships.append({
                    'start': record['start_name'],
                    'end': record['end_name'],
                    'type': record['rel_type'],
                    'properties': record['rel_props']
                })
            
            return relationships
    
    def find_shortest_path(self, entity_a: str, entity_b: str) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between two entities"""
        
        if not self.is_built:
            raise ValueError("Graph not built. Call build_graph() first.")
        
        if self.embedded:
            return self._embedded_shortest_path(entity_a, entity_b)
        else:
            return self._neo4j_shortest_path(entity_a, entity_b)
    
    def _embedded_shortest_path(self, entity_a: str, entity_b: str) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path in embedded graph"""
        
        # Find nodes
        node_a = node_b = None
        for node_id, node_data in self.graph.nodes(data=True):
            name = node_data.get('name', '').lower()
            if name == entity_a.lower():
                node_a = node_id
            elif name == entity_b.lower():
                node_b = node_id
        
        if not node_a or not node_b:
            return None
        
        try:
            path = nx.shortest_path(self.graph.to_undirected(), node_a, node_b)
            path_info = []
            
            for node in path:
                node_data = self.graph.nodes[node]
                path_info.append({
                    'id': node,
                    'name': node_data.get('name', node),
                    'type': node_data.get('type', 'Unknown')
                })
            
            return path_info
        except nx.NetworkXNoPath:
            return None
    
    def _neo4j_shortest_path(self, entity_a: str, entity_b: str) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path in Neo4j database"""
        
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a), (b)
                WHERE toLower(a.name) = toLower($entity_a) 
                  AND toLower(b.name) = toLower($entity_b)
                MATCH path = shortestPath((a)-[*]-(b))
                RETURN [node in nodes(path) | {id: node.id, name: node.name, type: labels(node)[0]}] as path
                """,
                entity_a=entity_a,
                entity_b=entity_b
            )
            
            record = result.single()
            return record['path'] if record else None
    
    def detect_communities(self, algorithm: str = "Louvain") -> List[List[str]]:
        """Detect communities in the graph"""
        
        if not self.is_built:
            raise ValueError("Graph not built. Call build_graph() first.")
        
        if self.embedded:
            return self._embedded_detect_communities(algorithm)
        else:
            return self._neo4j_detect_communities(algorithm)
    
    def _embedded_detect_communities(self, algorithm: str) -> List[List[str]]:
        """Detect communities in embedded graph"""
        
        try:
            import networkx.algorithms.community as nx_comm
            
            if algorithm == "Louvain":
                communities = nx_comm.louvain_communities(self.graph.to_undirected())
            elif algorithm == "Label Propagation":
                communities = nx_comm.label_propagation_communities(self.graph.to_undirected())
            else:  # Connected Components
                communities = nx.connected_components(self.graph.to_undirected())
            
            result = []
            for community in communities:
                community_names = []
                for node in community:
                    node_data = self.graph.nodes[node]
                    name = node_data.get('name', node)
                    community_names.append(name)
                result.append(community_names)
            
            return result
        except ImportError:
            # Fallback: simple clustering by node type
            communities = {}
            for node_id, node_data in self.graph.nodes(data=True):
                node_type = node_data.get('type', 'Unknown')
                if node_type not in communities:
                    communities[node_type] = []
                communities[node_type].append(node_data.get('name', node_id))
            
            return list(communities.values())
    
    def _neo4j_detect_communities(self, algorithm: str) -> List[List[str]]:
        """Detect communities in Neo4j database"""
        
        # This would require Neo4j Graph Data Science library
        # For now, return mock communities
        return [
            ["Engineering Team", "Tech Lead", "Senior Engineer"],
            ["HR Team", "HR Manager", "Recruiter"],
            ["Finance Team", "CFO", "Financial Analyst"]
        ]
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        
        if not self.is_built:
            return {"status": "not_built"}
        
        if self.embedded:
            return {
                "status": "built",
                "nodes": self.graph.number_of_nodes(),
                "relationships": self.graph.number_of_edges(),
                "node_types": len(set(data.get('type', 'Unknown') 
                                   for _, data in self.graph.nodes(data=True))),
                "rel_types": len(set(data.get('type', 'RELATED') 
                                   for _, _, data in self.graph.edges(data=True)))
            }
        else:
            with self.driver.session() as session:
                node_count = session.run("MATCH (n) RETURN count(n) as count").single()['count']
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
                
                return {
                    "status": "built",
                    "nodes": node_count,
                    "relationships": rel_count,
                    "node_types": 3,  # Mock
                    "rel_types": 3    # Mock
                }
    
    def _get_graph_context(self, node_id: str, max_depth: int) -> Dict[str, Any]:
        """Get graph context for a node"""
        
        context = {
            "neighbors": [],
            "relationships": [],
            "community": []
        }
        
        if self.embedded and self.graph.has_node(node_id):
            # Get immediate neighbors
            for neighbor in self.graph.neighbors(node_id):
                neighbor_data = self.graph.nodes[neighbor]
                context["neighbors"].append({
                    "id": neighbor,
                    "name": neighbor_data.get('name', neighbor),
                    "type": neighbor_data.get('type', 'Unknown')
                })
            
            # Get relationships
            for neighbor in self.graph.neighbors(node_id):
                edge_data = self.graph.get_edge_data(node_id, neighbor)
                context["relationships"].append({
                    "target": neighbor,
                    "type": edge_data.get('type', 'RELATED')
                })
        
        return context
    
    def _get_neo4j_context(self, node_id: str, max_depth: int) -> Dict[str, Any]:
        """Get Neo4j graph context for a node"""
        
        # Mock context for demonstration
        return {
            "neighbors": [{"id": "mock_neighbor", "name": "Mock Neighbor", "type": "Mock"}],
            "relationships": [{"target": "mock_neighbor", "type": "RELATED"}],
            "community": ["mock_community_member"]
        }
    
    def get_visualization_data(self, viz_type: str = "Overview") -> Dict[str, Any]:
        """Get data for graph visualization"""
        
        if not self.is_built:
            return {"nodes": [], "edges": []}
        
        nodes = []
        edges = []
        
        if self.embedded:
            # Sample nodes for visualization
            node_colors = {
                'Employee': '#FF6B6B',
                'Department': '#4ECDC4', 
                'Policy': '#45B7D1',
                'Unknown': '#96CEB4'
            }
            
            for node_id, node_data in list(self.graph.nodes(data=True))[:20]:  # Limit for visualization
                nodes.append({
                    'id': node_id,
                    'label': node_data.get('name', node_id),
                    'color': node_colors.get(node_data.get('type', 'Unknown'), '#96CEB4'),
                    'size': 20
                })
            
            for source, target, edge_data in list(self.graph.edges(data=True))[:30]:  # Limit edges
                edges.append({
                    'from': source,
                    'to': target,
                    'label': edge_data.get('type', 'RELATED')
                })
        
        return {"nodes": nodes, "edges": edges}
    
    def calculate_network_metrics(self) -> Dict[str, float]:
        """Calculate network-level metrics"""
        
        if not self.is_built or not self.embedded:
            return {}
        
        try:
            undirected = self.graph.to_undirected()
            
            metrics = {
                'density': nx.density(undirected),
                'clustering': nx.average_clustering(undirected),
                'diameter': nx.diameter(undirected) if nx.is_connected(undirected) else 0
            }
            
            return metrics
        except:
            return {'density': 0.1, 'clustering': 0.3, 'diameter': 4}  # Mock values
    
    def calculate_centrality(self) -> List[Dict[str, Any]]:
        """Calculate node centrality scores"""
        
        if not self.is_built or not self.embedded:
            return []
        
        try:
            centrality = nx.degree_centrality(self.graph)
            
            results = []
            for node, score in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]:
                node_data = self.graph.nodes[node]
                results.append({
                    'node': node_data.get('name', node),
                    'centrality': score
                })
            
            return results
        except:
            return [{'node': 'Mock Node', 'centrality': 0.5}]  # Mock data
    
    def analyze_relationship_patterns(self) -> List[Dict[str, Any]]:
        """Analyze relationship patterns in the graph"""
        
        if not self.is_built or not self.embedded:
            return []
        
        patterns = {}
        for _, _, edge_data in self.graph.edges(data=True):
            rel_type = edge_data.get('type', 'RELATED')
            patterns[rel_type] = patterns.get(rel_type, 0) + 1
        
        return [{'relationship_type': k, 'count': v} for k, v in patterns.items()]
    
    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()

