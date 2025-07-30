import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import tempfile

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from neo4j_rag import Neo4jRAG
    from hr_data_generator import HRDataGenerator
    from graph_visualizer import GraphVisualizer
except ImportError:
    st.error("Utils modules not found. Please ensure the utils directory is properly set up.")
    st.stop()

st.set_page_config(
    page_title="Neo4j Graph RAG",
    page_icon="🕸️",
    layout="wide"
)

def main():
    st.title("🕸️ Neo4j Graph RAG")
    st.markdown("### Relationship-Aware Information Retrieval")
    
    # Initialize session state
    if 'neo4j_rag' not in st.session_state:
        st.session_state.neo4j_rag = None
    if 'hr_data' not in st.session_state:
        st.session_state.hr_data = None
    if 'graph_built' not in st.session_state:
        st.session_state.graph_built = False
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Neo4j connection settings
    st.sidebar.subheader("Neo4j Connection")
    neo4j_uri = st.sidebar.text_input("Neo4j URI", "bolt://localhost:7687")
    neo4j_user = st.sidebar.text_input("Username", "neo4j")
    neo4j_password = st.sidebar.text_input("Password", "password", type="password")
    
    # Data generation section
    st.sidebar.subheader("Data Generation")
    num_employees = st.sidebar.slider("Number of Employees", 10, 100, 30)
    num_policies = st.sidebar.slider("Number of HR Policies", 5, 20, 8)
    
    if st.sidebar.button("Generate HR Graph Data"):
        with st.spinner("Generating HR graph data..."):
            generator = HRDataGenerator()
            st.session_state.hr_data = generator.generate_graph_data(
                num_employees=num_employees,
                num_policies=num_policies
            )
            st.sidebar.success("Graph data generated successfully!")
    
    # Graph RAG Configuration
    st.sidebar.subheader("Graph RAG Settings")
    max_depth = st.sidebar.slider("Graph Traversal Depth", 1, 4, 2)
    relationship_weight = st.sidebar.slider("Relationship Weight", 0.1, 1.0, 0.5)
    
    if st.sidebar.button("Initialize Neo4j RAG"):
        if st.session_state.hr_data is not None:
            with st.spinner("Connecting to Neo4j and building graph..."):
                try:
                    st.session_state.neo4j_rag = Neo4jRAG(
                        uri=neo4j_uri,
                        user=neo4j_user,
                        password=neo4j_password
                    )
                    st.session_state.neo4j_rag.build_graph(st.session_state.hr_data)
                    st.session_state.graph_built = True
                    st.sidebar.success("Neo4j Graph RAG initialized!")
                except Exception as e:
                    st.sidebar.error(f"Connection failed: {str(e)}")
                    st.sidebar.info("Using embedded graph simulation instead.")
                    # Fallback to embedded graph
                    st.session_state.neo4j_rag = Neo4jRAG(embedded=True)
                    st.session_state.neo4j_rag.build_graph(st.session_state.hr_data)
                    st.session_state.graph_built = True
        else:
            st.sidebar.error("Please generate HR data first!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Graph-Enhanced Query Interface")
        
        # Query types
        query_type = st.selectbox(
            "Query Type",
            ["Semantic Search", "Relationship Traversal", "Path Finding", "Community Detection"]
        )
        
        if query_type == "Semantic Search":
            st.markdown("**Find documents semantically similar to your query**")
            query = st.text_area("Enter your question:", height=100)
            top_k = st.slider("Number of Results", 1, 10, 5)
            
            if st.button("Search", type="primary"):
                if st.session_state.neo4j_rag is not None and query:
                    with st.spinner("Performing graph-enhanced search..."):
                        results = st.session_state.neo4j_rag.semantic_search(
                            query, top_k=top_k, max_depth=max_depth
                        )
                        display_search_results(results)
        
        elif query_type == "Relationship Traversal":
            st.markdown("**Explore relationships starting from an entity**")
            entity_name = st.text_input("Entity Name (e.g., employee name, department):")
            relationship_types = st.multiselect(
                "Relationship Types",
                ["WORKS_IN", "MANAGES", "REPORTS_TO", "RELATED_TO", "APPLIES_TO"],
                default=["WORKS_IN", "MANAGES"]
            )
            
            if st.button("Traverse", type="primary"):
                if st.session_state.neo4j_rag is not None and entity_name:
                    with st.spinner("Traversing graph relationships..."):
                        results = st.session_state.neo4j_rag.traverse_relationships(
                            entity_name, relationship_types, max_depth
                        )
                        display_relationship_results(results)
        
        elif query_type == "Path Finding":
            st.markdown("**Find connections between two entities**")
            col_a, col_b = st.columns(2)
            with col_a:
                entity_a = st.text_input("From Entity:")
            with col_b:
                entity_b = st.text_input("To Entity:")
            
            if st.button("Find Path", type="primary"):
                if st.session_state.neo4j_rag is not None and entity_a and entity_b:
                    with st.spinner("Finding shortest path..."):
                        path = st.session_state.neo4j_rag.find_shortest_path(entity_a, entity_b)
                        display_path_results(path)
        
        elif query_type == "Community Detection":
            st.markdown("**Discover communities and clusters in the organization**")
            algorithm = st.selectbox("Algorithm", ["Louvain", "Label Propagation", "Connected Components"])
            
            if st.button("Detect Communities", type="primary"):
                if st.session_state.neo4j_rag is not None:
                    with st.spinner("Detecting communities..."):
                        communities = st.session_state.neo4j_rag.detect_communities(algorithm)
                        display_community_results(communities)
    
    with col2:
        st.subheader("Graph Status")
        
        # Status indicators
        data_status = "✅ Ready" if st.session_state.hr_data is not None else "❌ Not Generated"
        graph_status = "✅ Ready" if st.session_state.graph_built else "❌ Not Built"
        
        st.write(f"**HR Data:** {data_status}")
        st.write(f"**Graph Database:** {graph_status}")
        
        if st.session_state.neo4j_rag is not None:
            st.subheader("Graph Statistics")
            stats = st.session_state.neo4j_rag.get_graph_stats()
            
            st.metric("Nodes", stats.get('nodes', 0))
            st.metric("Relationships", stats.get('relationships', 0))
            st.metric("Node Types", stats.get('node_types', 0))
            st.metric("Relationship Types", stats.get('rel_types', 0))
        
        # Graph visualization controls
        if st.session_state.graph_built:
            st.subheader("Graph Visualization")
            
            viz_type = st.selectbox("Visualization Type", ["Overview", "Department Focus", "Employee Network"])
            
            if st.button("Generate Visualization"):
                with st.spinner("Creating graph visualization..."):
                    create_graph_visualization(viz_type)
    
    # Sample queries section
    if st.session_state.graph_built:
        st.subheader("📝 Sample Graph Queries")
        
        sample_queries = {
            "Who reports to the Engineering Manager?": "relationship",
            "Find all policies related to vacation": "semantic",
            "What's the connection between HR and Finance departments?": "path",
            "Show me the organizational structure": "community"
        }
        
        cols = st.columns(len(sample_queries))
        for i, (query, qtype) in enumerate(sample_queries.items()):
            with cols[i]:
                if st.button(query, key=f"sample_{i}"):
                    execute_sample_query(query, qtype)
    
    # Graph analysis section
    if st.session_state.graph_built:
        st.subheader("📊 Graph Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Network Metrics", "Centrality Analysis", "Relationship Patterns"])
        
        with tab1:
            display_network_metrics()
        
        with tab2:
            display_centrality_analysis()
        
        with tab3:
            display_relationship_patterns()
    
    # Technical details
    with st.expander("🔧 Graph RAG Technical Details"):
        st.markdown("""
        ### Graph RAG Advantages
        
        1. **Relationship Context**: Leverages connections between entities for better understanding
        2. **Multi-hop Reasoning**: Can traverse multiple relationships to find relevant information
        3. **Structural Insights**: Reveals organizational patterns and hierarchies
        4. **Path-based Queries**: Find connections and dependencies between entities
        5. **Community Detection**: Identify clusters and groups within the organization
        
        ### Neo4j Features Used
        - **Cypher Queries**: Powerful graph query language
        - **Graph Algorithms**: Built-in algorithms for analysis
        - **Vector Similarity**: Combine graph structure with semantic similarity
        - **Real-time Traversal**: Dynamic relationship exploration
        
        ### Use Cases in HR
        - **Organizational Charts**: Dynamic hierarchy visualization
        - **Policy Dependencies**: Understanding how policies relate to each other
        - **Employee Networks**: Social and reporting relationships
        - **Knowledge Graphs**: Connecting documents, people, and processes
        """)

def display_search_results(results):
    """Display semantic search results with graph context"""
    if results:
        st.subheader("Graph-Enhanced Search Results")
        for i, result in enumerate(results):
            with st.expander(f"Result {i+1} (Score: {result['score']:.3f})"):
                st.write(result['content'])
                if 'graph_context' in result:
                    st.subheader("Graph Context")
                    st.json(result['graph_context'])
    else:
        st.warning("No results found.")

def display_relationship_results(results):
    """Display relationship traversal results"""
    if results:
        st.subheader("Relationship Traversal Results")
        for relationship in results:
            st.write(f"**{relationship['start']}** --[{relationship['type']}]--> **{relationship['end']}**")
            if 'properties' in relationship:
                st.json(relationship['properties'])
    else:
        st.warning("No relationships found.")

def display_path_results(path):
    """Display shortest path results"""
    if path:
        st.subheader("Shortest Path Found")
        path_str = " → ".join([node['name'] for node in path])
        st.write(f"**Path:** {path_str}")
        st.write(f"**Length:** {len(path) - 1} hops")
    else:
        st.warning("No path found between the entities.")

def display_community_results(communities):
    """Display community detection results"""
    if communities:
        st.subheader("Detected Communities")
        for i, community in enumerate(communities):
            with st.expander(f"Community {i+1} ({len(community)} members)"):
                st.write(", ".join(community))
    else:
        st.warning("No communities detected.")

def create_graph_visualization(viz_type):
    """Create interactive graph visualization"""
    if st.session_state.neo4j_rag is not None:
        graph_data = st.session_state.neo4j_rag.get_visualization_data(viz_type)
        
        # Create pyvis network
        net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
        
        # Add nodes and edges
        for node in graph_data['nodes']:
            net.add_node(node['id'], label=node['label'], color=node.get('color', '#97C2FC'))
        
        for edge in graph_data['edges']:
            net.add_edge(edge['from'], edge['to'], label=edge.get('label', ''))
        
        # Save and display
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, 'r') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=500)

def execute_sample_query(query, qtype):
    """Execute sample queries based on type"""
    if st.session_state.neo4j_rag is not None:
        if qtype == "semantic":
            results = st.session_state.neo4j_rag.semantic_search(query, top_k=3)
            display_search_results(results)
        elif qtype == "relationship":
            # Extract entity from query and traverse
            entity = "Engineering Manager"  # Simplified for demo
            results = st.session_state.neo4j_rag.traverse_relationships(entity, ["REPORTS_TO"], 1)
            display_relationship_results(results)
        # Add other query types as needed

def display_network_metrics():
    """Display network-level metrics"""
    if st.session_state.neo4j_rag is not None:
        metrics = st.session_state.neo4j_rag.calculate_network_metrics()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Density", f"{metrics.get('density', 0):.3f}")
        with col2:
            st.metric("Avg Clustering", f"{metrics.get('clustering', 0):.3f}")
        with col3:
            st.metric("Diameter", metrics.get('diameter', 0))

def display_centrality_analysis():
    """Display centrality analysis"""
    if st.session_state.neo4j_rag is not None:
        centrality = st.session_state.neo4j_rag.calculate_centrality()
        
        if centrality:
            df = pd.DataFrame(centrality)
            fig = px.bar(df, x='node', y='centrality', title="Node Centrality Scores")
            st.plotly_chart(fig, use_container_width=True)

def display_relationship_patterns():
    """Display relationship pattern analysis"""
    if st.session_state.neo4j_rag is not None:
        patterns = st.session_state.neo4j_rag.analyze_relationship_patterns()
        
        if patterns:
            df = pd.DataFrame(patterns)
            fig = px.pie(df, values='count', names='relationship_type', 
                        title="Relationship Type Distribution")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

