import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from google_kg_rag import GoogleKnowledgeGraphRAG
    from hr_data_generator import HRDataGenerator
except ImportError:
    st.error("Utils modules not found. Please ensure the utils directory is properly set up.")
    st.stop()

st.set_page_config(
    page_title="Google Knowledge Graph",
    page_icon="🌐",
    layout="wide"
)

def main():
    st.title("🌐 Google Knowledge Graph Integration")
    st.markdown("### External Knowledge Enhancement for HR Queries")
    
    # Initialize session state
    if 'gkg_rag' not in st.session_state:
        st.session_state.gkg_rag = None
    if 'hr_data' not in st.session_state:
        st.session_state.hr_data = None
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Google Knowledge Graph API settings
    st.sidebar.subheader("Google KG API Settings")
    api_key = st.sidebar.text_input("Google API Key", type="password", 
                                   help="Get your API key from Google Cloud Console")
    
    # Data generation section
    st.sidebar.subheader("Internal HR Data")
    num_employees = st.sidebar.slider("Number of Employees", 10, 100, 30)
    
    if st.sidebar.button("Generate HR Data"):
        with st.spinner("Generating HR data..."):
            generator = HRDataGenerator()
            st.session_state.hr_data = generator.generate_comprehensive_data(
                num_employees=num_employees,
                num_policies=10
            )
            st.sidebar.success("HR data generated successfully!")
    
    # Initialize Google KG RAG
    if st.sidebar.button("Initialize Google KG RAG"):
        if api_key:
            with st.spinner("Initializing Google Knowledge Graph RAG..."):
                try:
                    st.session_state.gkg_rag = GoogleKnowledgeGraphRAG(
                        api_key=api_key,
                        hr_data=st.session_state.hr_data
                    )
                    st.sidebar.success("Google KG RAG initialized!")
                except Exception as e:
                    st.sidebar.error(f"Initialization failed: {str(e)}")
                    st.sidebar.info("Using mock Google KG for demonstration.")
                    st.session_state.gkg_rag = GoogleKnowledgeGraphRAG(
                        api_key="mock",
                        hr_data=st.session_state.hr_data,
                        mock_mode=True
                    )
        else:
            st.sidebar.warning("Please provide a Google API key!")
            st.sidebar.info("Using mock Google KG for demonstration.")
            st.session_state.gkg_rag = GoogleKnowledgeGraphRAG(
                api_key="mock",
                hr_data=st.session_state.hr_data,
                mock_mode=True
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Hybrid Knowledge Query Interface")
        
        # Query modes
        query_mode = st.selectbox(
            "Query Mode",
            ["Hybrid (Internal + External)", "Internal Only", "External Only", "Entity Enrichment"]
        )
        
        # Sample queries that benefit from external knowledge
        sample_queries = [
            "What are the latest labor law changes affecting our vacation policy?",
            "How do our benefits compare to industry standards?",
            "What are the current market salary ranges for software engineers?",
            "Tell me about GDPR compliance requirements for HR data",
            "What are best practices for remote work policies?",
            "How do other companies handle parental leave?",
            "What are the tax implications of our stock option plan?",
            "Industry trends in employee wellness programs"
        ]
        
        selected_query = st.selectbox("Sample Queries", [""] + sample_queries)
        query = st.text_area("Enter your question:", value=selected_query, height=100)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            col_a, col_b = st.columns(2)
            with col_a:
                confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.7)
                max_external_results = st.slider("Max External Results", 1, 10, 5)
            with col_b:
                entity_types = st.multiselect(
                    "Entity Types to Search",
                    ["Person", "Organization", "Place", "Thing", "Event"],
                    default=["Organization", "Thing"]
                )
                languages = st.multiselect("Languages", ["en", "es", "fr", "de"], default=["en"])
        
        if st.button("Search with Google Knowledge Graph", type="primary"):
            if st.session_state.gkg_rag is not None and query:
                with st.spinner("Searching internal and external knowledge..."):
                    results = st.session_state.gkg_rag.hybrid_search(
                        query=query,
                        mode=query_mode,
                        confidence_threshold=confidence_threshold,
                        max_external_results=max_external_results,
                        entity_types=entity_types,
                        languages=languages
                    )
                    
                    display_hybrid_results(results, query_mode)
            else:
                st.warning("Please initialize the system and enter a query!")
    
    with col2:
        st.subheader("System Status")
        
        # Status indicators
        data_status = "✅ Ready" if st.session_state.hr_data is not None else "❌ Not Generated"
        gkg_status = "✅ Ready" if st.session_state.gkg_rag is not None else "❌ Not Initialized"
        
        st.write(f"**HR Data:** {data_status}")
        st.write(f"**Google KG RAG:** {gkg_status}")
        
        if st.session_state.gkg_rag is not None:
            st.subheader("API Usage")
            usage_stats = st.session_state.gkg_rag.get_usage_stats()
            st.metric("API Calls Made", usage_stats.get('api_calls', 0))
            st.metric("Entities Found", usage_stats.get('entities_found', 0))
            st.metric("Cache Hits", usage_stats.get('cache_hits', 0))
        
        # Knowledge source distribution
        if st.session_state.gkg_rag is not None:
            st.subheader("Knowledge Sources")
            sources = st.session_state.gkg_rag.get_knowledge_sources()
            
            if sources:
                fig = px.pie(
                    values=list(sources.values()),
                    names=list(sources.keys()),
                    title="Knowledge Source Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Entity exploration section
    if st.session_state.gkg_rag is not None:
        st.subheader("🔍 Entity Exploration")
        
        tab1, tab2, tab3 = st.tabs(["Entity Search", "Entity Details", "Related Entities"])
        
        with tab1:
            entity_query = st.text_input("Search for entities:")
            if entity_query and st.button("Find Entities"):
                entities = st.session_state.gkg_rag.search_entities(entity_query)
                display_entity_results(entities)
        
        with tab2:
            entity_id = st.text_input("Entity ID (from search results):")
            if entity_id and st.button("Get Entity Details"):
                details = st.session_state.gkg_rag.get_entity_details(entity_id)
                display_entity_details(details)
        
        with tab3:
            if st.button("Find Related Entities"):
                if entity_id:
                    related = st.session_state.gkg_rag.get_related_entities(entity_id)
                    display_related_entities(related)
    
    # Knowledge graph visualization
    if st.session_state.gkg_rag is not None:
        st.subheader("📊 Knowledge Graph Visualization")
        
        if st.button("Generate Knowledge Graph"):
            with st.spinner("Creating knowledge graph visualization..."):
                graph_data = st.session_state.gkg_rag.create_knowledge_graph()
                display_knowledge_graph(graph_data)
    
    # Analytics section
    if st.session_state.gkg_rag is not None:
        st.subheader("📈 Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Query performance analysis
            performance_data = st.session_state.gkg_rag.get_performance_metrics()
            if performance_data:
                fig = px.line(
                    performance_data,
                    x='timestamp',
                    y='response_time',
                    title="Query Response Time"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Entity type distribution
            entity_types_data = st.session_state.gkg_rag.get_entity_type_distribution()
            if entity_types_data:
                fig = px.bar(
                    x=list(entity_types_data.keys()),
                    y=list(entity_types_data.values()),
                    title="Entity Types Found"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Export and download section
    if st.session_state.gkg_rag is not None:
        st.subheader("💾 Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Query History"):
                history = st.session_state.gkg_rag.export_query_history()
                st.download_button(
                    "Download Query History",
                    history,
                    "query_history.json",
                    "application/json"
                )
        
        with col2:
            if st.button("Export Entity Cache"):
                cache = st.session_state.gkg_rag.export_entity_cache()
                st.download_button(
                    "Download Entity Cache",
                    cache,
                    "entity_cache.json",
                    "application/json"
                )
        
        with col3:
            if st.button("Export Knowledge Graph"):
                kg_data = st.session_state.gkg_rag.export_knowledge_graph()
                st.download_button(
                    "Download Knowledge Graph",
                    kg_data,
                    "knowledge_graph.json",
                    "application/json"
                )
    
    # Technical details
    with st.expander("🔧 Google Knowledge Graph Technical Details"):
        st.markdown("""
        ### Google Knowledge Graph API Features
        
        1. **Entity Recognition**: Automatic identification of entities in queries
        2. **Structured Data**: Rich metadata and relationships for entities
        3. **Multi-language Support**: Query and results in multiple languages
        4. **Real-time Data**: Access to up-to-date information
        5. **Confidence Scores**: Quality indicators for search results
        
        ### Integration Benefits
        - **External Context**: Supplement internal HR data with world knowledge
        - **Industry Insights**: Access to market trends and best practices
        - **Regulatory Information**: Current laws and compliance requirements
        - **Competitive Intelligence**: Industry standards and benchmarks
        
        ### Use Cases in HR
        - **Policy Benchmarking**: Compare policies against industry standards
        - **Compliance Checking**: Verify against current regulations
        - **Market Research**: Salary and benefits benchmarking
        - **Training Content**: Access to educational resources
        """)

def display_hybrid_results(results, mode):
    """Display hybrid search results from internal and external sources"""
    if results:
        st.subheader(f"Hybrid Search Results ({mode})")
        
        # Separate internal and external results
        internal_results = results.get('internal', [])
        external_results = results.get('external', [])
        
        if internal_results:
            st.markdown("#### 🏢 Internal HR Knowledge")
            for i, result in enumerate(internal_results):
                with st.expander(f"Internal Result {i+1} (Score: {result['score']:.3f})"):
                    st.write(result['content'])
                    if 'metadata' in result:
                        st.json(result['metadata'])
        
        if external_results:
            st.markdown("#### 🌐 External Knowledge Graph")
            for i, result in enumerate(external_results):
                with st.expander(f"External Result {i+1} (Confidence: {result['confidence']:.3f})"):
                    st.write(f"**Name:** {result['name']}")
                    st.write(f"**Description:** {result['description']}")
                    if 'types' in result:
                        st.write(f"**Types:** {', '.join(result['types'])}")
                    if 'url' in result:
                        st.write(f"**More Info:** [Link]({result['url']})")
        
        # Comparison visualization
        if internal_results and external_results:
            st.subheader("Source Comparison")
            source_data = {
                'Source': ['Internal'] * len(internal_results) + ['External'] * len(external_results),
                'Score': [r['score'] for r in internal_results] + [r['confidence'] for r in external_results]
            }
            df = pd.DataFrame(source_data)
            fig = px.box(df, x='Source', y='Score', title="Score Distribution by Source")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No results found.")

def display_entity_results(entities):
    """Display entity search results"""
    if entities:
        st.subheader("Found Entities")
        for entity in entities:
            with st.expander(f"{entity['name']} ({entity['id']})"):
                st.write(f"**Description:** {entity.get('description', 'N/A')}")
                st.write(f"**Types:** {', '.join(entity.get('types', []))}")
                st.write(f"**Score:** {entity.get('score', 0):.3f}")
    else:
        st.warning("No entities found.")

def display_entity_details(details):
    """Display detailed entity information"""
    if details:
        st.subheader("Entity Details")
        st.json(details)
    else:
        st.warning("Entity details not found.")

def display_related_entities(related):
    """Display related entities"""
    if related:
        st.subheader("Related Entities")
        for entity in related:
            st.write(f"**{entity['name']}** - {entity.get('description', 'N/A')}")
    else:
        st.warning("No related entities found.")

def display_knowledge_graph(graph_data):
    """Display knowledge graph visualization"""
    if graph_data:
        st.subheader("Knowledge Graph")
        # This would typically use a graph visualization library
        st.json(graph_data)  # Simplified for demo
    else:
        st.warning("No graph data available.")

if __name__ == "__main__":
    main()

