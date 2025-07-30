import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from chroma_rag import ChromaRAG
    from hr_data_generator import HRDataGenerator
except ImportError:
    st.error("Utils modules not found. Please ensure the utils directory is properly set up.")
    st.stop()

st.set_page_config(
    page_title="ChromaDB RAG",
    page_icon="🎨",
    layout="wide"
)

def main():
    st.title("🎨 ChromaDB RAG with Enhanced Filtering")
    st.markdown("### Advanced Metadata Filtering and Semantic Search")
    
    # Initialize session state
    if 'chroma_rag' not in st.session_state:
        st.session_state.chroma_rag = None
    if 'hr_data' not in st.session_state:
        st.session_state.hr_data = None
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Data generation section
    st.sidebar.subheader("Data Generation")
    num_employees = st.sidebar.slider("Number of Employees", 10, 100, 50)
    num_policies = st.sidebar.slider("Number of HR Policies", 5, 20, 10)
    
    if st.sidebar.button("Generate HR Data"):
        with st.spinner("Generating HR data..."):
            generator = HRDataGenerator()
            st.session_state.hr_data = generator.generate_comprehensive_data(
                num_employees=num_employees,
                num_policies=num_policies
            )
            st.sidebar.success("Data generated successfully!")
    
    # ChromaDB Configuration
    st.sidebar.subheader("ChromaDB Settings")
    collection_name = st.sidebar.text_input("Collection Name", "hr_documents")
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]
    )
    
    if st.sidebar.button("Initialize ChromaDB"):
        if st.session_state.hr_data is not None:
            with st.spinner("Initializing ChromaDB..."):
                st.session_state.chroma_rag = ChromaRAG(
                    collection_name=collection_name,
                    embedding_model=embedding_model
                )
                st.session_state.chroma_rag.build_collection(st.session_state.hr_data)
                st.sidebar.success("ChromaDB initialized!")
        else:
            st.sidebar.error("Please generate HR data first!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Advanced Query Interface")
        
        # Query configuration
        query_col1, query_col2 = st.columns(2)
        
        with query_col1:
            query = st.text_area("Enter your HR question:", height=100)
            top_k = st.slider("Number of Results", 1, 10, 5)
        
        with query_col2:
            st.subheader("Metadata Filters")
            
            # Department filter
            departments = ["All", "Engineering", "HR", "Finance", "Marketing", "Sales"]
            selected_dept = st.selectbox("Department", departments)
            
            # Document type filter
            doc_types = ["All", "Policy", "Procedure", "Form", "Guide"]
            selected_type = st.selectbox("Document Type", doc_types)
            
            # Priority filter
            priorities = ["All", "High", "Medium", "Low"]
            selected_priority = st.selectbox("Priority", priorities)
        
        # Sample queries with metadata context
        sample_queries = [
            "What is the vacation policy for engineering department?",
            "Show me high priority HR procedures",
            "Find finance department forms",
            "What are the benefits for marketing team?",
            "Emergency contact procedures"
        ]
        
        selected_query = st.selectbox("Sample Queries", [""] + sample_queries)
        if selected_query:
            query = selected_query
        
        if st.button("Search with Filters", type="primary"):
            if st.session_state.chroma_rag is not None and query:
                with st.spinner("Searching with filters..."):
                    # Build metadata filters
                    filters = {}
                    if selected_dept != "All":
                        filters["department"] = selected_dept
                    if selected_type != "All":
                        filters["doc_type"] = selected_type
                    if selected_priority != "All":
                        filters["priority"] = selected_priority
                    
                    results = st.session_state.chroma_rag.query_with_filters(
                        query, 
                        top_k=top_k,
                        filters=filters
                    )
                    
                    st.subheader("Filtered Search Results")
                    
                    if results:
                        # Display results with enhanced metadata
                        for i, result in enumerate(results):
                            with st.expander(f"Result {i+1} (Distance: {result['distance']:.3f})"):
                                st.write(result['content'])
                                
                                # Display metadata in a nice format
                                if 'metadata' in result:
                                    metadata_col1, metadata_col2 = st.columns(2)
                                    with metadata_col1:
                                        st.write("**Department:**", result['metadata'].get('department', 'N/A'))
                                        st.write("**Document Type:**", result['metadata'].get('doc_type', 'N/A'))
                                    with metadata_col2:
                                        st.write("**Priority:**", result['metadata'].get('priority', 'N/A'))
                                        st.write("**Last Updated:**", result['metadata'].get('last_updated', 'N/A'))
                        
                        # Visualize results by metadata
                        st.subheader("Result Analysis")
                        
                        # Distance distribution
                        distances = [r['distance'] for r in results]
                        fig_dist = px.bar(
                            x=list(range(1, len(distances)+1)),
                            y=distances,
                            title="Search Distance Scores (Lower = Better)",
                            labels={'x': 'Result Rank', 'y': 'Distance Score'}
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                        # Metadata distribution
                        if len(results) > 1:
                            metadata_df = pd.DataFrame([r['metadata'] for r in results if 'metadata' in r])
                            if not metadata_df.empty:
                                fig_meta = px.pie(
                                    metadata_df,
                                    names='department',
                                    title="Results by Department"
                                )
                                st.plotly_chart(fig_meta, use_container_width=True)
                    else:
                        st.warning("No results found with the specified filters.")
            else:
                st.warning("Please initialize ChromaDB and enter a query!")
    
    with col2:
        st.subheader("System Status")
        
        # Status indicators
        data_status = "✅ Ready" if st.session_state.hr_data is not None else "❌ Not Generated"
        chroma_status = "✅ Ready" if st.session_state.chroma_rag is not None else "❌ Not Initialized"
        
        st.write(f"**HR Data:** {data_status}")
        st.write(f"**ChromaDB:** {chroma_status}")
        
        if st.session_state.chroma_rag is not None:
            st.subheader("Collection Info")
            collection_info = st.session_state.chroma_rag.get_collection_info()
            st.metric("Documents", collection_info.get('count', 0))
            st.write(f"**Collection:** {collection_name}")
            st.write(f"**Model:** {embedding_model}")
        
        # Collection statistics
        if st.session_state.hr_data is not None:
            st.subheader("Data Distribution")
            
            # Department distribution
            all_docs = []
            for doc_list in st.session_state.hr_data.values():
                if isinstance(doc_list, list):
                    all_docs.extend(doc_list)
            
            if all_docs:
                dept_counts = {}
                for doc in all_docs:
                    if isinstance(doc, dict) and 'department' in doc:
                        dept = doc['department']
                        dept_counts[dept] = dept_counts.get(dept, 0) + 1
                
                if dept_counts:
                    fig_dept = px.pie(
                        values=list(dept_counts.values()),
                        names=list(dept_counts.keys()),
                        title="Documents by Department"
                    )
                    st.plotly_chart(fig_dept, use_container_width=True)
    
    # Advanced features section
    st.subheader("🔍 Advanced ChromaDB Features")
    
    if st.session_state.chroma_rag is not None:
        tab1, tab2, tab3 = st.tabs(["Similarity Search", "Metadata Analysis", "Collection Management"])
        
        with tab1:
            st.markdown("### Semantic Similarity Exploration")
            similarity_query = st.text_input("Enter a concept to explore:")
            if similarity_query and st.button("Find Similar Concepts"):
                similar_docs = st.session_state.chroma_rag.find_similar_documents(similarity_query, top_k=10)
                
                if similar_docs:
                    # Create similarity heatmap
                    similarities = [1 - doc['distance'] for doc in similar_docs]
                    labels = [f"Doc {i+1}" for i in range(len(similar_docs))]
                    
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=[similarities],
                        x=labels,
                        y=['Similarity'],
                        colorscale='Viridis'
                    ))
                    fig_heatmap.update_layout(title="Document Similarity Scores")
                    st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab2:
            st.markdown("### Metadata Analysis")
            if st.button("Analyze Collection Metadata"):
                metadata_analysis = st.session_state.chroma_rag.analyze_metadata()
                
                for field, analysis in metadata_analysis.items():
                    st.subheader(f"{field.title()} Distribution")
                    if isinstance(analysis, dict):
                        fig = px.bar(
                            x=list(analysis.keys()),
                            y=list(analysis.values()),
                            title=f"{field.title()} Frequency"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Collection Management")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Reset Collection"):
                    st.session_state.chroma_rag.reset_collection()
                    st.success("Collection reset successfully!")
            
            with col2:
                if st.button("Export Collection"):
                    export_data = st.session_state.chroma_rag.export_collection()
                    st.download_button(
                        "Download Collection Data",
                        export_data,
                        "chroma_collection.json",
                        "application/json"
                    )
    
    # Technical details
    with st.expander("🔧 ChromaDB Technical Details"):
        st.markdown("""
        ### ChromaDB Advantages
        
        1. **Metadata Filtering**: Rich filtering capabilities on document metadata
        2. **Built-in Embeddings**: Automatic embedding generation with multiple model options
        3. **Persistent Storage**: Data persists between sessions
        4. **Scalability**: Handles large collections efficiently
        5. **API Flexibility**: Rich query API with multiple search modes
        
        ### Enhanced Features
        - **Hybrid Search**: Combine semantic similarity with metadata filters
        - **Collection Management**: Easy data organization and versioning
        - **Real-time Updates**: Dynamic document addition and removal
        - **Analytics**: Built-in collection analysis and statistics
        """)

if __name__ == "__main__":
    main()

