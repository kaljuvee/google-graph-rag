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
    from vector_rag import VectorRAG
    from hr_data_generator import HRDataGenerator
except ImportError:
    st.error("Utils modules not found. Please ensure the utils directory is properly set up.")
    st.stop()

st.set_page_config(
    page_title="Basic Vector RAG",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("🔍 Basic Vector RAG with FAISS")
    st.markdown("### Traditional Embedding-Based Retrieval for HR Information")
    
    # Initialize session state
    if 'vector_rag' not in st.session_state:
        st.session_state.vector_rag = None
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
    
    # RAG Configuration
    st.sidebar.subheader("RAG Settings")
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]
    )
    chunk_size = st.sidebar.slider("Chunk Size", 100, 1000, 500)
    top_k = st.sidebar.slider("Top K Results", 1, 10, 5)
    
    if st.sidebar.button("Initialize Vector RAG"):
        if st.session_state.hr_data is not None:
            with st.spinner("Initializing Vector RAG..."):
                st.session_state.vector_rag = VectorRAG(
                    embedding_model=embedding_model,
                    chunk_size=chunk_size
                )
                st.session_state.vector_rag.build_index(st.session_state.hr_data)
                st.sidebar.success("Vector RAG initialized!")
        else:
            st.sidebar.error("Please generate HR data first!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Query Interface")
        
        # Sample queries
        sample_queries = [
            "What is the vacation policy?",
            "How do I submit a timesheet?",
            "Who is the HR manager for engineering?",
            "What are the health insurance benefits?",
            "How to request parental leave?"
        ]
        
        selected_query = st.selectbox("Sample Queries", [""] + sample_queries)
        query = st.text_area("Enter your HR question:", value=selected_query, height=100)
        
        if st.button("Search", type="primary"):
            if st.session_state.vector_rag is not None and query:
                with st.spinner("Searching..."):
                    results = st.session_state.vector_rag.query(query, top_k=top_k)
                    
                    st.subheader("Search Results")
                    
                    # Display results
                    for i, result in enumerate(results):
                        with st.expander(f"Result {i+1} (Score: {result['score']:.3f})"):
                            st.write(result['content'])
                            if 'metadata' in result:
                                st.json(result['metadata'])
                    
                    # Visualize similarity scores
                    if results:
                        scores = [r['score'] for r in results]
                        fig = px.bar(
                            x=list(range(1, len(scores)+1)),
                            y=scores,
                            title="Similarity Scores",
                            labels={'x': 'Result Rank', 'y': 'Similarity Score'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please initialize Vector RAG and enter a query!")
    
    with col2:
        st.subheader("System Status")
        
        # Status indicators
        data_status = "✅ Ready" if st.session_state.hr_data is not None else "❌ Not Generated"
        rag_status = "✅ Ready" if st.session_state.vector_rag is not None else "❌ Not Initialized"
        
        st.write(f"**HR Data:** {data_status}")
        st.write(f"**Vector RAG:** {rag_status}")
        
        if st.session_state.hr_data is not None:
            st.subheader("Data Overview")
            data_stats = {
                "Employees": len(st.session_state.hr_data.get('employees', [])),
                "Policies": len(st.session_state.hr_data.get('policies', [])),
                "Documents": len(st.session_state.hr_data.get('documents', []))
            }
            
            for key, value in data_stats.items():
                st.metric(key, value)
        
        if st.session_state.vector_rag is not None:
            st.subheader("RAG Configuration")
            st.write(f"**Model:** {embedding_model}")
            st.write(f"**Chunk Size:** {chunk_size}")
            st.write(f"**Index Size:** {st.session_state.vector_rag.get_index_size()}")
    
    # Data download section
    if st.session_state.hr_data is not None:
        st.subheader("📊 Sample Data Download")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.hr_data.get('employees'):
                df_employees = pd.DataFrame(st.session_state.hr_data['employees'])
                st.download_button(
                    "Download Employee Data",
                    df_employees.to_csv(index=False),
                    "employees.csv",
                    "text/csv"
                )
        
        with col2:
            if st.session_state.hr_data.get('policies'):
                df_policies = pd.DataFrame(st.session_state.hr_data['policies'])
                st.download_button(
                    "Download Policy Data",
                    df_policies.to_csv(index=False),
                    "policies.csv",
                    "text/csv"
                )
        
        with col3:
            if st.session_state.hr_data.get('documents'):
                df_documents = pd.DataFrame(st.session_state.hr_data['documents'])
                st.download_button(
                    "Download Document Data",
                    df_documents.to_csv(index=False),
                    "documents.csv",
                    "text/csv"
                )
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        ### How Vector RAG Works
        
        1. **Document Processing**: HR documents are split into chunks of configurable size
        2. **Embedding Generation**: Each chunk is converted to a vector using sentence transformers
        3. **Index Building**: FAISS creates an efficient similarity search index
        4. **Query Processing**: User queries are embedded and matched against the index
        5. **Result Ranking**: Results are ranked by cosine similarity scores
        
        ### FAISS Advantages
        - **Speed**: Optimized for large-scale similarity search
        - **Memory Efficiency**: Compressed vector representations
        - **Scalability**: Handles millions of vectors efficiently
        - **Flexibility**: Multiple index types for different use cases
        """)

if __name__ == "__main__":
    main()

