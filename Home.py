import streamlit as st
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

st.set_page_config(
    page_title="Google Graph RAG MVP",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🔍 Google Graph RAG MVP")
    st.markdown("### Advanced Retrieval-Augmented Generation with Google Technologies")
    
    st.markdown("""
    Welcome to the Google Graph RAG MVP - a comprehensive demonstration of advanced RAG techniques 
    using Google Cloud Platform technologies and various vector databases.
    
    ## 🎯 What is Graph RAG?
    
    Graph RAG combines traditional Retrieval-Augmented Generation with knowledge graphs to provide:
    - **Enhanced Context Understanding**: Leveraging relationships between entities
    - **Improved Answer Quality**: Using structured knowledge representations
    - **Better Information Retrieval**: Combining vector similarity with graph traversal
    
    ## 🏗️ Application Architecture
    
    This MVP demonstrates several RAG approaches:
    
    ### Traditional Vector RAG
    - **FAISS**: Facebook AI Similarity Search for efficient vector operations
    - **ChromaDB**: Open-source embedding database with built-in filtering
    - **LangChain Integration**: Streamlined RAG pipeline implementation
    
    ### Graph-Enhanced RAG
    - **Neo4j**: Graph database for relationship modeling
    - **Google Knowledge Graph API**: Access to Google's vast knowledge base
    - **Vertex AI Search & RAG Engine**: Google's enterprise RAG solution
    
    ## 📊 Use Case: HR Information System
    
    Our demonstration focuses on an HR domain where employees can:
    - Access payroll information
    - Find HR policies and procedures
    - Get directed to relevant HR contacts
    - Discover related information through graph relationships
    
    ## 🚀 Getting Started
    
    Navigate through the sidebar to explore different RAG implementations:
    
    1. **Basic Vector RAG** - Traditional embedding-based retrieval
    2. **ChromaDB RAG** - Enhanced filtering and metadata search
    3. **Neo4j Graph RAG** - Relationship-aware information retrieval
    4. **Google Knowledge Graph** - External knowledge integration
    5. **Vertex AI RAG Engine** - Enterprise-grade RAG solution
    
    Each section includes interactive examples, visualizations, and downloadable sample data.
    """)
    
    # Key Features Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔧 Technical Features
        - Multiple vector databases
        - Graph database integration
        - Google Cloud services
        - Interactive visualizations
        - Sample data generation
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Visualizations
        - Vector similarity heatmaps
        - Graph relationship diagrams
        - Performance comparisons
        - Query result analysis
        - Knowledge graph exploration
        """)
    
    with col3:
        st.markdown("""
        ### 💼 HR Use Cases
        - Employee information lookup
        - Policy document search
        - Contact directory
        - Organizational structure
        - Benefits information
        """)
    
    st.markdown("---")
    st.markdown("**Ready to explore?** Use the sidebar navigation to dive into specific RAG implementations!")

if __name__ == "__main__":
    main()

