import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from vertex_ai_rag import VertexAIRAG
    from hr_data_generator import HRDataGenerator
except ImportError:
    st.error("Utils modules not found. Please ensure the utils directory is properly set up.")
    st.stop()

st.set_page_config(
    page_title="Vertex AI RAG Engine",
    page_icon="🚀",
    layout="wide"
)

def main():
    st.title("🚀 Vertex AI Search & RAG Engine")
    st.markdown("### Enterprise-Grade RAG with Google Cloud")
    
    # Initialize session state
    if 'vertex_rag' not in st.session_state:
        st.session_state.vertex_rag = None
    if 'hr_data' not in st.session_state:
        st.session_state.hr_data = None
    if 'data_store_created' not in st.session_state:
        st.session_state.data_store_created = False
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Google Cloud settings
    st.sidebar.subheader("Google Cloud Settings")
    project_id = st.sidebar.text_input("Project ID", help="Your Google Cloud Project ID")
    location = st.sidebar.selectbox("Location", ["global", "us-central1", "europe-west1"])
    
    # Service account key
    service_account_key = st.sidebar.text_area(
        "Service Account Key (JSON)",
        height=100,
        help="Paste your service account key JSON here"
    )
    
    # Data store configuration
    st.sidebar.subheader("Data Store Configuration")
    data_store_id = st.sidebar.text_input("Data Store ID", "hr-knowledge-base")
    search_engine_id = st.sidebar.text_input("Search Engine ID", "hr-search-engine")
    
    # Data generation section
    st.sidebar.subheader("HR Data Generation")
    num_employees = st.sidebar.slider("Number of Employees", 10, 100, 50)
    num_policies = st.sidebar.slider("Number of HR Policies", 5, 30, 15)
    
    if st.sidebar.button("Generate HR Data"):
        with st.spinner("Generating comprehensive HR data..."):
            generator = HRDataGenerator()
            st.session_state.hr_data = generator.generate_enterprise_data(
                num_employees=num_employees,
                num_policies=num_policies
            )
            st.sidebar.success("Enterprise HR data generated!")
    
    # Initialize Vertex AI RAG
    if st.sidebar.button("Initialize Vertex AI RAG"):
        if project_id and service_account_key:
            with st.spinner("Initializing Vertex AI RAG Engine..."):
                try:
                    st.session_state.vertex_rag = VertexAIRAG(
                        project_id=project_id,
                        location=location,
                        service_account_key=service_account_key,
                        data_store_id=data_store_id,
                        search_engine_id=search_engine_id
                    )
                    st.sidebar.success("Vertex AI RAG initialized!")
                except Exception as e:
                    st.sidebar.error(f"Initialization failed: {str(e)}")
                    st.sidebar.info("Using mock Vertex AI for demonstration.")
                    st.session_state.vertex_rag = VertexAIRAG(
                        project_id="mock-project",
                        location=location,
                        mock_mode=True
                    )
        else:
            st.sidebar.warning("Please provide Project ID and Service Account Key!")
            st.sidebar.info("Using mock Vertex AI for demonstration.")
            st.session_state.vertex_rag = VertexAIRAG(
                project_id="mock-project",
                location=location,
                mock_mode=True
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enterprise RAG Interface")
        
        # Data store management
        if st.session_state.vertex_rag is not None:
            st.markdown("#### 📚 Data Store Management")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("Create Data Store"):
                    if st.session_state.hr_data is not None:
                        with st.spinner("Creating Vertex AI data store..."):
                            result = st.session_state.vertex_rag.create_data_store(st.session_state.hr_data)
                            if result['success']:
                                st.session_state.data_store_created = True
                                st.success("Data store created successfully!")
                            else:
                                st.error(f"Failed to create data store: {result['error']}")
                    else:
                        st.error("Please generate HR data first!")
            
            with col_b:
                if st.button("Update Data Store"):
                    if st.session_state.data_store_created:
                        with st.spinner("Updating data store..."):
                            result = st.session_state.vertex_rag.update_data_store(st.session_state.hr_data)
                            if result['success']:
                                st.success("Data store updated!")
                            else:
                                st.error(f"Update failed: {result['error']}")
                    else:
                        st.warning("Please create data store first!")
            
            with col_c:
                if st.button("Delete Data Store"):
                    if st.session_state.data_store_created:
                        with st.spinner("Deleting data store..."):
                            result = st.session_state.vertex_rag.delete_data_store()
                            if result['success']:
                                st.session_state.data_store_created = False
                                st.success("Data store deleted!")
                            else:
                                st.error(f"Deletion failed: {result['error']}")
        
        # Query interface
        if st.session_state.data_store_created:
            st.markdown("#### 🔍 Enterprise Search & RAG")
            
            # Query modes
            query_mode = st.selectbox(
                "Query Mode",
                ["Search Only", "RAG with Grounding", "Conversational RAG", "Multi-turn Chat"]
            )
            
            # Advanced settings
            with st.expander("Advanced RAG Settings"):
                col_x, col_y = st.columns(2)
                with col_x:
                    max_results = st.slider("Max Search Results", 1, 20, 10)
                    temperature = st.slider("Generation Temperature", 0.0, 1.0, 0.7)
                    max_tokens = st.slider("Max Output Tokens", 100, 2000, 500)
                with col_y:
                    use_grounding = st.checkbox("Enable Grounding", True)
                    include_citations = st.checkbox("Include Citations", True)
                    filter_safety = st.checkbox("Safety Filtering", True)
            
            # Sample enterprise queries
            sample_queries = [
                "What is our company's remote work policy and how does it compare to industry standards?",
                "Explain the process for requesting parental leave and the benefits available",
                "What are the performance review criteria for software engineers?",
                "How do I report a workplace safety incident?",
                "What training programs are available for leadership development?",
                "Explain our diversity and inclusion initiatives",
                "What are the steps to file a grievance with HR?",
                "How does our health insurance plan work?"
            ]
            
            selected_query = st.selectbox("Sample Enterprise Queries", [""] + sample_queries)
            query = st.text_area("Enter your question:", value=selected_query, height=100)
            
            if st.button("Execute Enterprise RAG", type="primary"):
                if query:
                    with st.spinner("Processing with Vertex AI RAG Engine..."):
                        results = st.session_state.vertex_rag.enterprise_rag_query(
                            query=query,
                            mode=query_mode,
                            max_results=max_results,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            use_grounding=use_grounding,
                            include_citations=include_citations,
                            filter_safety=filter_safety
                        )
                        
                        display_enterprise_results(results, query_mode)
                else:
                    st.warning("Please enter a query!")
    
    with col2:
        st.subheader("System Status")
        
        # Status indicators
        data_status = "✅ Ready" if st.session_state.hr_data is not None else "❌ Not Generated"
        vertex_status = "✅ Ready" if st.session_state.vertex_rag is not None else "❌ Not Initialized"
        store_status = "✅ Created" if st.session_state.data_store_created else "❌ Not Created"
        
        st.write(f"**HR Data:** {data_status}")
        st.write(f"**Vertex AI RAG:** {vertex_status}")
        st.write(f"**Data Store:** {store_status}")
        
        if st.session_state.vertex_rag is not None:
            st.subheader("Resource Usage")
            usage = st.session_state.vertex_rag.get_usage_metrics()
            
            st.metric("API Calls", usage.get('api_calls', 0))
            st.metric("Tokens Used", usage.get('tokens_used', 0))
            st.metric("Data Store Size", f"{usage.get('data_size_mb', 0)} MB")
        
        # Performance metrics
        if st.session_state.data_store_created:
            st.subheader("Performance Metrics")
            metrics = st.session_state.vertex_rag.get_performance_metrics()
            
            st.metric("Avg Response Time", f"{metrics.get('avg_response_time', 0):.2f}s")
            st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1%}")
            st.metric("Grounding Accuracy", f"{metrics.get('grounding_accuracy', 0):.1%}")
    
    # Analytics dashboard
    if st.session_state.data_store_created:
        st.subheader("📊 Enterprise Analytics Dashboard")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Query Analytics", "Performance Trends", "Content Analysis", "User Insights"])
        
        with tab1:
            display_query_analytics()
        
        with tab2:
            display_performance_trends()
        
        with tab3:
            display_content_analysis()
        
        with tab4:
            display_user_insights()
    
    # Data management section
    if st.session_state.vertex_rag is not None:
        st.subheader("🗄️ Enterprise Data Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Document Ingestion")
            uploaded_files = st.file_uploader(
                "Upload HR Documents",
                accept_multiple_files=True,
                type=['pdf', 'docx', 'txt', 'md']
            )
            
            if uploaded_files and st.button("Ingest Documents"):
                with st.spinner("Ingesting documents..."):
                    results = st.session_state.vertex_rag.ingest_documents(uploaded_files)
                    st.success(f"Ingested {len(results)} documents")
        
        with col2:
            st.markdown("#### Data Quality")
            if st.button("Run Quality Check"):
                quality_report = st.session_state.vertex_rag.check_data_quality()
                display_quality_report(quality_report)
        
        with col3:
            st.markdown("#### Export & Backup")
            if st.button("Export Data Store"):
                export_data = st.session_state.vertex_rag.export_data_store()
                st.download_button(
                    "Download Export",
                    export_data,
                    "vertex_ai_export.json",
                    "application/json"
                )
    
    # Technical details
    with st.expander("🔧 Vertex AI RAG Engine Technical Details"):
        st.markdown("""
        ### Vertex AI Search & RAG Features
        
        1. **Enterprise Scale**: Handle millions of documents with sub-second response times
        2. **Multi-modal Support**: Text, images, and structured data ingestion
        3. **Advanced Grounding**: Cite sources and provide evidence for answers
        4. **Safety & Compliance**: Built-in content filtering and safety checks
        5. **Real-time Updates**: Dynamic content updates without reindexing
        
        ### Key Capabilities
        - **Semantic Search**: Advanced understanding of query intent
        - **Conversational AI**: Multi-turn conversations with context retention
        - **Grounded Generation**: Answers backed by source documents
        - **Enterprise Security**: IAM integration and data governance
        - **Scalable Infrastructure**: Auto-scaling based on demand
        
        ### HR Use Cases
        - **Employee Self-Service**: Instant answers to policy questions
        - **Compliance Monitoring**: Ensure responses meet regulatory requirements
        - **Knowledge Management**: Centralized access to HR information
        - **Training & Onboarding**: Interactive learning experiences
        - **Analytics & Insights**: Understanding employee information needs
        
        ### Integration Benefits
        - **Google Workspace**: Seamless integration with existing tools
        - **Security**: Enterprise-grade security and compliance
        - **Monitoring**: Comprehensive logging and analytics
        - **Customization**: Tailored to specific organizational needs
        """)

def display_enterprise_results(results, mode):
    """Display enterprise RAG results with grounding and citations"""
    if results:
        st.subheader(f"Enterprise RAG Results ({mode})")
        
        # Main answer
        if 'answer' in results:
            st.markdown("#### 🎯 Generated Answer")
            st.write(results['answer'])
            
            # Confidence and safety scores
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{results.get('confidence', 0):.1%}")
            with col2:
                st.metric("Safety Score", f"{results.get('safety_score', 0):.1%}")
            with col3:
                st.metric("Grounding Score", f"{results.get('grounding_score', 0):.1%}")
        
        # Source documents
        if 'sources' in results:
            st.markdown("#### 📚 Source Documents")
            for i, source in enumerate(results['sources']):
                with st.expander(f"Source {i+1}: {source['title']} (Relevance: {source['relevance']:.3f})"):
                    st.write(source['content'])
                    if 'metadata' in source:
                        st.json(source['metadata'])
        
        # Citations
        if 'citations' in results:
            st.markdown("#### 📝 Citations")
            for citation in results['citations']:
                st.write(f"- {citation}")
        
        # Performance metrics
        if 'metrics' in results:
            st.markdown("#### ⚡ Performance Metrics")
            metrics = results['metrics']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Response Time", f"{metrics.get('response_time', 0):.2f}s")
            with col2:
                st.metric("Tokens Used", metrics.get('tokens_used', 0))
            with col3:
                st.metric("API Calls", metrics.get('api_calls', 0))
    else:
        st.warning("No results returned from Vertex AI RAG Engine.")

def display_query_analytics():
    """Display query analytics"""
    if st.session_state.vertex_rag is not None:
        analytics = st.session_state.vertex_rag.get_query_analytics()
        
        if analytics:
            # Query frequency
            query_freq = analytics.get('query_frequency', {})
            if query_freq:
                fig = px.bar(
                    x=list(query_freq.keys()),
                    y=list(query_freq.values()),
                    title="Most Frequent Queries"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Query categories
            categories = analytics.get('categories', {})
            if categories:
                fig = px.pie(
                    values=list(categories.values()),
                    names=list(categories.keys()),
                    title="Query Categories"
                )
                st.plotly_chart(fig, use_container_width=True)

def display_performance_trends():
    """Display performance trends over time"""
    if st.session_state.vertex_rag is not None:
        trends = st.session_state.vertex_rag.get_performance_trends()
        
        if trends:
            df = pd.DataFrame(trends)
            fig = px.line(
                df,
                x='timestamp',
                y=['response_time', 'confidence_score'],
                title="Performance Trends Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)

def display_content_analysis():
    """Display content analysis"""
    if st.session_state.vertex_rag is not None:
        analysis = st.session_state.vertex_rag.analyze_content()
        
        if analysis:
            # Document types
            doc_types = analysis.get('document_types', {})
            if doc_types:
                fig = px.bar(
                    x=list(doc_types.keys()),
                    y=list(doc_types.values()),
                    title="Document Types in Data Store"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Content freshness
            freshness = analysis.get('content_freshness', {})
            if freshness:
                fig = px.histogram(
                    x=list(freshness.keys()),
                    y=list(freshness.values()),
                    title="Content Freshness Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

def display_user_insights():
    """Display user behavior insights"""
    if st.session_state.vertex_rag is not None:
        insights = st.session_state.vertex_rag.get_user_insights()
        
        if insights:
            # User satisfaction
            satisfaction = insights.get('satisfaction_scores', [])
            if satisfaction:
                fig = px.histogram(
                    x=satisfaction,
                    title="User Satisfaction Distribution",
                    nbins=10
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Common failure points
            failures = insights.get('failure_points', {})
            if failures:
                st.subheader("Common Failure Points")
                for point, count in failures.items():
                    st.write(f"- {point}: {count} occurrences")

def display_quality_report(report):
    """Display data quality report"""
    if report:
        st.subheader("Data Quality Report")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Quality Score", f"{report.get('overall_score', 0):.1%}")
            st.metric("Documents Analyzed", report.get('total_documents', 0))
        with col2:
            st.metric("Issues Found", report.get('issues_count', 0))
            st.metric("Recommendations", report.get('recommendations_count', 0))
        
        # Quality issues
        issues = report.get('issues', [])
        if issues:
            st.subheader("Quality Issues")
            for issue in issues:
                st.warning(f"**{issue['type']}**: {issue['description']}")

if __name__ == "__main__":
    main()

