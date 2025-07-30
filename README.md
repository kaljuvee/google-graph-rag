# Google Graph RAG MVP

A comprehensive Streamlit application demonstrating Advanced Retrieval-Augmented Generation (RAG) methods with emphasis on Google Cloud Platform technologies and graph-based knowledge systems.

## 🎯 Overview

This MVP showcases various RAG implementations including:
- **Vector-based RAG** with FAISS and sentence transformers
- **ChromaDB RAG** with advanced filtering and metadata search
- **Neo4j Graph RAG** with relationship traversal and community detection
- **Google Knowledge Graph** integration for external knowledge enhancement
- **Vertex AI RAG Engine** for enterprise-grade RAG solutions

The application uses HR domain data as a use case, enabling employees to access information around HR policies, payroll, benefits, and organizational structure while providing intelligent routing to relevant contacts.

## 🏗️ Architecture

```
google-graph-rag/
├── Home.py                 # Main Streamlit application entry point
├── pages/                  # Streamlit pages for different RAG implementations
│   ├── 1_Basic_Vector_RAG.py
│   ├── 2_ChromaDB_RAG.py
│   ├── 3_Neo4j_Graph_RAG.py
│   ├── 4_Google_Knowledge_Graph.py
│   └── 5_Vertex_AI_RAG_Engine.py
├── utils/                  # Business logic and RAG implementations
│   ├── hr_data_generator.py
│   ├── vector_rag.py
│   ├── chroma_rag.py
│   ├── neo4j_rag.py
│   ├── google_kg_rag.py
│   ├── vertex_ai_rag.py
│   └── graph_visualizer.py
├── tests/                  # Test suite for all RAG implementations
│   ├── test_vector_rag.py
│   ├── test_chroma_rag.py
│   ├── test_neo4j_rag.py
│   └── run_all_tests.py
├── test-results/          # Test results and reports
├── data/                  # Sample data and exports
├── assets/               # Static assets and images
└── requirements.txt      # Python dependencies
```

## 🚀 Features

### Core RAG Implementations

1. **Basic Vector RAG**
   - FAISS vector database integration
   - Sentence transformer embeddings
   - Semantic similarity search
   - Real-time indexing and querying

2. **ChromaDB RAG**
   - Advanced metadata filtering
   - Collection management
   - Document similarity analysis
   - Export/import capabilities

3. **Neo4j Graph RAG**
   - Knowledge graph construction
   - Relationship traversal
   - Community detection algorithms
   - Path finding and centrality analysis

4. **Google Knowledge Graph**
   - External knowledge enhancement
   - Entity recognition and linking
   - Hybrid search (internal + external)
   - API integration with caching

5. **Vertex AI RAG Engine**
   - Enterprise-grade RAG solution
   - Document ingestion and indexing
   - Grounded response generation
   - Safety filtering and citations

### Visualization Features

- Interactive network graphs
- Performance metrics dashboards
- Query analytics and insights
- Knowledge source distribution
- Relationship pattern analysis

### Testing Framework

- Comprehensive test suite for all RAG implementations
- Performance benchmarking
- Sample data generation
- Automated test reporting

## 📋 Prerequisites

- Python 3.8 or higher
- Git
- Google Cloud Platform account (for Google services)
- Neo4j database (optional, embedded mode available)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/kaljuvee/google-graph-rag.git
cd google-graph-rag
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Additional Dependencies

```bash
# For vector operations
pip install sentence-transformers faiss-cpu

# For graph database
pip install chromadb

# For visualization
pip install networkx plotly

# For Neo4j (optional)
pip install neo4j
```

## ⚙️ Configuration

### Google Cloud Services Setup

1. **Google Knowledge Graph API**
   ```bash
   # Set your API key
   export GOOGLE_KG_API_KEY="your_api_key_here"
   ```

2. **Vertex AI Setup**
   ```bash
   # Set your project ID
   export GOOGLE_CLOUD_PROJECT="your_project_id"
   
   # Authenticate with Google Cloud
   gcloud auth application-default login
   ```

### Environment Variables

Create a `.env` file in the project root:

```env
# Google Cloud Configuration
GOOGLE_KG_API_KEY=your_google_kg_api_key
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1

# Neo4j Configuration (if using external Neo4j)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Application Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 🏃‍♂️ Running the Application

### 1. Start the Streamlit Application

```bash
streamlit run Home.py
```

The application will be available at `http://localhost:8501`

### 2. Alternative: Run with Custom Configuration

```bash
streamlit run Home.py --server.port 8501 --server.address 0.0.0.0
```

### 3. Docker Deployment (Optional)

```bash
# Build Docker image
docker build -t google-graph-rag .

# Run container
docker run -p 8501:8501 google-graph-rag
```

## 🧪 Testing

### Run All Tests

```bash
cd tests
python run_all_tests.py
```

### Run Individual Tests

```bash
# Test Vector RAG
python test_vector_rag.py

# Test ChromaDB RAG
python test_chroma_rag.py

# Test Neo4j Graph RAG
python test_neo4j_rag.py
```

### View Test Results

Test results are saved in the `test-results/` directory:
- `comprehensive_test_report.json` - Detailed test results
- `test_summary_report.txt` - Human-readable summary
- Individual test result files for each RAG implementation

## 📊 Usage Examples

### Basic Vector Search

```python
from utils.vector_rag import VectorRAG
from utils.hr_data_generator import HRDataGenerator

# Generate sample data
generator = HRDataGenerator()
hr_data = generator.generate_comprehensive_data()

# Initialize Vector RAG
vector_rag = VectorRAG()
vector_rag.build_index(hr_data)

# Query the system
results = vector_rag.query("What is the vacation policy?", top_k=5)
```

### Graph-based Search

```python
from utils.neo4j_rag import Neo4jRAG

# Initialize Neo4j RAG
neo4j_rag = Neo4jRAG(embedded=True)
neo4j_rag.build_graph(hr_data)

# Semantic search with graph context
results = neo4j_rag.semantic_search("engineering team", max_depth=2)

# Find relationships
relationships = neo4j_rag.traverse_relationships("Engineering", ["WORKS_IN", "MANAGES"])
```

### Google Knowledge Graph Integration

```python
from utils.google_kg_rag import GoogleKnowledgeGraphRAG

# Initialize with API key
kg_rag = GoogleKnowledgeGraphRAG(api_key="your_api_key", hr_data=hr_data)

# Hybrid search
results = kg_rag.hybrid_search("employment law", mode="Hybrid (Internal + External)")
```

## 🎨 User Interface

The Streamlit application provides an intuitive interface with:

1. **Home Page** - Overview and navigation
2. **Basic Vector RAG** - Vector similarity search interface
3. **ChromaDB RAG** - Advanced filtering and metadata search
4. **Neo4j Graph RAG** - Graph visualization and relationship exploration
5. **Google Knowledge Graph** - External knowledge integration
6. **Vertex AI RAG Engine** - Enterprise RAG capabilities

Each page includes:
- Interactive query interface
- Real-time results display
- Visualization components
- Performance metrics
- Export capabilities

## 📈 Performance Metrics

The application tracks various performance metrics:

- **Response Time** - Query processing speed
- **Relevance Scores** - Result quality metrics
- **Cache Hit Rates** - Efficiency measurements
- **API Usage** - External service utilization
- **Memory Usage** - Resource consumption

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **Google API Authentication**
   ```bash
   # Check API key configuration
   echo $GOOGLE_KG_API_KEY
   
   # Verify Google Cloud authentication
   gcloud auth list
   ```

3. **Neo4j Connection Issues**
   ```bash
   # For embedded mode, no setup required
   # For external Neo4j, check connection settings
   ```

4. **Memory Issues with Large Datasets**
   ```python
   # Reduce chunk size or batch size
   vector_rag = VectorRAG(chunk_size=200)
   ```

### Performance Optimization

1. **Vector Search Optimization**
   - Use appropriate embedding models
   - Optimize chunk sizes
   - Implement caching strategies

2. **Graph Database Optimization**
   - Create appropriate indexes
   - Optimize query patterns
   - Use connection pooling

3. **API Rate Limiting**
   - Implement exponential backoff
   - Use caching for repeated queries
   - Monitor API quotas

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Ensure all tests pass

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Cloud Platform for providing advanced AI/ML services
- Streamlit for the excellent web application framework
- The open-source community for various RAG and vector database libraries
- Neo4j for graph database capabilities
- ChromaDB for vector storage and retrieval

## 📞 Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Contact the development team
- Check the documentation and examples

## 🔮 Future Enhancements

- Multi-modal RAG with image and document support
- Advanced prompt engineering techniques
- Integration with additional Google Cloud services
- Real-time collaboration features
- Enhanced security and access controls
- Mobile-responsive design improvements

---

**Note**: This is a demonstration MVP. For production use, ensure proper security measures, error handling, and scalability considerations are implemented.

