# Playground Demo Scripts

This directory contains demo scripts that showcase different RAG (Retrieval-Augmented Generation) approaches using the Google Graph RAG MVP system.

## 📁 Files

- `basic_vector_demo.py` - Basic vector search using FAISS
- `graph_search_demo.py` - Graph-based search using Neo4j
- `google_kg_demo.py` - Google Knowledge Graph integration
- `comprehensive_demo.py` - Runs all demos and compares performance

## 🚀 Quick Start

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Set up Google API key for Knowledge Graph demo:
```bash
export GOOGLE_API_KEY='your_api_key_here'
```

### Running Individual Demos

#### Basic Vector Search
```bash
cd playground
python basic_vector_demo.py
```

#### Graph-based Search
```bash
cd playground
python graph_search_demo.py
```

#### Google Knowledge Graph
```bash
cd playground
python google_kg_demo.py
```

#### Comprehensive Demo (All Approaches)
```bash
cd playground
python comprehensive_demo.py
```

## 📊 Output

All demos generate data in the `../test-data/` directory:

### Sample Data Files
- `hr_sample_data.json` - Generated HR data for vector search
- `hr_graph_data.json` - Generated HR data for graph search
- `hr_kg_data.json` - Generated HR data for knowledge graph
- `comprehensive_hr_data.json` - Comprehensive HR dataset

### Results Files
- `vector_search_results.json` - Vector search query results
- `graph_search_results.json` - Graph search query results
- `google_kg_results.json` - Knowledge graph search results
- `comprehensive_*.json` - Results from comprehensive demo

### Analysis Files
- `rag_comparison.json` - Performance comparison between approaches
- `demo_summary.json` - Overall demo summary and metrics

## 🔍 What Each Demo Does

### Basic Vector Search Demo
- Generates sample HR data (employees, policies, departments)
- Initializes FAISS vector database with embeddings
- Runs semantic search queries on HR policies and procedures
- Saves results with similarity scores

### Graph Search Demo
- Creates Neo4j graph database with HR relationships
- Performs semantic search with graph context
- Traverses relationships between entities
- Demonstrates graph-based information retrieval

### Google Knowledge Graph Demo
- Combines internal HR data with external knowledge
- Tests hybrid search modes (Internal, External, Hybrid)
- Searches for external entities and concepts
- Shows integration with Google's knowledge base

### Comprehensive Demo
- Runs all three approaches on the same dataset
- Compares performance metrics and result quality
- Generates summary reports and comparisons
- Provides insights into different RAG strategies

## 🛠️ Customization

### Modifying Sample Data
Edit the demo scripts to change:
- Number of employees (`num_employees`)
- Number of policies (`num_policies`)
- Test queries
- Search parameters

### Adding New Queries
Add your own test queries to the `test_queries` lists in each demo script.

### Changing Models
Modify the embedding model in vector demos:
```python
vector_rag = VectorRAG(
    embedding_model="sentence-transformers/all-mpnet-base-v2",  # Change model
    chunk_size=500
)
```

## 📈 Performance Analysis

The comprehensive demo provides:
- Execution time comparison
- Result count analysis
- Query performance metrics
- Data generation statistics

## 🔧 Troubleshooting

### Neo4j Issues
- Ensure Neo4j is running or use embedded mode
- Check Neo4j connection settings in `utils/neo4j_rag.py`

### Google API Issues
- Verify your Google API key is set correctly
- Check API quotas and permissions
- The demo will run in simulation mode without a valid key

### Memory Issues
- Reduce `num_employees` and `num_policies` in data generation
- Use smaller embedding models
- Close Neo4j connections properly

## 📝 Notes

- All demos are designed to be self-contained
- Results are saved as JSON for easy analysis
- The test-data directory is created automatically
- Each demo can be run independently
- Error handling is included for graceful failures 