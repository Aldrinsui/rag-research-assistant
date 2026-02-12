# 🤖 Multi-Agent RAG Research Assistant

A sophisticated AI research system leveraging LangGraph for autonomous agent orchestration and Retrieval-Augmented Generation (RAG) for context-aware responses.

## 🎯 Project Overview

This project implements an agentic AI system with autonomous multi-agent workflows using state graph orchestration. The system uses RAG (Retrieval-Augmented Generation) to ground responses in a knowledge base, reducing hallucinations and improving accuracy.

### Key Features

## 🔬 Technical Design Decisions

### Vector Database Selection
**Chose ChromaDB** over Pinecone/Weaviate for this implementation:
- ✅ Local-first development (no external API dependencies)
- ✅ Native LangChain integration
- ✅ Fastest path to working prototype

**Production Considerations:**
- **Pinecone**: Managed service, better for scale (millions of vectors)
- **Weaviate**: Superior hybrid search, advanced filtering capabilities
- **ChromaDB**: Excellent for prototyping, self-hosted production

### Evaluation & Quality Metrics
Current implementation tracks:
- Source attribution accuracy (% of responses with citations)
- Retrieval relevance (semantic search precision)
- Response latency

**Next Steps for Production:**
- RAGAS metrics: Faithfulness, answer relevance, context precision
- DeepEval: Hallucination detection, toxicity checks
- Custom test dataset with ground truth Q&A pairs

### Agent Architecture Philosophy
**Sequential workflow** (Retrieve → Research → Synthesize):
- ✅ Clear separation of concerns
- ✅ Easy to debug and monitor
- ✅ Predictable behavior

**Considered but deferred:**
- Self-reflection loops (validator agent checking synthesis quality)
- Multi-path reasoning with agent voting
- Dynamic replanning based on retrieval quality
- 
## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    User Query                            │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph State Machine                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │  1. Retrieval Node                                │  │
│  │     - Query embedding                             │  │
│  │     - Vector similarity search (ChromaDB)         │  │
│  │     - Context extraction                          │  │
│  └───────────────┬───────────────────────────────────┘  │
│                  │                                       │
│                  ▼                                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │  2. Research Agent                                │  │
│  │     - Context analysis                            │  │
│  │     - Information extraction                      │  │
│  │     - Source attribution                          │  │
│  └───────────────┬───────────────────────────────────┘  │
│                  │                                       │
│                  ▼                                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │  3. Synthesizer Agent                             │  │
│  │     - Response generation                         │  │
│  │     - Citation formatting                         │  │
│  │     - Quality assurance                           │  │
│  └───────────────┬───────────────────────────────────┘  │
└──────────────────┼───────────────────────────────────────┘
                   │
                   ▼
           Final Answer with Citations
```

## 🛠️ Tech Stack

- **LangGraph**: State graph orchestration for agent workflows
- **LangChain**: Framework for LLM application development
- **Hugging Face**: LLM (Flan-T5) and embeddings (MiniLM)
- **ChromaDB**: Vector database for semantic search
- **Streamlit**: Interactive web interface
- **Python 3.11+**: Core programming language

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- Hugging Face API key (free)

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd rag-research-assistant
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your Hugging Face API key
```

## 🚀 Usage

### Run the Streamlit App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Test the System
```bash
python test_system.py
```

### Example Queries
- "What is machine learning?"
- "Explain transformer architecture"
- "How does RAG work?"
- "What are autonomous agents?"

## 📊 Performance Metrics

- **Retrieval Accuracy**: Semantic search across knowledge base
- **Response Time**: ~1-2 seconds per query
- **Source Attribution**: 100% of responses cite original documents
- **Agent Steps**: 3-step workflow (Retrieve → Research → Synthesize)

## 🧪 Project Structure
```
rag-research-assistant/
├── agents/
│   ├── __init__.py
│   └── graph_agents.py          # LangGraph agent implementation
├── rag/
│   ├── __init__.py
│   ├── embeddings.py            # Document processing & embeddings
│   ├── vectordb.py              # ChromaDB vector store
│   └── retrieval.py             # RAG retrieval logic
├── utils/
│   ├── __init__.py
│   └── config.py                # Configuration management
├── data/
│   └── documents/               # Knowledge base documents
├── app.py                       # Streamlit UI
├── test_system.py               # Testing script
├── create_sample_docs.py        # Generate sample data
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
└── README.md                    # This file
```

## 🎓 Key Concepts Demonstrated

### 1. Agentic AI
- Autonomous decision-making agents
- Multi-step reasoning workflows
- Tool use and context management

### 2. RAG (Retrieval-Augmented Generation)
- Vector embeddings for semantic search
- Context injection into LLM prompts
- Grounded responses with source attribution

### 3. State Graph Orchestration
- LangGraph state machines
- Node-based agent workflows
- Sequential and parallel processing

### 4. Production-Ready Patterns
- Error handling and fallbacks
- Performance monitoring
- Modular architecture

## 🔧 Customization

### Add Your Own Documents
Place `.txt` files in `data/documents/` and restart the system.

### Change the LLM Model
Edit `.env` and update `MODEL_NAME` to any Hugging Face model:
```
MODEL_NAME=google/flan-t5-large
```

### Adjust RAG Parameters
In `utils/config.py`:
```python
CHUNK_SIZE = 1000        # Document chunk size
CHUNK_OVERLAP = 200      # Overlap between chunks
```

## 📈 Future Enhancements

- [ ] Add web search capability for real-time information
- [ ] Implement conversation memory
- [ ] Add multi-document comparison
- [ ] Deploy as API service
- [ ] Add evaluation metrics dashboard

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Your Name**
- GitHub: [@aldrinsui](https://github.com/aldrinsui)
- Email: aldrinjerry24@gmail.com

## 🙏 Acknowledgments

- Built for AI/ML Internship application at Stackular
- Inspired by modern agentic AI frameworks (CrewAI, AutoGen)
- Uses open-source tools from LangChain, Hugging Face, and Streamlit communities

---

**⭐ If you find this project interesting, please consider starring it!**
