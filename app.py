import streamlit as st
from agents.graph_agents import MultiAgentRAG
import time

st.set_page_config(
    page_title="LangGraph Multi-Agent RAG",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 LangGraph Multi-Agent RAG System")
st.markdown("*Autonomous agent workflow with Hugging Face models (100% FREE!)*")

# Initialize
@st.cache_resource
def get_agent_system():
    return MultiAgentRAG()

agent_system = get_agent_system()

# Sidebar
with st.sidebar:
    st.header("📊 System Architecture")
    st.markdown("""
    **Agent Workflow (LangGraph):**
    
    1. 🔍 **Retrieval Node**
       - Vector DB search
       - Context extraction
    
    2. 📚 **Research Node**
       - Analyze context
       - Extract key info
    
    3. ✍️ **Synthesis Node**
       - Generate answer
       - Cite sources
    
    **Tech Stack:**
    - 🤗 Hugging Face (LLM & Embeddings)
    - LangGraph (State graphs)
    - ChromaDB (Vector store)
    - Mistral-7B-Instruct
    """)
    
    st.markdown("---")
    st.success("💰 100% FREE - No API costs!")

# Main interface
query = st.text_input(
    "Enter your research question:",
    placeholder="What is machine learning and how does it work?"
)

if st.button("🚀 Execute Agent Workflow", type="primary"):
    if query:
        with st.spinner("🤖 Agents processing (may take 30-60s with free API)..."):
            result = agent_system.process_query(query)
        
        st.success(f"✅ Completed in {result['processing_time']}s")
        
        # Workflow steps
        with st.expander("🔄 Agent Workflow Steps", expanded=True):
            for i, step in enumerate(result['workflow_steps'], 1):
                st.markdown(f"**Step {i}:** {step}")
        
        # Answer
        st.markdown("### 💡 Final Answer")
        st.markdown(result['answer'])
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📚 Sources Used", result['num_sources'])
        with col2:
            st.metric("⏱️ Processing Time", f"{result['processing_time']}s")
        
        # Sources
        with st.expander("📖 View Source Documents"):
            for i, source in enumerate(result['sources'], 1):
                st.code(f"{i}. {source}", language="text")
    else:
        st.warning("⚠️ Please enter a question")

# Example queries
st.markdown("---")
st.markdown("### 💡 Example Queries")

col1, col2, col3 = st.columns(3)
examples = [
    ("What is machine learning?", col1),
    ("Explain RAG systems", col2),
    ("What are transformers?", col3)
]

for example, col in examples:
    with col:
        if st.button(example, key=example):
            st.rerun()
