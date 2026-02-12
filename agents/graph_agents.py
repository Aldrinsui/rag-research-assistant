from typing import TypedDict, Annotated
from langchain_huggingface import HuggingFaceEndpoint
from langgraph.graph import StateGraph, END
from rag.retrieval import RAGRetriever
from utils.config import Config
import operator
import os

class AgentState(TypedDict):
    query: str
    context: str
    sources: list
    research_findings: str
    final_answer: str
    messages: Annotated[list, operator.add]

class MultiAgentRAG:
    def __init__(self):
        print("🤗 Initializing Hugging Face LLM...")
        
        # Get API token from environment
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or Config.HUGGINGFACE_API_KEY
        
        self.llm = HuggingFaceEndpoint(
            repo_id=Config.MODEL_NAME,
            huggingfacehub_api_token=api_token,
            temperature=0.3,
            max_new_tokens=512,
            timeout=120
        )
        self.retriever = RAGRetriever()
        self.workflow = self._build_graph()
    
    def retrieve_context(self, state: AgentState) -> AgentState:
        """Node: Retrieve relevant context from vector DB"""
        print("🔍 Retrieving context...")
        
        rag_result = self.retriever.retrieve_context(state['query'], k=4)
        
        state['context'] = rag_result['context']
        state['sources'] = rag_result['sources']
        state['messages'].append(f"Retrieved {len(rag_result['sources'])} relevant sources")
        
        return state
    
    def research_agent(self, state: AgentState) -> AgentState:
        """Node: Research agent analyzes context"""
        print("📚 Research agent analyzing...")
        
        prompt = f"""Analyze this context and answer the query.

Query: {state['query']}

Context: {state['context'][:800]}

Provide a clear analysis:"""
        
        try:
            # Use invoke instead of calling directly
            response = self.llm.invoke(prompt)
            state['research_findings'] = response if isinstance(response, str) else str(response)
        except Exception as e:
            print(f"⚠️ LLM Error: {e}")
            # Fallback: use context directly
            state['research_findings'] = f"Based on the documents: {state['context'][:500]}"
        
        state['messages'].append("Research analysis complete")
        
        return state
    
    def synthesizer_agent(self, state: AgentState) -> AgentState:
        """Node: Synthesizer creates final answer"""
        print("✍️ Synthesizing final answer...")
        
        # Simple synthesis without LLM call for now
        sources_text = ", ".join([s.split('/')[-1] for s in state['sources']])
        
        state['final_answer'] = f"""Based on the research findings:

{state['research_findings'][:600]}

Sources: [{sources_text}]"""
        
        state['messages'].append("Final answer synthesized")
        
        return state
    
    def _build_graph(self):
        """Build the agent workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve_context)
        workflow.add_node("research", self.research_agent)
        workflow.add_node("synthesize", self.synthesizer_agent)
        
        # Add edges (workflow)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "research")
        workflow.add_edge("research", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def process_query(self, query: str):
        """Execute the agent workflow"""
        print(f"\n{'='*60}")
        print(f"Processing: {query}")
        print(f"{'='*60}\n")
        
        import time
        start_time = time.time()
        
        initial_state = {
            "query": query,
            "context": "",
            "sources": [],
            "research_findings": "",
            "final_answer": "",
            "messages": []
        }
        
        result = self.workflow.invoke(initial_state)
        
        elapsed_time = time.time() - start_time
        
        return {
            'query': query,
            'answer': result['final_answer'],
            'sources': result['sources'],
            'num_sources': len(result['sources']),
            'processing_time': round(elapsed_time, 2),
            'workflow_steps': result['messages']
        }
