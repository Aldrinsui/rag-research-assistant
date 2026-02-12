from agents.graph_agents import MultiAgentRAG

print("🚀 Testing Multi-Agent RAG System with Hugging Face\n")

# Test
agent_system = MultiAgentRAG()

test_queries = [
    "What is machine learning?",
]

for query in test_queries:
    result = agent_system.process_query(query)
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {result['sources']}")
    print(f"Time: {result['processing_time']}s")
    print(f"{'='*60}\n")
