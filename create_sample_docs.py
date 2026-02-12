sample_topics = {
    "machine_learning.txt": """
Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves. The primary aim is to allow computers to learn automatically without human intervention and adjust actions accordingly.

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models. Unsupervised learning finds patterns in unlabeled data. Reinforcement learning learns through trial and error with rewards and penalties.

Common applications include image recognition, natural language processing, recommendation systems, and autonomous vehicles. Modern ML heavily relies on neural networks and deep learning architectures.
    """,
    
    "transformers.txt": """
Transformers are a type of neural network architecture that has revolutionized natural language processing. Introduced in the paper "Attention Is All You Need" in 2017, transformers use self-attention mechanisms to process sequential data in parallel rather than sequentially.

The key innovation is the attention mechanism, which allows the model to weigh the importance of different parts of the input when making predictions. This has led to breakthrough models like BERT, GPT, and T5 that excel at various NLP tasks.

Transformers consist of encoder and decoder blocks. The encoder processes input sequences, while the decoder generates output sequences. The self-attention mechanism allows each position to attend to all positions in the previous layer.
    """,
    
    "rag_systems.txt": """
Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. RAG systems first retrieve relevant documents from a knowledge base, then use this context to generate more accurate and informed responses.

The process involves: 1) Converting documents into embeddings and storing them in a vector database, 2) Embedding user queries and finding similar documents, 3) Passing retrieved context to a language model for generation. This approach reduces hallucinations and grounds responses in factual information.

RAG is particularly useful for question-answering systems, chatbots, and knowledge management applications. It allows LLMs to access external knowledge without retraining the entire model.
    """,
    
    "vector_databases.txt": """
Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They are essential for modern AI applications that use embeddings to represent data semantically.

Popular vector databases include ChromaDB, Pinecone, Weaviate, and Milvus. These databases use algorithms like HNSW (Hierarchical Navigable Small World) or IVF (Inverted File Index) to perform fast similarity searches across millions of vectors.

Vector databases enable semantic search, where queries return results based on meaning rather than exact keyword matches. This is crucial for RAG systems, recommendation engines, and similarity detection applications.
    """,
    
    "langchain.txt": """
LangChain is a framework for developing applications powered by language models. It provides tools for chaining together different components like prompts, models, and data sources to create complex AI applications.

Key features include: agent frameworks for autonomous decision-making, memory systems for conversation history, document loaders for various data sources, and chains for combining multiple operations. LangChain simplifies building production-ready LLM applications.

LangGraph, an extension of LangChain, enables building stateful, multi-agent workflows using graph-based orchestration. This allows for more complex agent interactions and decision-making processes.
    """,
    
    "autonomous_agents.txt": """
Autonomous agents are AI systems that can perceive their environment, make decisions, and take actions to achieve specific goals without constant human oversight. In the context of LLMs, these agents can use tools, retrieve information, and execute multi-step reasoning.

Modern agent frameworks like LangGraph, CrewAI, and AutoGen enable building sophisticated multi-agent systems. These agents can collaborate, delegate tasks, and execute complex workflows autonomously.

Key components of autonomous agents include: reasoning engines, tool use capabilities, memory systems, and planning mechanisms. They represent the next evolution in AI systems beyond simple prompt-response interactions.
    """
}

import os
os.makedirs("data/documents", exist_ok=True)

for filename, content in sample_topics.items():
    with open(f"data/documents/{filename}", "w") as f:
        f.write(content.strip())

print("✅ Created 6 sample documents in data/documents/")
print("\nDocuments:")
for filename in sample_topics.keys():
    print(f"  - {filename}") 
