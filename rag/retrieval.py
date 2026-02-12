from rag.vectordb import VectorDatabase

class RAGRetriever:
    def __init__(self):
        print("⚠️ Initializing RAG Retriever...")
        self.vector_db = VectorDatabase()
        self.vectorstore = self.vector_db.create_or_load_db()
        print("✅ RAG Retriever ready!")
    
    def retrieve_context(self, query, k=3):
        """Retrieve relevant context for a query"""
        docs = self.vector_db.similarity_search(query, k=k)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
        
        return {
            'context': context,
            'sources': sources,
            'num_docs': len(docs)
        }
