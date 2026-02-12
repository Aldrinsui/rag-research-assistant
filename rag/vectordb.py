from langchain_community.vectorstores import Chroma
from rag.embeddings import DocumentProcessor
from utils.config import Config
import os

class VectorDatabase:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.embeddings = self.processor.get_embeddings()
        self.vectorstore = None
    
    def create_or_load_db(self):
        """Create vector database or load existing one"""
        if os.path.exists(Config.CHROMA_PATH):
            print("📂 Loading existing vector database...")
            self.vectorstore = Chroma(
                persist_directory=Config.CHROMA_PATH,
                embedding_function=self.embeddings
            )
        else:
            print("🔨 Creating new vector database...")
            documents = self.processor.load_documents()
            chunks = self.processor.split_documents(documents)
            
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=Config.CHROMA_PATH
            )
            print(f"✅ Vector database created at {Config.CHROMA_PATH}")
        
        return self.vectorstore
    
    def similarity_search(self, query, k=3):
        """Search for similar documents"""
        if not self.vectorstore:
            self.create_or_load_db()
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
