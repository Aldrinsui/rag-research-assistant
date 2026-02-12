from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from utils.config import Config
import os

class DocumentProcessor:
    def __init__(self):
        print("🤗 Using Hugging Face embeddings (free!)")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def load_documents(self, directory="data/documents"):
        """Load all text documents from directory"""
        loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents):
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ Split into {len(chunks)} chunks")
        return chunks
    
    def get_embeddings(self):
        """Return embeddings model"""
        return self.embeddings
