from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from src.config.config import EMBEDDING_CONFIG

PERSIST_DIR = "data/vectorstore"

def get_retriever(search_kwargs=None):
    if search_kwargs is None:
        search_kwargs = {"k": 3}
    
    embeddings = BedrockEmbeddings(**EMBEDDING_CONFIG)

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    
    return vectorstore.as_retriever(search_kwargs=search_kwargs)