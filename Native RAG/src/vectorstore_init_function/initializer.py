from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from src.config.config import EMBEDDING_CONFIG
from src.vectorstore_init_function.loader import load_candidates

PERSIST_DIR = "data/vectorstore"
CANDIDATES_PATH = "data/raw/candidates.json"

def initialize_vectorstore():
    documents = load_candidates(CANDIDATES_PATH)

    embeddings = BedrockEmbeddings(**EMBEDDING_CONFIG)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    print(f"{len(documents)}件の人材データをベクトルストアに格納しました。")
    return vectorstore

if __name__ == "__main__":
    initialize_vectorstore()