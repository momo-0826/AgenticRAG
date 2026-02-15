from src.recommend_function.state import AgentState
from src.utils.retriever import get_retriever

def retrieve(state: AgentState) -> dict:
    question = state["question"]
    retriever = get_retriever()
    documents = retriever.invoke(question)
    return {"documents": documents}