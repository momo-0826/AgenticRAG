from langgraph.graph import StateGraph, START, END
from src.recommend_function.state import AgentState
from src.recommend_function.retrieve import retrieve
from src.recommend_function.analyze import analyze
from src.recommend_function.recommend import recommend
from src.recommend_function.validate import validate

def should_retry(state: AgentState) -> str:
    if state["needs_retry"]:
        return "retrieve"
    return "analyze"

def create_graph():
    graph = StateGraph(AgentState)

    # ノードを追加
    graph.add_node("retrieve", retrieve)
    graph.add_node("validate", validate)
    graph.add_node("analyze", analyze)
    graph.add_node("recommend", recommend)

    # エッジを追加
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "validate")
    graph.add_conditional_edges("validate", should_retry)
    graph.add_edge("analyze", "recommend")
    graph.add_edge("recommend", END)

    return graph.compile()