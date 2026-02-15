from langchain_core.prompts import ChatPromptTemplate
from src.recommend_function.state import AgentState
from src.utils.model import get_llm

ANALYZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "あなたは人材マッチングの専門家です。\n"
        "ユーザーの要望と候補者情報を比較し、各候補者のマッチング度を分析してください。\n"
        "各候補者について以下を評価してください：\n"
        "- マッチング度（高/中/低）\n"
        "- マッチする点\n"
        "- マッチしない点\n\n"
        "## 候補者情報\n"
        "{context}"
    ),
    ("human", "{question}"),
])

def _format_docs(documents):
    return "\n\n---\n\n".join(doc.page_content for doc in documents)

def analyze(state: AgentState) -> dict:
    question = state["question"]
    documents = state["documents"]
    context = _format_docs(documents)

    llm = get_llm()
    chain = ANALYZE_PROMPT | llm

    result = chain.invoke({"context": context, "question": question})
    return {"messages": [result]}