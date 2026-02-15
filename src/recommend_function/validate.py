from langchain_core.prompts import ChatPromptTemplate
from src.recommend_function.state import AgentState
from src.utils.model import get_llm

VALIDATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "あなたは人材推薦の品質チェック担当です。\n"
        "以下の候補者情報がユーザーの要望を満たしているか判定してください。\n"
        "十分な候補者がいる場合は「十分」、不十分な場合は「不十分」とだけ回答してください。\n\n"
        "## 候補者情報\n"
        "{context}"
    ),
    ("human", "{question}"),
])

MAX_RETRY = 2

def _format_docs(documents):
    return "\n\n---\n\n".join(doc.page_content for doc in documents)

def validate(state: AgentState) -> dict:
    question = state["question"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0)
    context = _format_docs(documents)

    llm = get_llm()
    chain = VALIDATE_PROMPT | llm

    result = chain.invoke({"context": context, "question": question})

    needs_retry = "不十分" in result.content and retry_count < MAX_RETRY
    
    return {
        "needs_retry": needs_retry,
        "retry_count": retry_count + 1,
    }