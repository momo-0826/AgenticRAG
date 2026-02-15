from langchain_core.prompts import ChatPromptTemplate
from src.recommend_function.state import AgentState
from src.utils.model import get_llm

RECOMMEND_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "あなたは副業人材を紹介する専門のコンサルタントです。\n"
        "以下の分析結果をもとに、最適な人材を推薦してください。\n"
        "推薦理由、候補者の強み、想定される活躍シーンを具体的に説明してください。\n\n"
        "## 分析結果\n"
        "{analysis}"
    ),
    ("human", "{question}")
])

def recommend(state: AgentState) -> dict:
    question = state["question"]
    analysis = state["messages"][-1].content

    llm = get_llm()
    chain = RECOMMEND_PROMPT | llm

    result = chain.invoke({"analysis": analysis, "question": question})
    return {"generation": result.content}