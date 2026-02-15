from langchain_core.prompts import ChatPromptTemplate

RECOMMEND_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "あなたは副業人材を紹介する専門のコンサルタントです。\n"
        "以下の候補者情報をもとに、ユーザーの要望に最適な人材を推薦してください。\n"
        "推薦理由も具体的に説明してください。\n\n"
        "## 候補者情報\n"
        "{context}"
    ),
    ("human", "{question}"),
])