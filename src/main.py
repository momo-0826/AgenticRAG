from src.recommend_function.graph import create_graph

def main():
    graph = create_graph()
    result = graph.invoke({
        "question": "Pythonが得意なバックエンドエンジニアを探して",
        "documents": [],
        "messages": [],
        "generation": "",
        "retry_count": 0,
        "needs_retry": False,
    })
    print(result["generation"])

if __name__ == "__main__":
    main()