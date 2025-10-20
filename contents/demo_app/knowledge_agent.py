import os
import re
import json

from databricks.sdk import WorkspaceClient
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from databricks_langchain.chat_models import ChatDatabricks
from langgraph.prebuilt import create_react_agent

INDEX_NAME = os.environ.get("INDEX_NAME")
SLIDE_NUMBER_COL = "slide_number"
COLUMNS_TO_FETCH = [SLIDE_NUMBER_COL, "slide_content", "file_url", "image_path"]
MAX_RESULT = os.environ.get("MAX_RESULT", 3)
SIMILARITY_THRESHOLD = 0.6
EMBEDDING_MODEL_ENDPOINT_NAME = os.environ.get(
    "EMBEDDING_MODEL_ENDPOINT_NAME",
    "databricks-bge-large-en",
)
KNOWLEDGE_AGENT_MODEL = "databricks-llama-4-maverick"

w = WorkspaceClient()
openai_client = w.serving_endpoints.get_open_ai_client()


def get_embeddings(text):
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL_ENDPOINT_NAME, input=text
        )
        return response.data[0].embedding
    except Exception as e:
        return f"Error generating embeddings: {e}"


def run_vector_search(prompt: str) -> str:
    prompt_vector = get_embeddings(prompt)
    if prompt_vector is None or isinstance(prompt_vector, str):
        return f"Failed to generate embeddings: {prompt_vector}"

    try:
        query_result = w.vector_search_indexes.query_index(
            index_name=INDEX_NAME,
            columns=COLUMNS_TO_FETCH,
            query_vector=prompt_vector,
            num_results=MAX_RESULT,
        )
        data_array = query_result.result.data_array
        result = []
        for data in data_array:
            # 類似度スコアで足切り
            if data[-1] < SIMILARITY_THRESHOLD:
                continue
            result.append(dict(zip(COLUMNS_TO_FETCH, data)))
        return result
    except Exception as e:
        return f"Error during vector search: {e}"


# Define tools
@tool
def retriever_tool(query: str) -> dict:
    """検索クエリに関連する情報を取得する.

    Args:
        query: 検索クエリ。必ず英語で入力すること。
    """
    return run_vector_search(query)


def create_knowlege_agent():
    llm = ChatDatabricks(
        endpoint=KNOWLEDGE_AGENT_MODEL,
        temperature=0,
    )
    agent = create_react_agent(
        model=llm,
        tools=[retriever_tool],
        prompt="あなたはユーザーが必要な情報をツールを実行して検索し、その内容について要約する役目を持ったエージェントです。回答は日本語で行ってください。",
    )
    return agent


def create_assistant_message(ai_response):
    """agentのレスポンスからテキスト部分と画像部分を取り出し、返す。"""
    ai_message = ""
    reference = "\n### 参照スライド\n"
    image_files = []
    for message in ai_response:
        if isinstance(message, AIMessage):
            ai_message += message.content
        elif isinstance(message, ToolMessage):
            tool_content = message.content

            # vector search toolが1件もヒットしなかった場合、listで返ってくるのでその場合にはスキップ
            if isinstance(message.content, list):
                continue

            # tool_content を Dict に変換
            toolmessage_data = json.loads(tool_content)

            # 参照元 URL の Markdown を生成
            file_urls = [d["file_url"] for d in toolmessage_data if d.get("file_url")]
            slide_urls = []
            for url in file_urls:
                m = re.search(r"#(p\d+)$", url)
                label = m.group(1) if m else url  # 見つからなければ URL 全体をラベルに
                slide_urls.append(f"- [{label}]({url})\n")
            reference += "".join(slide_urls)

            # 画像パスタのリストを生成
            image_files = [
                d["image_path"] for d in toolmessage_data if d.get("image_path")
            ]

    return {"text": ai_message + "\n" + reference, "images": image_files}
