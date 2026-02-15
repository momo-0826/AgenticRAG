from langchain_aws import ChatBedrockConverse
from src.config.config import BEDROCK_CONFIG

def get_llm():
    return ChatBedrockConverse(**BEDROCK_CONFIG)