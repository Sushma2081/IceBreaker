from unittest import result

from dotenv import load_dotenv
import os
from pathlib import Path
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent's response"""
    answer:str=Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used in the response")


llm = ChatOllama(temperature=0, model="llama3.2:3b")
tools = [
    TavilySearch(
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        max_results=5,
        search_depth="advanced",
        include_answer="advanced"
    )
]

agent = create_agent(model=llm, tools=tools, response_format=ToolStrategy(AgentResponse))

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":[HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?")]})
    #print(result)
    print("\n--- STRUCTURED RESPONSE ---")
    print(result.get("structured_response"))

    print("\n--- MESSAGE DETAILS ---")
    for msg in result["messages"]:
        print("Message type:", type(msg).__name__)
        print("Content:", msg.content)

        if hasattr(msg, "tool_calls"):
            print("Tool calls:", msg.tool_calls)


if __name__ == "__main__":
    main()
