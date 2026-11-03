from dotenv import load_dotenv
import os
from pathlib import Path
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatOllama(temperature=0, model="llama3.2:3b")
tools = [
    TavilySearch(
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        max_results=5,
        search_depth="advanced",
        include_answer="advanced"
    )
]

agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":[HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?")]})
    print(result)


if __name__ == "__main__":
    main()
