import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
tools = [TavilySearchResults(max_results=1)]


prompt = hub.pull("hwchase17/react")


llm = ChatGroq(
    model = "llama3-70b-8192",
    max_retries = 5,
    api_key = os.getenv("GROQ_API_KEY")
)

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


answer = agent_executor.invoke({"input": "rules for selection in ARR Conference."})

print(answer)
