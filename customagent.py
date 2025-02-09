from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from tools.wikepedia_tools import wikipedia_tool
from langchain.agents import initialize_agent,Tool
from dotenv import load_dotenv
import os

# load_dotenv()

# print(os.getenv("OPENAI_API_KEY"))

# wiki_tool = Tool(
#     name="Wikipedia",
#     func=wikipedia_tool.run,
#     description="Useful for retrieving research information about people, events, and topics from Wikipedia."
# )



# llm = ChatOpenAI(base_url="https://api.aimlapi.com/v1", api_key=os.getenv("OPENAI_API_KEY"), model="deepseek/deepseek-r1", verbose=True)
# # message = [HumanMessage(content="what's 39 plus 10?")]
# # response = llm(message)

# agent = initialize_agent([wiki_tool], llm, agent="zero-shot-react-description", verbose=True)

# response = agent.run("Provide a detailed research on Mahakumbh Mela")
# print(response)

class CustomAgent:
    def __init__(self,tools,base_url="https://api.aimlapi.com/v1", api_key=os.getenv("DEEPSEEK_API_KEY"), model="deepseek/deepseek-r1"):
        self.llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model, verbose=True)
        # wiki_tool = Tool(
        #     name="Wikipedia",
        #     func=wikipedia_tool.run,
        #     description="Useful for retrieving research information about people, events, and topics from Wikipedia."
        # )
        self.agent = initialize_agent(tools, self.llm, agent="zero-shot-react-description", verbose=True)
    
    def run(self,prompt : str):
        response = self.agent.run(prompt)
        return response