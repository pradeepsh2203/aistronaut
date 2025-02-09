from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from tools.wikepedia_tools import wikipedia_tool
from langchain.agents import initialize_agent,Tool
from dotenv import load_dotenv
from tools.youtube_tool import youtube_tool
import os
load_dotenv()

class HappyAgent:
    def __init__(self,base_url="https://api.aimlapi.com/v1", api_key=os.getenv("DEEPSEEK_API_KEY"), model="gpt-4o"):
        self.llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model, verbose=True)
        tools = [youtube_tool]
        self.agent = initialize_agent(tools, self.llm, agent="zero-shot-react-description", verbose=True,handle_parsing_errors=True)
    
    def run(self,prompt):
        response = self.agent.run(prompt)
        return response
        

happy_agent = HappyAgent()