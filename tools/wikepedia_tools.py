from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool

# Configure the API wrapper (adjust parameters as needed)
api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=4000)

# Initialize the Wikipedia tool
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Test the tool by searching for "LangChain"
# print(wikipedia_tool.run("LangChain"))

wiki_tool = Tool(
    name="Wikipedia",
    func=wikipedia_tool.run,
    description="Useful for retrieving research information about people, events, and topics from Wikipedia."
)