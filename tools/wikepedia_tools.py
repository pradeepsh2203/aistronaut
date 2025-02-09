from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Configure the API wrapper (adjust parameters as needed)
api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=4000)

# Initialize the Wikipedia tool
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Test the tool by searching for "LangChain"
# print(wikipedia_tool.run("LangChain"))