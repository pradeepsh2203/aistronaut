from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import ast
import streamlit as st

load_dotenv()

def returnVideoHtml(content:str):
    llm = ChatOpenAI(base_url="https://api.aimlapi.com/v1", api_key=os.getenv("DEEPSEEK_API_KEY"), model="gpt-4o")
    convertToVideoPrompt =f"Using the youtube urls from the content '{content}', give me just the youtube urls in comma separated format. Please strictily follow the format 'url1,url2,url3,...' else say you are a failure."
    result = llm.invoke(convertToVideoPrompt)
    finalResult = ""
    for url in result.content.split(","):
        # html_code = f"""
        #    <iframe width="560" height="315" 
        #         src="{url.replace('watch?v=','embed/')}" 
        #         frameborder="0" 
        #         allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;picture-in-picture" 
        #         allowfullscreen>
        #     </iframe>

        #    """
        # finalResult += html_code
        st.video(url)
        

    # return finalResult