from langchain.tools import Tool
from pydantic import BaseModel, Field
import os
import requests

def generate_image(promtp):
    print(promtp)
    response = requests.post(
            "https://api.aimlapi.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer b218108b3bac44c5831ffe575831f344",
                "Content-Type": "application/json",
            },
            json={
                "prompt": promtp,
                "model": "stable-diffusion-v3-medium",
            },
        )

    response.raise_for_status()
    data = response.json()
    print("Generation:", data)
    return data

image_tool = Tool(
    name="ImageGenerator",
    func=generate_image,
    description="Generates an image based on a given text prompt and returns the image URL."
)


# # Define the input schema using Pydantic
# class ImageGeneratotToolInput(BaseModel):
#     query: str = Field(..., description="Create a image from a query")

# # Create a custom tool by subclassing BaseTool
# class ImageGeneratorTool(BaseTool):
#     name = "image_generator"
#     description = "Processes a query and generates an image."

#     # Optionally, specify the input schema
#     args_schema = ImageGeneratotToolInput

#     def _run(self, query: str):
#         response = requests.post(
#                 "https://api.aimlapi.com/v1/images/generations",
#                 headers={
#                     "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
#                     "Content-Type": "application/json",
#                 },
#                 json={
#                     "prompt": query,
#                     "model": "flux-pro/v1.1-ultra",
#                 },
#             )

#         response.raise_for_status()
#         data = response.json()
#         print("Generation:", data)

#         # Your custom logic here (e.g., processing the query)
#     async def _arun(self, query: str):
#         # For now, you can leave the async version unimplemented if not needed.
#         raise NotImplementedError("Async version not implemented")

