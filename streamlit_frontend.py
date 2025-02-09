import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from index import emotion_check

from worker_agents.sad_agent import sad_agent
from worker_agents.happy_agent import happy_agent
from worker_agents.neutral_agent import neutral_agent
from utilities.videoHtml import returnVideoHtml
import os
def main():
    st.title("Astronaut Buddy")

    # File uploader (more robust way)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # To read file as bytes:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        # st.write(bytes_data) # You can use this to debug.

        # Save the uploaded file to a temporary file
        # temp_filename = "temp_image." + uploaded_file.name.split('.')[-1] # Get the correct extension
        # with open(temp_filename, "wb") as f:
        #     f.write(bytes_data)
        
        st.write("Astronaut's Image:")
        st.image(uploaded_file, caption="Uploaded Image.", use_container_width=True)

        # Call the image processing function
        # result = emotion_check(temp_filename)
        with st.spinner("Processing..."):
            result =emotion_check(img_array)


        if result:
            st.write("Emotion detected:", result[0]["dominant_emotion"])
            st.write("Age detected:", result[0]["age"])
            st.write("Gender detected:", result[0]["dominant_gender"])
            st.write("Race detected:", result[0]["dominant_race"])


            if(result[0]["dominant_emotion"]=="sad"):
                sadPrompt = f"Create a image of a animal to lighten the mood of a person who is feeling sad. Use img html to display the image. also create a joke to make the person laugh. Use features of the user to make the joke more relatable, use h1 tag for the joke. age:{result[0]['age']} race:{result[0]['dominant_race']} gender:{result[0]['dominant_gender']}. your response should be in html format."
                with st.spinner("Processing..."):
                    response = sad_agent.run(sadPrompt)
                st.success("Below is what I can do to make you feel better")
                st.html(response)
            elif(result[0]["dominant_emotion"]=="happy"):
                happyPrompt = f"Find a good dance or song video for the user to vibe on by using the YouTube tool, ensuring that the search query contains no commas and personalizing the content using the user’s age ({result[0]['age']}), race ({result[0]['race']}), and gender ({result[0]['dominant_gender']}); respond in the exact format 'Thought: extract youtube urls', 'Action: [specify the tool you want to use]', and 'Action Input: [provide the input to that tool]'"
                with st.spinner("Processing..."):
                    response = happy_agent.run(happyPrompt)
                # print(response)
                st.success("Below is what I can do to make you feel better")
                result = returnVideoHtml(response)
            else:
                neutralPrompt = f"Write an poem in the user's native language that is deeply motivational and uplifting. Tailor the content to resonate with the user by incorporating their demographic details: age ({result[0]['age']}), race ('{result[0]['dominant_race']}'), and gender ({result[0]['dominant_gender']}). The poem should emphasize positivity in life, human values, and the importance of national pride and identity, celebrating the beauty and heritage of their culture while inspiring the reader to embrace their potential and contribute positively to society.Also add few motivating quotes in the poem. Once you have the poem completed use the text_to_speech tool to convert the entire poem to audio format."
                with st.spinner("Processing..."):
                    response = neutral_agent.run(neutralPrompt)
                dist = os.path.join("./tools/audio.wav")
                st.success("Below is what I can do to make you feel better")
                st.audio(dist, format='audio/wav')

    



if __name__ == "__main__":
    main()