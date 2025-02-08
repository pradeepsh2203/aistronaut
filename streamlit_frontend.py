import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from index import emotion_check


def main():
    st.title("Image Processing with Streamlit")

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
        
        st.write("Uploaded Image:")
        st.image(uploaded_file, caption="Uploaded Image.", use_container_width=True)

        # Call the image processing function
        # result = emotion_check(temp_filename)
        result =emotion_check(img_array)


        if result:
            st.success(result)
            # You can display other results from process_image() here



    



if __name__ == "__main__":
    main()