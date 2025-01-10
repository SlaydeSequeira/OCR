import streamlit as st
import google.generativeai as genai
import os
from PIL import Image

# Configure Gemini API
os.environ["API_KEY"] = "AIzaSyB6C57eFm4IsRdNeyHTCbUOX9TDwU9Jc7g"  # Replace with your Gemini API key
genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

# Waste categories
waste_list = ['wet_waste', 'plastic_waste', 'metal', 'cardboard', 'paper', 'glass']

# Streamlit UI
st.title("Waste Classification using AI")
st.write("Upload an image of waste to classify its type.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Save the image locally
        file_extension = uploaded_file.name.split('.')[-1]
        image_path = f"temp_image.{file_extension}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Upload file to Gemini
        uploaded_file_gemini = genai.upload_file(
            path=image_path,
            mime_type=f"image/{file_extension}",
            display_name="Uploaded Waste Image"
        )

        # Generate classification prompt
        prompt = f"Classify the waste in this image. The possible types are: {', '.join(waste_list)}."
        response = model.generate_content([uploaded_file_gemini, prompt])

        # Parse the response for waste type
        result = "Unknown"
        for item in waste_list:
            if item in response.text.lower():
                result = item
                break

        # Display and log result
        st.write(f"### Predicted Waste Type: {result}")
        print(f"Predicted Waste Type: {result}")

    except Exception as e:
        st.error(f"Error during classification: {e}")
        print(f"Error: {e}")

    finally:
        # Clean up the temporary file
        if os.path.exists(image_path):
            os.remove(image_path)
