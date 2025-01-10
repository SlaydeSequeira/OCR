import streamlit as st
import google.generativeai as genai
import os
from PIL import Image

# Configure Gemini API
os.environ["API_KEY"] = "AIzaSyB6C57eFm4IsRdNeyHTCbUOX9TDwU9Jc7g"  # Replace with your Gemini API key
genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

# Waste categories
waste_categories = {
    "biodegradable": ["wet_waste", "food_waste", "garden_waste"],
    "recyclable": ["plastic", "metal", "cardboard", "paper", "glass"],
    "hazardous": ["batteries", "chemicals", "e-waste", "medical_waste"]
}

# Streamlit UI
st.title("Waste Classification using AI")
st.write("Upload an image of waste to classify it as biodegradable, recyclable, or hazardous.")

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
        prompt = (
            "Analyze the waste in the image and classify it into one of these categories: "
            "biodegradable, recyclable, or hazardous. Provide a single classification."
        )
        response = model.generate_content([uploaded_file_gemini, prompt])

        # Parse the response for waste category
        result = "Unknown"
        response_text = response.text.lower()
        for category, keywords in waste_categories.items():
            if any(keyword in response_text for keyword in keywords) or category in response_text:
                result = category.capitalize()
                break

        # Display and log result
        st.write(f"### Predicted Waste Category: {result}")
        print(f"Predicted Waste Category: {result}")

    except Exception as e:
        st.error(f"Error during classification: {e}")
        print(f"Error: {e}")

    finally:
        # Clean up the temporary file
        if os.path.exists(image_path):
            os.remove(image_path)
