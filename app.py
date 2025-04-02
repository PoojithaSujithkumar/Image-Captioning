import google.generativeai as genai
import gradio as gr
from PIL import Image

# Set up Gemini API key
import os
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Replace with your API key

def generate_caption(image):
    """Generate a caption for the uploaded image using Google Gemini Pro Vision API."""

    model = genai.GenerativeModel("gemini-1.5-flash")

    # Open image using PIL
    img = Image.open(image)

    # Generate caption
    response = model.generate_content(
        [img, "Describe the contents of this image in detail."]
    )
    
    # Extract caption
    caption = response.text
    return caption

# Gradio UI
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    title="Image Captioning App",
    description="Upload an image, and a descriptive caption will be generated."
)

# Run the app
if __name__ == "__main__":
    iface.launch()
