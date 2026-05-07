# Program title: Storytelling App
# Description: A storytelling application for 3-10 year old kids.
#              Users upload an image, the app generates a caption,
#              creates a kid-friendly story, and converts it to audio.

# Import part
import streamlit as st
from transformers import pipeline
from PIL import Image

# ============================================================
# Function part
# ============================================================

# img2text: Extract a caption from the uploaded image
def img2text(url):
    """
    Use an image-to-text pipeline to generate a caption from the image.
    Args:
        url: Path to the image file.
    Returns:
        text: Generated caption string.
    """
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text


# text2story: Generate a kid-friendly story from the caption
def text2story(text):
    """
    Use a text-generation pipeline to create a short story (50-100 words)
    suitable for children aged 3-10 based on the image caption.
    Args:
        text: The image caption string.
    Returns:
        story_text: Generated story string.
    """
    story_pipe = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

    # Design a prompt that guides the model to produce a kid-friendly story
    prompt = (
        f"Write a fun, magical, and easy-to-read short story for little kids "
        f"about the following scene: {text}. "
        f"The story should be happy, imaginative, and between 50 to 100 words. "
        f"Once upon a time,"
    )

    story_results = story_pipe(
        prompt,
        max_length=180,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )
    story_text = story_results[0]["generated_text"]

    # Remove the original prompt portion so only the story remains
    # Keep "Once upon a time," as the story beginning
    if "Once upon a time," in story_text:
        story_text = "Once upon a time," + story_text.split("Once upon a time,", 1)[1]

    # Trim to roughly 100 words to stay within the 50-100 word target
    words = story_text.split()
    if len(words) > 100:
        story_text = " ".join(words[:100]) + "."

    return story_text


# text2audio: Convert the story text into audio
def text2audio(story_text):
    """
    Use a text-to-audio pipeline to convert the story into speech audio.
    Args:
        story_text: The story string to convert to speech.
    Returns:
        audio_data: Dictionary containing 'audio' array and 'sampling_rate'.
    """
    audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story_text)
    return audio_data


# ============================================================
# Main part
# ============================================================

# Page configuration
st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
st.header("Turn Your Image to Audio Story")
st.subheader("A magical storytelling app for kids aged 3-10! 🌟")

# Image uploader
uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save file locally
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Stage 1: Image to Text (Using the img2text function)
    with st.spinner("🔍 Looking at the image..."):
        scenario = img2text(uploaded_file.name)
    st.write(f"**Scenario:** {scenario}")

    # Stage 2: Text to Story (Using the text2story function)
    with st.spinner("📖 Generating a story..."):
        story = text2story(scenario)
    st.write(f"**Story:** {story}")

    # Stage 3: Story to Audio (Using the text2audio function)
    with st.spinner("🔊 Generating audio data..."):
        audio_data = text2audio(story)

    # Play button
    if st.button("Play Audio"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
