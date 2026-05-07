# Program title: Storytelling App
# Description: A storytelling application for 3-10 year old kids.
#              Users upload an image, the app generates a caption,
#              creates a kid-friendly story, and converts it to audio.
#              Includes content safety filtering for child-appropriate output.

# Import part
import re
import streamlit as st
from transformers import pipeline

# ============================================================
# Safety Configuration: Prohibited words for child safety
# ============================================================

# List of prohibited words and themes not suitable for children aged 3-10
PROHIBITED_WORDS = [
    # Violence
    "kill", "killed", "killing", "murder", "murdered", "blood", "bloody",
    "death", "dead", "die", "died", "dying", "weapon", "gun", "knife",
    "sword", "fight", "attack", "war", "bomb", "shoot", "shot", "wound",
    "violent", "violence", "punch", "hit", "hurt", "destroy", "evil",
    # Alcohol and drugs
    "alcohol", "beer", "wine", "whiskey", "vodka", "drunk", "drinking",
    "smoke", "smoking", "cigarette", "tobacco", "drug", "drugs", "weed",
    "marijuana", "cocaine", "pill", "pills", "addiction", "bar", "pub",
    # Fear and horror
    "horror", "scary", "terrifying", "nightmare", "ghost", "demon",
    "devil", "hell", "zombie", "monster", "scream", "fear", "afraid",
    "creepy", "haunted", "curse", "cursed",
    # Inappropriate content
    "hate", "stupid", "ugly", "fat", "dumb", "idiot", "fool",
    "steal", "thief", "rob", "crime", "prison", "jail", "liar",
    "cheat", "bully", "cruel", "abuse", "suffer", "pain",
    "angry", "rage", "revenge", "poison", "toxic",
    # Adult themes
    "kiss", "romantic", "sexy", "naked", "divorce",
    "gambling", "casino", "bet",
]

# Mapping of prohibited words to kid-friendly replacements
SAFE_REPLACEMENTS = {
    "kill": "stop", "killed": "stopped", "murder": "trouble",
    "death": "nap", "dead": "sleeping", "die": "rest", "died": "rested",
    "fight": "play", "attack": "surprise visit", "war": "game",
    "blood": "red paint", "evil": "naughty", "dark": "dim",
    "scary": "surprising", "ghost": "friendly spirit",
    "monster": "fluffy creature", "scream": "shout with joy",
    "hate": "dislike", "stupid": "silly", "ugly": "different",
    "steal": "borrow", "cry": "sniffle", "sad": "a little blue",
    "angry": "grumpy", "hurt": "bump", "afraid": "unsure",
    "smoke": "cloud", "smoking": "making clouds",
    "drunk": "sleepy", "drinking": "sipping juice",
    "gun": "water squirter", "knife": "spoon", "sword": "magic wand",
    "poison": "juice", "bomb": "balloon", "weapon": "toy",
    "hell": "oh my", "devil": "silly imp", "demon": "little imp",
    "zombie": "sleepyhead", "nightmare": "funny dream",
    "prison": "time-out room", "jail": "time-out room",
    "bully": "grumpy friend", "cruel": "unkind", "abuse": "mean act",
    "revenge": "lesson", "toxic": "yucky", "curse": "silly spell",
    "rage": "grumpy mood", "pain": "ouch", "wound": "scratch",
    "thief": "sneaky raccoon", "rob": "borrow without asking",
    "beer": "apple juice", "wine": "grape juice", "alcohol": "fizzy drink",
    "whiskey": "warm cocoa", "vodka": "water", "casino": "playground",
    "gambling": "playing games",
}


def check_content_safety(text):
    """
    Check if the generated text contains any prohibited words.
    Args:
        text: The text string to check.
    Returns:
        is_safe: Boolean, True if text is safe for kids.
        flagged_words: List of prohibited words found in the text.
    """
    text_lower = text.lower()
    flagged_words = []
    for word in PROHIBITED_WORDS:
        # Use word boundary matching to avoid false positives
        # e.g., "hit" should not flag "white"
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, text_lower):
            flagged_words.append(word)
    is_safe = len(flagged_words) == 0
    return is_safe, flagged_words


def replace_prohibited_words(text):
    """
    Replace any prohibited words in the text with kid-friendly alternatives.
    Args:
        text: The text string to clean.
    Returns:
        cleaned_text: The cleaned text with prohibited words replaced.
    """
    cleaned_text = text
    for bad_word, good_word in SAFE_REPLACEMENTS.items():
        # Case-insensitive word-boundary replacement
        pattern = re.compile(r'\b' + re.escape(bad_word) + r'\b', re.IGNORECASE)
        cleaned_text = pattern.sub(good_word, cleaned_text)
    return cleaned_text


# ============================================================
# Function part
# ============================================================

# img2text: Extract a caption from the uploaded image
# Model URL: https://huggingface.co/Salesforce/blip-image-captioning-base
def img2text(url):
    """
    Use an image-to-text pipeline to generate a caption from the image.
    Args:
        url: Path to the image file.
    Returns:
        text: Generated caption string.
    """
    image_to_text_model = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )
    text = image_to_text_model(url)[0]["generated_text"]
    return text


# text2story: Generate a kid-friendly story from the caption
# Model URL: https://huggingface.co/pranavpsv/genre-story-generator-v2
def text2story(text):
    """
    Use a text-generation pipeline to create a short, safe, and fun story
    (50-100 words) suitable for children aged 3-10 based on the image caption.
    Includes a carefully designed prompt for child-appropriate content and
    content safety filtering for prohibited words.
    Args:
        text: The image caption string.
    Returns:
        story_text: Generated kid-friendly story string.
    """
    story_pipe = pipeline(
        "text-generation",
        model="pranavpsv/genre-story-generator-v2"
    )

    # Carefully designed prompt for generating child-safe stories
    # Key design choices:
    #   - Explicit age range (3-10) to guide tone and vocabulary
    #   - Positive themes: kindness, friendship, curiosity, sharing
    #   - Explicit exclusions: violence, scary content, adult themes
    #   - Simple language requirement for young readers
    #   - "Once upon a time" opening for familiar story structure
    prompt = (
        f"Write a short, cheerful, and magical bedtime story for little children "
        f"aged 3 to 10. "
        f"The story must be safe, gentle, warm, and fun. "
        f"Use simple and easy words and short sentences that kids can understand. "
        f"The story should only include happy and positive themes like kindness, "
        f"friendship, sharing, curiosity, adventure, and helping others. "
        f"Do not include any violence, scary things, bad words, smoking, "
        f"drinking alcohol, weapons, fighting, or anything inappropriate "
        f"for young children. "
        f"The scene is: {text}. "
        f"Once upon a time,"
    )

    story_results = story_pipe(
        prompt,
        max_length=200,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )
    story_text = story_results[0]["generated_text"]

    # Remove the prompt portion, keep only the story
    if "Once upon a time," in story_text:
        story_text = "Once upon a time," + story_text.split("Once upon a time,", 1)[1]

    # Trim story to stay within the 50-100 word target
    words = story_text.split()
    if len(words) > 100:
        story_text = " ".join(words[:100])
        # End at the last complete sentence if possible
        if "." in story_text:
            story_text = story_text[:story_text.rfind(".") + 1]
        else:
            story_text = story_text + "."

    # Safety check: filter any prohibited words from the generated story
    is_safe, flagged_words = check_content_safety(story_text)
    if not is_safe:
        story_text = replace_prohibited_words(story_text)

    return story_text


# text2audio: Convert the story text into audio
# Model URL: https://huggingface.co/Matthijs/mms-tts-eng
def text2audio(story_text):
    """
    Use a text-to-audio pipeline to convert the story into speech audio.
    Args:
        story_text: The story string to convert to speech.
    Returns:
        audio_data: Dictionary containing 'audio' array and 'sampling_rate'.
    """
    audio_pipe = pipeline(
        "text-to-audio",
        model="Matthijs/mms-tts-eng"
    )
    audio_data = audio_pipe(story_text)
    return audio_data


# ============================================================
# Main part: Streamlit User Interface
# ============================================================

# Page configuration with kid-friendly title and icon
st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")

# App header
st.header("🦜 Turn Your Image to Audio Story")
st.markdown(
    "Welcome, little explorer! 🌈 Upload a picture and let the magic begin! "
    "I will look at your picture, write a fun story, and even read it to you! "
    "🎉"
)

# Sidebar with app instructions for parents and kids
st.sidebar.title("📖 How to Use")
st.sidebar.markdown(
    "1. 📷 Upload a picture below\n"
    "2. 🔍 Wait for the magic to happen\n"
    "3. 📖 Read your fun story\n"
    "4. 🔊 Click **Play Audio** to hear it!\n"
    "5. 🔄 Try another picture anytime!"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "🛡️ **Safe for Kids**\n\n"
    "This app is designed for children aged 3-10. "
    "All generated stories are automatically checked "
    "to make sure they are fun, safe, and appropriate "
    "for young readers."
)

# Image uploader with supported file types
uploaded_file = st.file_uploader(
    "📷 Select an Image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a picture (JPG or PNG) and I will create a magical story!"
)

if uploaded_file is not None:
    # Save file locally for model processing
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    # Display the uploaded image
    st.image(uploaded_file, caption="📷 Your Uploaded Image", use_column_width=True)
    st.markdown("---")

    # ----------------------------------------------------------
    # Stage 1: Image to Text (Using the img2text function)
    # ----------------------------------------------------------
    st.markdown("### 🔍 Stage 1: Understanding Your Image")
    with st.spinner("Looking at your picture..."):
        scenario = img2text(uploaded_file.name)
    st.success("Image processed successfully!")
    st.write(f"**Scenario:** {scenario}")
    st.markdown("---")

    # ----------------------------------------------------------
    # Stage 2: Text to Story (Using the text2story function)
    # ----------------------------------------------------------
    st.markdown("### 📖 Stage 2: Creating Your Story")
    with st.spinner("Writing a magical story just for you..."):
        story = text2story(scenario)
    st.success("Story created successfully!")
    st.write(f"**Story:** {story}")
    st.markdown("---")

    # ----------------------------------------------------------
    # Stage 3: Story to Audio (Using the text2audio function)
    # ----------------------------------------------------------
    st.markdown("### 🔊 Stage 3: Listen to Your Story")
    with st.spinner("Preparing the storyteller's voice..."):
        audio_data = text2audio(story)
    st.success("Audio is ready! Click the button below to listen.")

    # Play button for audio playback
    if st.button("▶️ Play Audio"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)

    st.markdown("---")

    # ----------------------------------------------------------
    # Encourage kids to try another image
    # ----------------------------------------------------------
    st.markdown("### 🎉 Want More Stories?")
    st.markdown(
        "🔄 **Try another picture!** Simply upload a new image above "
        "to create a brand new magical story. "
        "Every picture has a wonderful story waiting to be told! ✨"
    )
    st.markdown(
        "💡 **Ideas to try:** Pictures of animals 🐶, nature 🌺, "
        "toys 🧸, food 🍕, or your favorite places 🏖️!"
    )
    st.balloons()

else:
    # Friendly placeholder when no image is uploaded yet
    st.markdown("---")
    st.info(
        "👆 **Upload a picture above to get started!**\n\n"
        "Try pictures of animals, nature, toys, or your favorite things! "
        "The storyteller is ready and waiting! 🎭"
    )
