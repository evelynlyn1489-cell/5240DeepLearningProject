# Program title: Storytelling App
# Description: A storytelling app for kids aged 3-10.
#              Upload an image -> generate a story -> listen to it!

# Import part
import re
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Safety: Prohibited words not suitable for kids aged 3-10
# ============================================================
PROHIBITED_WORDS = [
    "kill", "killed", "murder", "blood", "death", "dead", "die", "died",
    "weapon", "gun", "knife", "sword", "fight", "attack", "war", "bomb",
    "shoot", "shot", "violent", "violence", "destroy", "evil",
    "alcohol", "beer", "wine", "whiskey", "vodka", "drunk",
    "smoke", "smoking", "cigarette", "tobacco", "drug", "drugs",
    "horror", "scary", "terrifying", "nightmare", "ghost", "demon",
    "devil", "hell", "zombie", "scream", "creepy", "haunted",
    "hate", "stupid", "ugly", "dumb", "idiot",
    "steal", "thief", "crime", "prison", "jail",
    "bully", "cruel", "abuse", "poison", "toxic",
    "sexy", "naked", "gambling", "casino",
]

SAFE_REPLACEMENTS = {
    "kill": "stop", "killed": "stopped", "murder": "trouble",
    "death": "nap", "dead": "sleeping", "die": "rest", "died": "rested",
    "fight": "play", "attack": "surprise", "war": "game",
    "blood": "red paint", "evil": "naughty",
    "scary": "surprising", "ghost": "friendly spirit",
    "smoke": "cloud", "smoking": "making clouds",
    "drunk": "sleepy", "gun": "toy", "knife": "spoon",
    "sword": "magic wand", "poison": "juice", "bomb": "balloon",
    "beer": "apple juice", "wine": "grape juice", "alcohol": "fizzy drink",
    "cigarette": "lollipop", "tobacco": "candy",
    "hell": "oh my", "devil": "little imp", "demon": "little imp",
    "zombie": "sleepyhead", "nightmare": "funny dream",
    "prison": "time-out room", "jail": "time-out room",
    "hate": "dislike", "stupid": "silly", "ugly": "different",
    "bully": "grumpy friend", "cruel": "unkind",
    "scary": "surprising", "scream": "shout",
}


def check_and_clean(text):
    """Check for prohibited words and replace them with safe alternatives."""
    cleaned = text
    for bad_word, good_word in SAFE_REPLACEMENTS.items():
        pattern = re.compile(r'\b' + re.escape(bad_word) + r'\b', re.IGNORECASE)
        cleaned = pattern.sub(good_word, cleaned)
    return cleaned


# ============================================================
# Function part
# ============================================================

# img2text
# Model: https://huggingface.co/Salesforce/blip-image-captioning-base
def img2text(url):
    """Generate a caption from the uploaded image."""
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text


# text2story
# Model: https://huggingface.co/roneneldan/TinyStories-33M
def text2story(text):
    """Generate a kid-friendly short story based on the image caption."""
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    # Prompt designed for safe, fun, kid-friendly stories
    prompt = (
        f"Once upon a time, there was {text}. "
        f"It was a bright sunny day and everyone was happy. "
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=150,
        num_beams=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    story_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Trim to 50-100 words
    words = story_text.split()
    if len(words) > 100:
        story_text = " ".join(words[:100])
        if "." in story_text:
            story_text = story_text[:story_text.rfind(".") + 1]
        else:
            story_text += "."

    # Safety filter: clean any prohibited words
    story_text = check_and_clean(story_text)

    return story_text


# text2audio
# Model: https://huggingface.co/Matthijs/mms-tts-eng
def text2audio(story_text):
    """Convert the story text into speech audio."""
    audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story_text)
    return audio_data


# ============================================================
# Main part
# ============================================================
st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
st.header("🦜 Turn Your Image to Audio Story")
st.markdown("A fun storytelling app for kids! Upload a picture and hear a story!")

uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save file locally
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Stage 1: Image to Text
    st.text('Processing img2text...')
    scenario = img2text(uploaded_file.name)
    # Clean the caption before showing it (e.g. "smoking" -> "making clouds")
    scenario = check_and_clean(scenario)
    st.write(f"**Scenario:** {scenario}")

    # Stage 2: Text to Story
    st.text('Generating a story...')
    story = text2story(scenario)
    st.write(f"**Story:** {story}")

    # Stage 3: Story to Audio
    st.text('Generating audio data...')
    audio_data = text2audio(story)

    # Play button
    if st.button("Play Audio"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)

    # Encourage trying another image
    st.markdown("---")
    st.markdown("🔄 **Want another story? Upload a new image above!**")
