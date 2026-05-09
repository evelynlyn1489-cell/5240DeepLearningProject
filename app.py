# Program title: Storytelling App
# This app is made for kids aged 3-10.

# Import part
import re
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Safety part: words that kids should not see or hear

# These are bad words we don't want in any story for little kids
prohibited_words = [
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

# If a bad word shows up, we swap it with a nice word instead
safe_replacements = {
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
    "scream": "shout",
}


def check_and_clean(text):
    """Go through the text, find any bad words, and replace them with safe ones."""
    cleaned = text
    for bad_word, good_word in safe_replacements.items():
        # \b makes sure we match whole words only (e.g. "hit" won't match "white")
        pattern = re.compile(r'\b' + re.escape(bad_word) + r'\b', re.IGNORECASE)
        cleaned = pattern.sub(good_word, cleaned)
    return cleaned


# Function part

# img2text: look at the picture and describe what's in it
# Model used: https://huggingface.co/Salesforce/blip-image-captioning-base
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text


# text2story: take that description and write a fun story for kids
# Model used: https://huggingface.co/roneneldan/TinyStories-33M
# Tokenizer: https://huggingface.co/EleutherAI/gpt-neo-125M
def text2story(text):
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    # This prompt tells the model to start a happy, kid-friendly story
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

    # Keep the story between 50-100 words so it's not too long for little kids
    words = story_text.split()
    if len(words) > 100:
        story_text = " ".join(words[:100])
        # Try to end at the last full sentence
        if "." in story_text:
            story_text = story_text[:story_text.rfind(".") + 1]
        else:
            story_text += "."

    # Run safety check to remove any bad words
    story_text = check_and_clean(story_text)

    return story_text


# text2audio: read the story out loud
# Model used: https://huggingface.co/Matthijs/mms-tts-eng
def text2audio(story_text):
    audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story_text)
    return audio_data


# Main part

# Set up the page
st.set_page_config(page_title="Your Image to Audio Story", page_icon="https://icons8.com/icon/114461/story-book")
st.header("🌈 Magic Story Time! 🧸")
st.markdown("Hi there, little friend! 🎉 Pick a picture and I'll tell you a fun story!")

# Let the kid upload a picture
uploaded_file = st.file_uploader("🖼️ Choose your favorite picture!", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file so the model can read it
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    # Show the picture on screen
    st.image(uploaded_file, caption="Your Picture", use_column_width=True)

    # Step 1: Look at the picture and describe it
    with st.spinner("🔍 Let me look at your picture..."):
        scenario = img2text(uploaded_file.name)
        # Clean the description too, in case the model sees something inappropriate
        scenario = check_and_clean(scenario)

    # Step 2: Write a story based on what's in the picture
    with st.spinner("✨ Writing a magical story for you..."):
        story = text2story(scenario)

    # Show the story to the kid
    st.write(f"**📖 Your Story:** {story}")

    # Step 3: Turn the story into audio so kids can listen
    with st.spinner("Getting the storyteller ready..."):
        audio_data = text2audio(story)

    # Button to play the audio
    if st.button("🔊 Read the Story to Me!"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)

    # Tell the kid they can try again
    st.markdown("---")
    st.markdown("🌟 **Wow, that was fun! Want to hear another story? Pick a new picture above!** 🎈")

else:
    # When no picture is uploaded yet, show a friendly hint
    st.markdown("---")
    st.markdown("👆 **Pick a picture to start the magic!** Try a photo of your pet 🐱, a flower 🌻, or your toy 🧸!")
