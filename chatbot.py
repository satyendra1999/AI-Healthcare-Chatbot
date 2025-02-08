import streamlit as st
import time
from transformers import pipeline
import nltk
from pathlib import Path

# Cache NLTK downloads
if not Path("nltk_cache").exists():
    nltk.download("punkt", download_dir="nltk_cache")
    nltk.download("stopwords", download_dir="nltk_cache")
nltk.data.path.append("nltk_cache")

# Load Healthcare-specific Model
@st.cache_resource
def load_chatbot():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

chatbot = load_chatbot()

# Manual Responses for Quick Answers (Hindi + English)
manual_responses = {
   "hello": {
    "en": "Namaste!! How can I assist you today?",
    "hi": "नमस्ते!! मैं आज आपकी कैसे मदद कर सकता हूँ?"
    },
    "fever": {
        "en": "A fever is usually a sign of infection. Stay hydrated, rest, and monitor your temperature. Consult a doctor if it persists for more than 3 days.",
        "hi": "बुखार आमतौर पर संक्रमण का संकेत होता है। पानी पिएं, आराम करें और तापमान को मॉनिटर करें। यदि यह 3 दिनों से अधिक बना रहे, तो डॉक्टर से संपर्क करें।"
    },
    "cough": {
        "en": "A dry cough could be due to allergies, while a wet cough may indicate infection. Drink warm fluids and consult a doctor if severe.",
        "hi": "सूखी खांसी एलर्जी के कारण हो सकती है, जबकि गीली खांसी संक्रमण का संकेत हो सकती है। गर्म तरल पदार्थ पिएं और गंभीर होने पर डॉक्टर से सलाह लें।"
    },
    "headache": {
        "en": "Headaches can be due to stress, dehydration, or lack of sleep. Try resting, drinking water, or taking a mild pain reliever.",
        "hi": "सिरदर्द तनाव, निर्जलीकरण या नींद की कमी के कारण हो सकता है। आराम करें, पानी पिएं या हल्की दर्द निवारक दवा लें।"
    },
    "appointment": {
        "en": "You can book an appointment by calling your nearest hospital or using an online healthcare booking service.",
        "hi": "आप अपने नजदीकी अस्पताल में कॉल करके या ऑनलाइन हेल्थकेयर बुकिंग सेवा का उपयोग करके अपॉइंटमेंट बुक कर सकते हैं।"
    },
    "book appointment": {
        "en": "You can book an appointment by calling your nearest hospital or using an online healthcare booking service.",
        "hi": "आप अपने नजदीकी अस्पताल में कॉल करके या ऑनलाइन हेल्थकेयर बुकिंग सेवा का उपयोग करके अपॉइंटमेंट बुक कर सकते हैं।"
    },
    
    "cancel appointment": {
        "en": "You can cancel your appointment by contacting the hospital or through the online booking platform you used.",
        "hi": "आप अपने अस्पताल से संपर्क करके या जिस ऑनलाइन बुकिंग प्लेटफॉर्म का उपयोग किया था, उसके माध्यम से अपॉइंटमेंट रद्द कर सकते हैं।"
    },
    
    "reschedule appointment": {
        "en": "To reschedule, call the hospital or visit the website where you booked your appointment.",
        "hi": "अपॉइंटमेंट को रीशेड्यूल करने के लिए, अस्पताल को कॉल करें या जिस वेबसाइट से बुकिंग की थी, वहां जाएं।"
    },
    
    "appointment documents": {
        "en": "You may need an ID proof, previous medical reports, and a referral letter if required.",
        "hi": "आपको पहचान पत्र, पिछले मेडिकल रिपोर्ट्स और यदि आवश्यक हो तो रेफरल पत्र की आवश्यकता हो सकती है।"
    },
    
    "walk-in appointment": {
        "en": "Some hospitals allow walk-in consultations, but booking an appointment is recommended to avoid long waiting times.",
        "hi": "कुछ अस्पताल वॉक-इन परामर्श की अनुमति देते हैं, लेकिन लंबी प्रतीक्षा से बचने के लिए अपॉइंटमेंट बुक करना बेहतर होता है।"
    }
}


# Function to Detect Language
def detect_language(text):
    hindi_chars = set("अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
    return "hi" if any(char in hindi_chars for char in text) else "en"

# Healthcare Chatbot Logic
def healthcare_chatbot(user_input):
    user_input = user_input.lower()
    lang = detect_language(user_input)
    
    # Check for quick responses first
    for key in manual_responses.keys():
        if key in user_input:
            return manual_responses[key][lang]
    
    # AI-generated response
    response = chatbot(question=user_input, context="This is a healthcare chatbot providing medical advice.")
    return response["answer"]

# Typing Animation Effect with st.empty()
def typing_effect(text):
    displayed_text = ""
    placeholder = st.empty()  # Create a placeholder to update text dynamically
    for char in text:
        displayed_text += char
        placeholder.markdown(f"**Healthcare Assistant:** {displayed_text}█", unsafe_allow_html=True)
        time.sleep(0.03)

# Streamlit Web App with Animation
def main():
    st.title("🚑 AI-Healthcare Assistant Chatbot (हिंदी + English)")

    # Custom Styling for Chat UI
    st.markdown(
        """
        <style>
        .stChatMessage {
            background-color: #f0f2f6;
            border-radius: 12px;
            padding: 10px;
            margin-bottom: 8px;
        }
        .stChatInput input {
            font-size: 16px;
            border-radius: 10px;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    user_input = st.chat_input("How can I assist you today? | आज मैं आपकी कैसे मदद कर सकता हूँ?")
    
    if user_input:
        st.write("👤 **User:**", user_input)
        
        with st.spinner("💡 Processing your query... | आपके प्रश्न का उत्तर खोजा जा रहा है..."):
            time.sleep(1.5)  # Simulating processing time
            response = healthcare_chatbot(user_input)

        typing_effect(response)  # Display response with animation

if __name__ == "__main__":
    main()
