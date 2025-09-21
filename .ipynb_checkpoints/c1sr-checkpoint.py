import asyncio
import streamlit as st
import torch
import torchaudio
import numpy as np
import cv2
import os
import speech_recognition as sr
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import noisereduce as nr
import soundfile as sf
import pvporcupine
import pyaudio
import struct
from googletrans import Translator
from moviepy.video.io.VideoFileClip import VideoFileClip
import time

# Custom Styling with Background and Fonts
st.markdown(
    """
    <style>
        body {
            background-color: #121212;
            color: white;
            font-family: Arial, sans-serif;
        }
        .stTitle {
            font-size: 36px;
            font-weight: bold;
            color: #ff4b4b;
            text-align: center;
        }
        .stSidebar {
            background-color: #1e1e1e;
            color: white;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar for Navigation
st.sidebar.markdown("<h1 style='color:#ff4b4b;'>TransTalk</h1>", unsafe_allow_html=True)
st.sidebar.image("C:/Users/Pranjal Oza/Downloads/human_13852054.png", width=100)
st.sidebar.markdown("### Options", unsafe_allow_html=True)
input_method = st.sidebar.radio("Choose Input Method", ["Upload Audio File", "Upload Video File", "Live Audio"])

# Load Sentiment & Emotion Analysis Models
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", framework="pt")
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", framework="pt")
translator = Translator()

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform.mean(dim=0, keepdim=True))
    return waveform.squeeze().numpy(), 16000

def reduce_noise(audio, rate):
    return nr.reduce_noise(y=audio, sr=rate)

def extract_audio_from_video(video_path, output_audio_path="extracted_audio.wav"):
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            raise ValueError("No audio found in the video.")
        video.audio.write_audiofile(output_audio_path)
        return output_audio_path
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None

def recognize_live_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source)
        with open("live_audio.wav", "wb") as f:
            f.write(audio_data.get_wav_data())
        return "live_audio.wav"

# Keyword Spotting (Wake Word Detection)
def detect_wake_word(timeout=20):
    start_time = time.time()
    porcupine = pvporcupine.create(access_key="G01a+afHqzWPIh74GrxkuOKYq3QSNrlehhyi08XHqYjZvbSdh62FMA==", keyword_paths=["C:/Users/Pranjal Oza/Downloads/Hey-Marvel_en_windows_v3_0_0/Hey-Marvel_en_windows_v3_0_0.ppn"])
    pa = pyaudio.PyAudio()
    stream = pa.open(rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=porcupine.frame_length)

    st.write("Listening for wake word... (Timeout in 20 seconds)")
    while time.time() - start_time < timeout:
        pcm = stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            st.write("Wake word detected!")
            break
    else:
        st.error("Wake word not detected. Please try again.")
    
    stream.close()
    pa.terminate()

def transcribe_audio(audio_file, language="en"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(device)
    
    audio, rate = load_audio(audio_file)
    audio_denoised = reduce_noise(audio, rate)
    input_features_denoised = processor(audio_denoised, return_tensors="pt", sampling_rate=16000).input_features.to(device)
    forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language=language, task="transcribe")
    
    generated_ids_denoised = model.generate(input_features_denoised, forced_decoder_ids=forced_decoder_ids)
    transcription_denoised = processor.batch_decode(generated_ids_denoised, skip_special_tokens=True)[0]
    
    return transcription_denoised

audio_file = None

if input_method == "Upload Audio File":
    uploaded_audio = st.file_uploader("📂 Upload an audio file", type=["wav", "mp3"])
    if uploaded_audio:
        audio_file = "uploaded_audio.wav"
        with open(audio_file, "wb") as f:
            f.write(uploaded_audio.read())

elif input_method == "Upload Video File":
    uploaded_video = st.file_uploader("📹 Upload a video file", type=["mp4", "avi"])
    if uploaded_video:
        video_file = "uploaded_video.mp4"
        with open(video_file, "wb") as f:
            f.write(uploaded_video.read())
        extracted_audio = extract_audio_from_video(video_file)
        if extracted_audio:
            audio_file = extracted_audio


elif input_method == "Live Audio":
    detect_wake_word()
    audio_file = recognize_live_speech()

if audio_file:
    if "transcribed_text" not in st.session_state:
        with st.spinner("⏳ Processing audio..."):
            st.session_state.transcribed_text = transcribe_audio(audio_file)
            with open("transcription.txt", "w", encoding="utf-8") as text_file:
                text_file.write(st.session_state.transcribed_text)
        st.success("✅ Transcription saved to `transcription.txt`.")

if "transcribed_text" in st.session_state and st.session_state.transcribed_text.strip():
    st.markdown("## 🎙️ Transcribed Text", unsafe_allow_html=True)
    st.success(st.session_state.transcribed_text)
    col1, col2 = st.columns(2)
    with col1:
        sentiment_result = sentiment_analyzer(st.session_state.transcribed_text)[0]
        st.markdown(f"**Sentiment:** `{sentiment_result['label']}`")
    with col2:
        emotion_result = emotion_analyzer(st.session_state.transcribed_text)[0]
        st.markdown(f"**Emotion:** `{emotion_result['label']}`")
            
        emotion_gif = {
                "joy": "https://media.giphy.com/media/fPRwBcYd71Lox1v7p2/giphy.gif",
                "anger": "https://media.giphy.com/media/3o6Zt481isNVuQI1l6/giphy.gif",
                "sadness": "https://media.giphy.com/media/d2lcHJTG5Tscg/giphy.gif",
                "fear": "https://media.giphy.com/media/l2JehQ2GitHGdVG9y/giphy.gif",
                "surprise": "https://media.giphy.com/media/1BXa2alBjrCXC/giphy.gif"
            }


        gif_url = emotion_gif.get(emotion_result['label'].lower(), "https://media.giphy.com/media/3o6ZsYRFmeIUbfnBfy/giphy.gif")
        st.markdown(f'<img src="{gif_url}" width="200" alt="emotion gif">', unsafe_allow_html=True)
        language_options = {
            'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Hindi': 'hi',
            'Gujarati': 'gu', 'Marathi': 'mr', 'Chinese': 'zh-cn', 'Japanese': 'ja',
            'Russian': 'ru', 'Korean': 'ko', 'Arabic': 'ar', 'Portuguese': 'pt'
         }
    target_lang = st.selectbox("🌍 Choose target language", options=list(language_options.keys()), index=1)
    if st.button("🌐 Translate Text"):
        lang_code = language_options[target_lang]
        translated_text = translator.translate(st.session_state.transcribed_text, dest=lang_code).text
        st.markdown("## 🌐 Translated Text", unsafe_allow_html=True)
        st.info(translated_text)





