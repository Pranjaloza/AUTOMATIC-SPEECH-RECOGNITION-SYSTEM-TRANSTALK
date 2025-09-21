# 🎙️ TransTalk: Automatic Speech Recognition & Analysis

TransTalk is an AI-powered **Automatic Speech Recognition (ASR)** system built with **Streamlit** and **Transformers**.  
It transcribes speech from **audio, video, or live microphone input**, detects **sentiment & emotion**, and translates into multiple languages.

---

## 🚀 Features
- 🎤 **Input Methods**
  - Upload Audio File (`.wav`, `.mp3`)
  - Upload Video File (`.mp4`, `.avi`)
  - Live Microphone Input (with Wake-Word Detection using **Porcupine**)
- 📝 **Speech-to-Text** transcription using **OpenAI Whisper**
- 😊 **Sentiment Analysis** (Positive / Negative / Neutral)
- 🎭 **Emotion Detection** (Joy, Anger, Sadness, Fear, Surprise)
- 🌍 **Multi-language Translation** with **Deep Translator**
- 🎬 **Emotion-based GIFs** for visualization
- ⬇️ **Downloadable Transcriptions & Translations**

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) – Web App Framework
- [OpenAI Whisper](https://huggingface.co/openai/whisper-large-v2) – Speech Recognition
- [Transformers](https://huggingface.co/transformers/) – NLP Pipelines
- [Torchaudio](https://pytorch.org/audio/) – Audio Processing
- [MoviePy](https://zulko.github.io/moviepy/) – Extract Audio from Video
- [Deep Translator](https://pypi.org/project/deep-translator/) – Multi-language Translation
- [Porcupine](https://picovoice.ai/platform/porcupine/) – Wake Word Detection
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) – Microphone Input

---

## ⚡ Installation
Clone the repo:
```bash
git clone https://github.com/YOUR_USERNAME/TransTalk.git
cd TransTalk
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the app:
```bash
streamlit run Transtalk.py
```

## 📜 License
