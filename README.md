# 🎤 Arabic ASR with Whisper Small

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Fine--Tuned-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

---

## 📌 Project Overview

This project fine-tunes OpenAI's `whisper-small` model on the **Arabic Speech Corpus** to build a robust Arabic Automatic Speech Recognition (ASR) system. The model takes raw Arabic audio as input and outputs clean, normalized Arabic transcriptions. It was built to bridge the gap for Arabic speakers who need accurate, open-source speech-to-text without relying on closed APIs.

---

## 🖥️ Demo

> **Drop your screenshot or screen recording here.**
> Replace the placeholder below with an actual image or GIF of your Streamlit app in action.

```
[ Screenshot / GIF Placeholder ]
Place your demo image here: docs/demo.gif
```

---

## ⚡ Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/arabic-asr-whisper.git
cd arabic-asr-whisper
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **requirements.txt** should include:
> `transformers==4.44.2`, `accelerate==0.33.0`, `datasets`, `evaluate`, `jiwer`, `librosa`, `streamlit`, `torch`

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. Upload a `.wav` file or record directly through the interface to get an instant Arabic transcription.

---

## 🧠 Model Training Details

### Dataset
The model was trained on the [**Arabic Speech Corpus**](http://www.arabicspeechcorpus.com/) — a high-quality Modern Standard Arabic (MSA) dataset containing paired WAV audio files and orthographic transcriptions in Buckwalter transliteration format.

### Base Model
`openai/whisper-small` — a compact yet capable multilingual ASR model from OpenAI, fine-tuned here specifically for the Arabic language with the `transcribe` task.

### Training Highlights

| Detail | Value |
|---|---|
| Base Model | `openai/whisper-small` |
| Language | Arabic (`ar`) |
| Max Steps | 600 |
| Learning Rate | `2e-5` |
| Batch Size | 8 (with 2 gradient accumulation steps) |
| Evaluation Metric | WER (Word Error Rate) |
| Optimizer | AdamW + weight decay `0.01` |

### Key Techniques

- **Buckwalter → Arabic conversion** — The dataset transcriptions are stored in Buckwalter transliteration. A custom `buckwalter_to_arabic()` function converts them to proper Unicode Arabic before training.
- **Arabic text normalization** — Diacritics (tashkeel) are stripped, and characters like `أ / إ / آ → ا`, `ى → ي`, and `ة → ه` are unified to reduce vocabulary noise.
- **Encoder freezing** — The Whisper encoder is frozen during training so only the decoder weights are updated, dramatically reducing memory usage and training time while preserving the pre-trained audio representations.
- **SpecAugment** — Applied via `mask_time_prob=0.05` and `mask_feature_prob=0.05` to prevent overfitting on the relatively small dataset.
- **Custom Data Collator** — A hand-crafted `DataCollatorSpeechSeq2SeqWithPadding` class handles proper label padding with `-100` masking and performs a manual right-shift to build `decoder_input_ids`, solving gradient checkpointing conflicts that arise with Whisper's default collation.
- **Early Stopping** — Training halts automatically if WER stops improving for 3 consecutive epochs, preventing unnecessary compute waste.
- **Google Drive checkpointing** — All checkpoints and the final model are saved directly to Google Drive to survive Colab session resets.
