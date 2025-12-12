# Emotion Recognition from Speech (SER)

This repository contains the code and experiments for the project  
**“Emotion Recognition from Speech Using CNNs and Wav2Vec2: A Comparative Study with Supplementary Analysis of Rudeness as a Social Tone Trait.”**

The project investigates how modern self-supervised speech representations compare against traditional spectrogram-based models for Speech Emotion Recognition (SER), with a particular emphasis on **cross-dataset generalization** and **robustness under domain shift**.

---

## 📌 Project Overview

Speech Emotion Recognition (SER) aims to identify human emotions from vocal signals by analyzing prosodic, spectral, and temporal cues. While classical SER approaches rely on handcrafted features and convolutional architectures, recent self-supervised models such as **Wav2Vec2** learn rich representations directly from raw waveforms.

This project has two primary goals:

1. **Comparative Evaluation**  
   Compare a CNN-based mel-spectrogram baseline with a fine-tuned **Wav2Vec2** model for emotion recognition.

2. **Generalization & Social Traits**  
   - Evaluate robustness across datasets (CREMA-D and RAVDESS).
   - Conduct a supplementary exploratory study on **rudeness** as a continuous social tone trait using IEMOCAP.

---

## 📁 Repository Structure

SER/
├── wav2vec2.py # Wav2Vec2 training, evaluation, and inference
├── crema_cnn_baseline.ipynb # CNN baseline using mel-spectrograms
├── iemocap_main.py # Rudeness analysis on IEMOCAP
├── make_report_figures.py # Generates confusion matrices and plots
├── output/ # Saved predictions, reports, and figures
└── README.md # Project documentation

yaml
Copy code

---

## 📊 Datasets

### CREMA-D
- 7,442 utterances from 91 actors
- 6 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad
- Used primarily for training

### RAVDESS
- 1,440 studio-quality recordings
- Distinct recording conditions and speakers
- Used to evaluate **cross-dataset generalization**

### IEMOCAP (Supplementary Study)
- Dyadic conversational dataset
- Used for exploratory annotation and analysis of **rudeness** as a social tone trait

---

## 🧠 Models Implemented

### 1. CNN Baseline
- Input: Log-mel spectrograms
- Architecture: 4 convolutional layers + fully connected classifier
- Trained on CREMA-D, evaluated on RAVDESS
- Serves as a traditional SER baseline

### 2. Wav2Vec2-Based Model
- Pretrained self-supervised speech transformer
- Input: Raw waveform (16 kHz)
- Fine-tuned end-to-end with a linear classification head
- Demonstrates strong transfer and robustness under domain shift

---

## 🧪 Experimental Design (Wav2Vec2)

To analyze the impact of dataset composition, we conduct three experiments:

| Experiment | Train → Test | Accuracy | Macro-F1 |
|----------|--------------|----------|----------|
| Exp-1 | CREMA-D + RAVDESS → Mixed split | 0.87 | 0.86 |
| Exp-2 | CREMA-D → CREMA-D | 0.85 | 0.84 |
| Exp-3 | 80% CREMA-D → (20% CREMA-D + RAVDESS) | 0.82 | 0.82 |

**Key takeaway:**  
Mixed-dataset training improves robustness, while partial domain shift introduces a modest but expected performance drop.

---

## 📈 Key Results

- **CNN baseline**:  
  - Accuracy: 53%  
  - Macro-F1: 0.52  
  - Struggles under cross-dataset evaluation

- **Wav2Vec2**:  
  - Accuracy: ~81%  
  - Macro-F1: ~0.81  
  - Strong per-class performance, especially for Angry and Neutral

- Largest gains observed for low-arousal emotions (Neutral, Disgust, Fear).

---

## 🔍 Supplementary Study: Rudeness Detection

Beyond categorical emotions, we explore **rudeness** as a nuanced social tone trait:

- Manual annotation of rude vs. not-rude samples from IEMOCAP
- PCA and UMAP analysis of Wav2Vec2 embeddings
- Prosodic analysis (spectral centroid, RMS energy)
- Construction of a **rudeness trait vector**:
  
v_rude = mean(E_rude) − mean(E_not_rude)

yaml
Copy code

**Finding:**  
Rudeness does not form a discrete class but occupies a consistent region in embedding space, supporting a **continuous trait-based modeling approach**.

---

## ▶️ Running the Code

### Inspect dataset
```bash
python3 wav2vec2.py inspect --source_dir <AUDIO_DIR>
Create train/val/test split
bash
Copy code
python3 wav2vec2.py split --source_dir <AUDIO_DIR> --target_dir ./dataset_split
Train Wav2Vec2
bash
Copy code
python3 wav2vec2.py train --data_root ./dataset_split --out_dir ./ser_wav2vec2_ckpt
Evaluate model
bash
Copy code
python3 wav2vec2.py test --data_root ./dataset_split --ckpt_dir ./ser_wav2vec2_ckpt
Predict emotion for a single audio file
bash
Copy code
python3 wav2vec2.py predict_wav \
  --wav_path <PATH_TO_WAV> \
  --ckpt_dir ./ser_wav2vec2_ckpt
⚠️ Notes & Limitations
CNN baseline is not state-of-the-art and serves as a reference point.

Rudeness annotations are limited in size and culturally subjective.

Fear and sadness remain challenging due to overlapping acoustic cues.

🚀 Future Work
Multitask learning for emotion + social tone traits

Larger-scale rudeness annotation

Multimodal fusion (text, facial expressions)

Prosody-aware or attention-based pooling strategies

👩‍💻 Authors
Swathi Arasu
MS Data Science, Northeastern University

Shreya Chavan
MS Data Science, Northeastern University     
