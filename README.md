# CEG3004 DSP Mini-Project: Environmental Sound Classification

Group ID: Pr_6

## Project Description
A machine learning pipeline that classifies environmental sounds into 50 categories 
(e.g. dog, chirping birds, vacuum cleaner) from short audio clips. The model is trained 
on 1200 labeled `.wav` files and predicts on unseen test data including clean, noisy, 
and band-limited audio variants.

## Requirements
- Python 3.x
- numpy
- scipy
- pandas
- scikit-learn
- librosa
- soundfile
- tqdm
- gdown
- joblib

Install all dependencies with:
```
pip install numpy scipy pandas scikit-learn librosa soundfile tqdm gdown joblib
```

## Dataset Structure
```
data/
├── train/
│   ├── audio/        ← 1200 labeled .wav files
│   └── labels.csv    ← clip_id, label (50 classes)
└── submission/
    ├── audio/        ← unlabeled .wav files (__clean, __noisy, __bandlimited)
    └── metadata.csv
```

## How to Run
1. Open the notebook in **Google Colab**
2. Run **Section 1** to install dependencies
3. Run **Section 2** to download and extract the dataset from Google Drive
4. In **Section 3**, set `DATA_ROOT` to your Google Drive path (e.g. `/content/drive/MyDrive`)
5. Run all remaining cells in order to:
   - Load and visualize audio
   - Preprocess and extract features
   - Train the model
   - Generate predictions

## DSP Design Decisions

### Why Pre-emphasis Filter?
Amplifies high frequencies before feature extraction, compensating for
natural roll-off in environmental sounds. Improves MFCC quality.

### Why Silence Trimming?
Removes non-informative silence at the start/end of clips, ensuring the
model focuses on the actual sound event rather than dead audio.

### Why MFCCs + Log-Mel + ZCR Combined?
- **MFCCs** capture timbral texture of the sound
- **Log-Mel spectrogram** captures spectral shape across frequency bands
- **Zero Crossing Rate** distinguishes tonal vs percussive/noisy sounds

Combining all three features improves robustness across clean, noisy,
and band-limited audio conditions.

## Pipeline Details

### Preprocessing (Section 5)
- Silence trimming (`top_db=25`)
- Pre-emphasis filter (`coeff=0.97`)
- Peak normalization
- Fixed-length padding/truncation to 5 seconds at 16000 Hz

### Feature Extraction (Section 6)
Feature vector of size **208** per clip, combining:
- **MFCC stats** — 20 MFCCs + deltas + delta-deltas (mean & std)
- **Log-mel spectrogram stats** — 40 mel bands (mean & std)
- **Spectral stats** — zero-crossing rate (mean & std)

### Model (Section 8)
- **Classifier:** ExtraTreesClassifier
- **Hyperparameters:** 500 estimators, `max_features='sqrt'`, `random_state=42`
- **Pipeline:** StandardScaler → ExtraTreesClassifier
- **Validation split:** 80/20 stratified

## Experiments

| Model | Validation Macro-F1 |
|-------|-------------------|
| Logistic Regression (baseline) | ~0.40 |
| ExtraTreesClassifier (final) | 0.61 |

ExtraTrees was chosen because it trains faster than RandomForest,
handles high-dimensional feature vectors well, and is less prone to overfitting.

## Robustness to Distortions
The pipeline is designed to handle noisy and band-limited audio because:
- MFCCs are relatively noise-resistant as they capture spectral shape, not raw signal
- Peak normalization handles amplitude distortions
- Combining multiple feature types reduces dependence on any single frequency range

## Limitations
- Model trained only on clean audio without augmentation
- Performance may drop on heavily distorted clips
- Fixed 5-second length may truncate longer meaningful sound events

## Output Files (Auto-Generated)
- `Pr_6_model.joblib` — saved trained model
- `Pr_6_predictions.csv` — predictions on submission set

## Reproducing Results
Run the notebook top-to-bottom in Google Colab with the dataset
downloaded via the provided Google Drive link. All outputs are
automatically saved and downloaded at the end of the notebook.
