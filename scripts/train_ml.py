# ==========================================
# ML TRAINING MODULE FOR COPYRIGHT SYSTEM
# (Final Version with Model Saving)
# ==========================================

import numpy as np
import faiss
import random
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# ------------------------------
# CONFIGURATION
# ------------------------------

TOP_K = 6
NUM_POS = 8000
NUM_NEG = 8000
RANDOM_STATE = 42
BATCH_SIZE = 64
MODEL_TYPE = "logistic"   # choose: "logistic" or "svm"

np.random.seed(RANDOM_STATE)

# ------------------------------
# CREATE MODEL SAVE DIRECTORY
# ------------------------------

SAVE_DIR = "../saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------
# LOAD LIBRISPEECH EMBEDDINGS + FAISS
# ------------------------------

embeddings = np.load("../embeddings/embeddings.npy")
index = faiss.read_index("../database/faiss_index.bin")

# ------------------------------
# LOAD TED NEGATIVE CORPUS
# ------------------------------

ted_sentences = np.load("../embeddings/ted_transcripts.npy", allow_pickle=True)
print("Total TED chunks:", len(ted_sentences))

# ------------------------------
# LOAD SBERT MODEL
# ------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

X = []
y = []

# =====================================
# POSITIVE SAMPLE GENERATION
# =====================================

print("Generating positive samples...")

for i in range(NUM_POS):
    query = embeddings[i].reshape(1, -1)
    D, I = index.search(query, TOP_K)

    sims = D[0][1:6]

    mean = np.mean(sims)
    std = np.std(sims)
    gap = sims[0] - sims[1]

    features = list(sims) + [mean, std, gap]

    X.append(features)
    y.append(1)

print("Positive samples created:", NUM_POS)

# =====================================
# NEGATIVE SAMPLE GENERATION
# =====================================

print("Generating negative samples from TED...")

selected_ted = random.sample(list(ted_sentences), NUM_NEG)

for i in range(0, NUM_NEG, BATCH_SIZE):

    batch = selected_ted[i:i+BATCH_SIZE]

    batch_embeddings = model.encode(
        batch,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE
    )

    D, I = index.search(batch_embeddings, TOP_K)

    for row in D:
        sims = row[0:5]

        mean = np.mean(sims)
        std = np.std(sims)
        gap = sims[0] - sims[1]

        features = list(sims) + [mean, std, gap]

        X.append(features)
        y.append(0)

print("Negative samples created:", NUM_NEG)

# -------------------------------------
# Convert to numpy
# -------------------------------------

X = np.array(X)
y = np.array(y)

# =====================================
# TRAIN / TEST SPLIT
# =====================================

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# =====================================
# FEATURE SCALING
# =====================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# =====================================
# MODEL SELECTION
# =====================================

if MODEL_TYPE == "logistic":
    print("\nTraining Logistic Regression...")
    clf = LogisticRegression(max_iter=2000)
else:
    print("\nTraining SVM (RBF)...")
    clf = SVC(kernel="rbf", probability=True)

clf.fit(X_train, y_train)

# =====================================
# EVALUATION
# =====================================

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print("AUC Score:", roc_auc)

# =====================================
# THRESHOLD BASELINE EVALUATION
# =====================================

print("\n=== Threshold Baseline (s1 > 0.75) ===")

threshold = 0.75
s1_test = X_test_raw[:, 0]

y_pred_threshold = (s1_test > threshold).astype(int)

print(classification_report(y_test, y_pred_threshold))

cm_threshold = confusion_matrix(y_test, y_pred_threshold)
print("Confusion Matrix:\n", cm_threshold)

# =====================================
# SAVE MODEL + SCALER
# =====================================

joblib.dump(clf, os.path.join(SAVE_DIR, "copyright_model.pkl"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "feature_scaler.pkl"))

# Save configuration
config = {
    "top_k": TOP_K,
    "num_pos": NUM_POS,
    "num_neg": NUM_NEG,
    "random_state": RANDOM_STATE,
    "model_type": MODEL_TYPE,
    "feature_count": X.shape[1]
}

joblib.dump(config, os.path.join(SAVE_DIR, "training_config.pkl"))

print("\nModel and scaler saved successfully in:", SAVE_DIR)

