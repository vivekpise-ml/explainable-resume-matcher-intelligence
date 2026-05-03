import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data_loader import create_pairs, load_labels
from src.matcher_training import ResumeJDDataset

from sklearn.metrics import confusion_matrix

from src.domain_role.encoders import domain_map, role_map, DOMAIN_LIST, ROLE_LIST
from src.features.features_config import FEATURE_ORDER

import json


import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =========================================================
# 🔴 HOOKS
# =========================================================
USE_BERT = True   # 🔥 turn OFF to test only structured features
USE_FUSION_MODEL = False # to switch between BertWithFeatureFusion model or HybridModel

# =========================================================
# 🔴 K-FOLD HOOK (NEW - SAFE)
# =========================================================
USE_KFOLD = True
N_SPLITS = 5

# =========================================================
# 🔴 Dropout disabling hook
# =========================================================
DISABLE_DROPOUT = True

# -----------------------------
# Config
# -----------------------------
model_name = "bert-base-uncased"
batch_size = 4
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-base-uncased"
GLOBAL_BERT = BertModel.from_pretrained(model_name)
# -----------------------------
# Load data
# -----------------------------
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Commentiing this below as I am now training for more resumes
#data_dir = os.path.join(BASE_DIR, "data", "raw")

data_dir = os.path.join(BASE_DIR, "data", "raw/more-resumes")

# Commentiing this below as I am now training for more resumes
#csv_path = os.path.join(BASE_DIR, "data", "sap_ta_abap_1_Data.csv")

csv_path = os.path.join(BASE_DIR, "data", "sap_ta_abap_1_Data_new.csv")

print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", data_dir)
print("Exists:", os.path.exists(data_dir))

label_map = load_labels(csv_path)
pairs = create_pairs(data_dir, label_map)


labels = [item["label"] for item in pairs]

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained(model_name)

# -----------------------------
# Load skill files
# -----------------------------
with open("data/annotations/skill_dict.json") as f:
    skill_dict = json.load(f)

with open("data/annotations/skill_graph.json") as f:
    skill_graph = json.load(f)

# Make graph bidirectional (SAFE)
for k, v in list(skill_graph.items()):
    for neighbor in v:
        skill_graph.setdefault(neighbor, []).append(k)

    # GLOBAL_BERT = BertModel.from_pretrained(model_name)

    # =========================================================
    # 🔴 NEW: Controlled Fusion Model (UNCHANGED)
    # =========================================================
    class BertWithFeatureFusion(nn.Module):
        def __init__(self, num_labels, feature_dim):
            super().__init__()

            #self.bert = BertModel.from_pretrained("bert-base-uncased")
            self.bert = GLOBAL_BERT

            self.bert_fc = nn.Linear(768, 128)
            self.feature_fc = nn.Linear(feature_dim, 128)

            self.alpha_layer = nn.Linear(256, 1)

            self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_labels)
            )

        def forward(self, input_ids, attention_mask, extra_features):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            cls_output = outputs.pooler_output

            bert_embed = self.bert_fc(cls_output)
            feature_embed = self.feature_fc(extra_features)

            fusion_input = torch.cat([bert_embed, feature_embed], dim=1)
            alpha = torch.sigmoid(self.alpha_layer(fusion_input))

            fused = alpha * bert_embed + (1 - alpha) * feature_embed

            logits = self.classifier(fused)

            return logits

    # -----------------------------
    # Model
    # -----------------------------
    class HybridModel(nn.Module):
        def __init__(self, extra_dim, num_labels):
            super().__init__()

            if USE_BERT:
                #self.bert = BertModel.from_pretrained(model_name)
                self.bert = GLOBAL_BERT
                input_dim = 768 + extra_dim
            else:
                self.bert = None
                input_dim = extra_dim

            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, num_labels)
            )

        def forward(self, input_ids, attention_mask, extra_features):

            if USE_BERT:
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls = outputs.last_hidden_state[:, 0, :]
                combined = torch.cat((cls, extra_features), dim=1)
            else:
                combined = extra_features

            return self.classifier(combined)


# =========================================================
# 🔴 PIPELINE WRAPPER (ONLY ADDITION)
# =========================================================
def run_pipeline(train_pairs, test_pairs):

    # debug
    #print("SAMPLE TRAIN PAIR:", train_pairs[0])   # 🔥 ADD HERE

    # -----------------------------
    # Dataset
    # -----------------------------
    train_dataset = ResumeJDDataset(train_pairs, tokenizer, skill_dict, skill_graph)
    test_dataset = ResumeJDDataset(test_pairs, tokenizer, skill_dict, skill_graph)

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)

    ### To verify if CSV is off ####
    sample = next(iter(train_loader))

    print("\nSample extra_features:")

    '''
    FEATURE_ORDER = [
        "exact_ratio",
        "related_ratio",
        "coverage_ratio",
        "resume_skill_count_norm",
        "jd_skill_count_norm",
        "exact_vs_related",
        "match_density",
        "balance_score"
    ]
    '''
    # Add domain + role labels dynamically
    feature_names = (
        FEATURE_ORDER
        + [f"domain_{d}" for d in DOMAIN_LIST]
        + [f"role_{r}" for r in ROLE_LIST]
    )
    features = sample['extra_features'][0]

    print("\n===== FEATURE BREAKDOWN =====")
    for name, value in zip(feature_names, features):
        print(f"{name:30s}: {value:.4f}")
    print("============================\n")

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # -----------------------------
    # Get feature size dynamically
    # -----------------------------
    sample = next(iter(train_loader))
    extra_dim = sample['extra_features'].shape[1]

    # debug
    print("🔥 Final Feature Length:", extra_dim)

    import torch.nn as nn
    from transformers import BertModel

    
    if USE_FUSION_MODEL:
        model = BertWithFeatureFusion(num_labels=4, feature_dim=extra_dim).to(device)
    else:
        model = HybridModel(extra_dim, num_labels=4).to(device)

    # 🔴 Disable Dropout (SAFE INSERT POINT)
    if DISABLE_DROPOUT:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    # -----------------------------
    # Training
    # -----------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # now resumes increased from 121 to 240 but is imbalaced as class 3 resumes are quite high hence
    # commenting below weights - which were needed for small dataset like 121 resumes
    # however, in general weights to assigned - this is not needed but for our datasets used
    #weights = torch.tensor([1.0, 1.0, 1.5, 1.0]).to(device)

    from collections import Counter

    # Since static weights removed above, hence this line
    train_labels = [item["label"] for item in train_pairs]

    label_counts = Counter(train_labels)

    
    total = sum(label_counts.values())

    weights = torch.tensor([
        total / label_counts[i] for i in range(4)
    ]).to(device)

    #criterion = nn.CrossEntropyLoss(weight = weights)

    criterion = nn.CrossEntropyLoss(weight = weights, label_smoothing=0.1)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            extra_features = batch['extra_features'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, extra_features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # -----------------------------
    # Evaluation
    # -----------------------------
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            extra_features = batch['extra_features'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, extra_features)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    print("\nAccuracy:", acc)
    print("F1:", f1_score(all_labels, all_preds, average="weighted"))

    print("\nDetailed Report:\n")
    print(classification_report(all_labels, all_preds))

    # -----------------------------
    # 🔴 Confusion Matrix (NEW - SAFE)
    # -----------------------------
    cm = confusion_matrix(all_labels, all_preds)

    print("\nConfusion Matrix:")
    print("Classes: [0, 1, 2, 3]")
    print(cm)

    # -----------------------------
    # Train Evaluation (UNCHANGED)
    # -----------------------------
    train_preds, train_labels = [], []

    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            extra_features = batch['extra_features'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, extra_features)
            preds = torch.argmax(outputs, dim=1)

            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(train_labels, train_preds)
    print("\nTrain Accuracy:", train_acc)

    torch.save(model.state_dict(), "models/hybrid_model.pt")

    return acc


# =========================================================
# 🔴 EXECUTION (ONLY CHANGE)
# =========================================================
# =========================================================
# 🔴 EXECUTION (FIXED - SAFE)
# =========================================================
def train_main():

    if USE_KFOLD:

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

        fold_accuracies = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(pairs, labels)):
            print(f"\n========== Fold {fold+1} ==========")

            train_pairs = [pairs[i] for i in train_idx]
            test_pairs = [pairs[i] for i in test_idx]

            acc = run_pipeline(train_pairs, test_pairs)
            fold_accuracies.append(acc)

        print("\n========== FINAL K-FOLD ==========")
        print(f"Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")

    else:

        train_pairs, test_pairs = train_test_split(
            pairs, test_size=0.2, random_state=42, stratify=labels
        )

        run_pipeline(train_pairs, test_pairs)


# 🔥 CRITICAL: only run when file is executed directly
if __name__ == "__main__":
    train_main()