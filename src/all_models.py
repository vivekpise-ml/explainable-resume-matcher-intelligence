import torch
import torch.nn as nn
from transformers import BertModel

# =========================================================
# 🔴 HOOKS
# =========================================================
USE_BERT = True

# -----------------------------
# Config
# -----------------------------
model_name = "bert-base-uncased"

# -----------------------------
# Shared BERT
# -----------------------------
GLOBAL_BERT = BertModel.from_pretrained(model_name)

# =========================================================
# 🔴 Controlled Fusion Model (NEW: Controlled Fusion Model (UNCHANGED))
# =========================================================
class BertWithFeatureFusion(nn.Module):

    def __init__(self, num_labels, feature_dim):
        super().__init__()

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


# =========================================================
# 🔴 Hybrid Model
# =========================================================
class HybridModel(nn.Module):

    def __init__(self, extra_dim, num_labels):
        super().__init__()

        if USE_BERT:
            self.bert = GLOBAL_BERT
            input_dim = 768 + extra_dim
        else:
            self.bert = None
            input_dim = extra_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, extra_features):

        if USE_BERT:

            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            cls = outputs.last_hidden_state[:, 0, :]

            combined = torch.cat((cls, extra_features), dim=1)

        else:
            combined = extra_features

        return self.classifier(combined)
