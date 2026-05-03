import torch
from torch.utils.data import Dataset
import re
import math

# =========================================================
# 🔴 HOOKS (toggle easily for experiments)
# =========================================================
USE_DYNAMIC_SKILLS = True      # skill_dict + skill_graph
USE_CSV_FEATURES = False        # CSV scores
#USE_EXPERIENCE = True
#USE_QUALIFICATION = True


# -----------------------------
# Skill extraction
# -----------------------------
def extract_skills(text, skill_dict):
    text = text.lower()
    found = set()

    for skill in skill_dict:
        if skill.lower() in text:
            found.add(skill)

    return found


# -----------------------------
# Graph matching (legacy - kept)
# -----------------------------
def is_related(skill, resume_skills, skill_graph):
    related = skill_graph.get(skill, [])
    return any(rs in related for rs in resume_skills)


# -----------------------------
# Normalization (NEW - for future)
# -----------------------------
def normalize(s):
    return s.lower().strip()


# -----------------------------
# Multi-hop graph traversal (kept for future)
# -----------------------------
def get_related_multi_hop(skill, skill_graph, max_depth=2):
    skill = normalize(skill)

    visited = set()
    queue = [(skill, 0)]
    hop_dict = {}

    while queue:
        node, depth = queue.pop(0)

        if depth >= max_depth:
            continue

        for nei in skill_graph.get(node, []):
            nei_norm = normalize(nei)

            if nei_norm not in visited:
                visited.add(nei_norm)
                hop_dict[nei_norm] = depth + 1
                queue.append((nei_norm, depth + 1))

    return hop_dict

# -----------------------------
# Make graph bidirectional (NEW) - For future
# -----------------------------
def make_bidirectional(graph):
    new_graph = dict(graph)

    for k, vals in graph.items():
        for v in vals:
            if v not in new_graph:
                new_graph[v] = []
            if k not in new_graph[v]:
                new_graph[v].append(k)

    return new_graph


# -----------------------------
# Experience extraction (fallback)
# -----------------------------
def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*years', text.lower())
    return max([int(m) for m in matches], default=0)


# -----------------------------
# Safe value
# -----------------------------
def safe_value(val, fallback):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return fallback, 1.0
    return val, 0.0


# -----------------------------
# Feature computation
# -----------------------------
def compute_features(item, skill_dict, skill_graph):
    resume = item['resume']
    jd = item['jd']

    # -----------------------------
    # Dynamic skill features
    # -----------------------------
    if USE_DYNAMIC_SKILLS:
        resume_skills = extract_skills(resume, skill_dict)
        jd_skills = extract_skills(jd, skill_dict)

        required = len(jd_skills)

        exact = resume_skills & jd_skills

        '''
        related = set()
        for s in jd_skills:
            if s not in exact and is_related(s, resume_skills, skill_graph):
                related.add(s)
        '''

        # -----------------------------
        # Multi-hop matching (controlled use)
        # -----------------------------
        related_1hop = set()
        related_2hop = set()   # kept for future

        for s in jd_skills:
            if s in exact:
                continue

            # Only 1-hop logic is effectively used
            neighbors = skill_graph.get(s, [])

            for rs in resume_skills:
                if rs in neighbors:
                    related_1hop.add(s)

        exact_count = len(exact)

        related_1hop_count = len(related_1hop)

        # 🔴 IMPORTANT: Only 1-hop used
        related_count = related_1hop_count

        matched = exact_count + related_count

        # ----------- WEIGHTED MATCH -----------
        weighted_match = (
            exact_count * 1.0 +
            related_1hop_count * 0.7
        )

        missing = required - weighted_match
        missing = max(missing, 0.0)

    else:
        resume_skills = set()
        jd_skills = set()
        exact_count = related_count = matched = missing = required = 0.0
        weighted_match = 0.0

    # -----------------------------
    # CSV skills (clean handling)
    # -----------------------------
    if USE_CSV_FEATURES:
        raw_skill = item.get('skill_score')
        skill_missing = 1 if raw_skill is None else 0
        skill_score = 0.0 if raw_skill is None else raw_skill / 100
    else:
        skill_score = 0.0
        skill_missing = 1

    # -----------------------------
    # CSV features - qualification and experience
    # -----------------------------
    exp_csv = item.get('experience_score')
    exp_missing = 1 if exp_csv is None else 0
    exp_csv = 0.0 if exp_csv is None else exp_csv / 100

    qual_csv = item.get('qualification_score')
    qual_missing = 1 if qual_csv is None else 0
    qual_csv = 0.0 if qual_csv is None else qual_csv / 100

    EXP_WEIGHT = 0.05
    QUAL_WEIGHT = 0.05

    # -----------------------------
    # Build feature vector
    # -----------------------------
    feature_dict = {}

    if USE_DYNAMIC_SKILLS:

        # ----------- CORE MATCH FEATURES -----------
        feature_dict["exact_ratio"] = exact_count / (required + 1e-5)

        feature_dict["related_ratio"] = related_count / (required + 1e-5)

        feature_dict["coverage_ratio"] = matched / (required + 1e-5)

        feature_dict["missing_ratio"] = missing / (required + 1e-5)

        feature_dict["weighted_coverage"] = weighted_match / (required + 1e-5)

        # ----------- STRUCTURE FEATURES -----------
        feature_dict["resume_skill_count_norm"] = len(resume_skills) / 20
        feature_dict["jd_skill_count_norm"] = required / 20

        ratio = exact_count / (related_count + 1e-5)

        feature_dict["exact_vs_related"] = math.log1p(ratio) / 2.0

        feature_dict["match_density"] = matched / (len(resume_skills) + 1e-5)

        feature_dict["balance_score"] = (
            min(exact_count, related_count) /
            (max(exact_count, related_count) + 1e-5)
        )

    else:
        for key in [
            "exact_ratio",
            "related_ratio",
            "coverage_ratio",
            "weighted_coverage",
            "resume_skill_count_norm",
            "jd_skill_count_norm",
            "exact_vs_related",
            "match_density",
            "balance_score"
        ]:
            feature_dict[key] = 0.0

    # -----------------------------
    # CSV features
    # -----------------------------
    if USE_CSV_FEATURES:
        feature_dict["skill_score"] = skill_score
        feature_dict["exp_score"] = exp_csv * EXP_WEIGHT
        feature_dict["qual_score"] = qual_csv * QUAL_WEIGHT

        feature_dict["skill_missing"] = skill_missing
        feature_dict["exp_missing"] = exp_missing
        feature_dict["qual_missing"] = qual_missing

    # -----------------------------
    # Convert to tensor
    # -----------------------------
    features = torch.tensor(list(feature_dict.values()), dtype=torch.float)

    # -----------------------------
    # NORMALIZATION
    # -----------------------------
    if features.numel() > 0:
        features = (features - features.mean()) / (features.std() + 1e-6)

    return features


# -----------------------------
# Dataset
# -----------------------------
class ResumeJDDataset(Dataset):
    def __init__(self, pairs, tokenizer, skill_dict, skill_graph):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.skill_dict = skill_dict
        self.skill_graph = skill_graph

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]

        encoding = self.tokenizer(
            item['resume'],
            item['jd'],
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        extra_features = compute_features(
            item,
            self.skill_dict,
            self.skill_graph
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'extra_features': extra_features,
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }