import torch
from torch.utils.data import Dataset
import re
import math

from src.domain_role.encoders import domain_map, role_map, one_hot, DOMAIN_LIST, ROLE_LIST
from src.domain_role.domain_detector import detect_domain
from src.domain_role.role_mapper import detect_role, normalize_role
from src.domain_role.domain_mapper import normalize_domain


from src.domain_role.domain_detector import load_domain_config
from src.domain_role.role_mapper import load_role_config

from src.skill_extraction import extract_skills

from src.features.features_config import FEATURE_ORDER

domain_config = load_domain_config()
role_config = load_role_config()

# =========================================================
# 🔴 HOOKS (toggle easily for experiments)
# =========================================================
USE_DYNAMIC_SKILLS = True      # skill_dict + skill_graph
USE_CSV_FEATURES = False        # CSV scores
#USE_EXPERIENCE = True
#USE_QUALIFICATION = True

'''
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
'''

# -----------------------------
# Graph matching
# -----------------------------
def is_related(skill, resume_skills, skill_graph):
    related = skill_graph.get(skill, [])
    return any(rs in related for rs in resume_skills)


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
        return fallback, 1.0  # value, missing_flag
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
        #resume_skills = extract_skills(resume, skill_dict)
        #jd_skills = extract_skills(jd, skill_dict)

        resume_tech, resume_soft = extract_skills(resume, skill_dict)
        jd_tech, jd_soft = extract_skills(jd, skill_dict)

        resume_skills = set(resume_tech) | set(resume_soft)
        jd_skills = set(jd_tech) | set(jd_soft)

        required = len(jd_skills)

        exact = resume_skills & jd_skills

        related = set()
        for s in jd_skills:
            if s not in exact and is_related(s, resume_skills, skill_graph):
                related.add(s)

        exact_count = len(exact)
        related_count = len(related)

        matched = exact_count + related_count
        missing = required - matched

    else:
        resume_skills = set()
        jd_skills = set()
        exact_count = related_count = matched = missing = required = 0.0

    # -----------------------------
    # CSV skills (clean handling)
    # -----------------------------
    if USE_CSV_FEATURES:
        raw_skill = item.get('skill_score')
        skill_missing = 1 if raw_skill is None else 0
        skill_score = 0.0 if raw_skill is None else raw_skill / 100  # normalize
    else:
        skill_score = 0.0
        skill_missing = 1

    # -----------------------------
    # CSV features - qualification and experience
    #-------------------------------
    #exp_csv, exp_missing = safe_value(item.get('experience_score'), extract_experience(resume))
    #qual_csv, qual_missing = safe_value(item.get('qualification_score'), 0.0)

    exp_csv = item.get('experience_score')
    exp_missing = 1 if exp_csv is None else 0
    exp_csv = 0.0 if exp_csv is None else exp_csv / 100

    qual_csv = item.get('qualification_score')
    qual_missing = 1 if qual_csv is None else 0
    qual_csv = 0.0 if qual_csv is None else qual_csv / 100

    # -----------------------------
    # Optional weighting (simple static fallback)
    # -----------------------------
    EXP_WEIGHT = 0.05
    QUAL_WEIGHT = 0.05

    # -----------------------------
    # Build feature vector
    # -----------------------------
    feature_dict = {}

    if USE_DYNAMIC_SKILLS:

        # ----------- CORE MATCH FEATURES -----------
        feature_dict["exact_ratio"] = exact_count / (required + 1e-5)
        #feature_dict["exact_count_norm"] = exact_count / 10

        feature_dict["related_ratio"] = related_count / (required + 1e-5)
        #feature_dict["related_count_norm"] = related_count / 10

        feature_dict["coverage_ratio"] = matched / (required + 1e-5)
        #feature_dict["missing_ratio"] = missing / (required + 1e-5)

        # ----------- STRUCTURE FEATURES -----------
        feature_dict["resume_skill_count_norm"] = len(resume_skills) / 20
        feature_dict["jd_skill_count_norm"] = required / 20

        
        ratio = exact_count / (related_count + 1e-5)

        # log compression (key fix)
        #feature_dict["exact_vs_related"] = math.log1p(ratio)/2.0

        feature_dict["exact_vs_related"] = math.log1p(ratio)
        
        feature_dict["match_density"] = matched / (len(resume_skills) + 1e-5)

        '''
        # ----------- STRONG MATCH COMPOSITE (NEW - SAFE)  This didnt work -----------
        strong_match_score = (
            0.5 * feature_dict["exact_ratio"] +
            0.3 * feature_dict["coverage_ratio"] +
            0.2 * feature_dict["match_density"]
        )

        feature_dict["strong_match_score"] = strong_match_score
        '''

        #feature_dict["jd_unmatched_pressure"] = missing / (required + 1e-5)

        feature_dict["balance_score"] = (
            min(exact_count, related_count) / (max(exact_count, related_count) + 1e-5)
        )

        # ----------- DOMAIN ALIGNMENT (NEW) -----------

        raw_domain = item.get("domain", None)

        if isinstance(raw_domain, str) and raw_domain.strip() != "":
            domain = normalize_domain(raw_domain)
        else:
            domain = detect_domain(jd, domain_config)

        domain_skills = set(
            domain_config.get(domain, {}).get("core_skills", [])
        )

        overlap = len(resume_skills & domain_skills)

        precision = overlap / max(1, len(resume_skills))
        recall = overlap / max(1, len(domain_skills))

        domain_score = (
            2 * precision * recall
            / (precision + recall + 1e-6)
        )

        feature_dict["domain_alignment_score"] = domain_score

    else:
        # fallback zeros
        for key in [
            "exact_ratio", 
            "related_ratio", 
            "coverage_ratio", 
            "resume_skill_count_norm", "jd_skill_count_norm",
            "exact_vs_related", "match_density",
            "balance_score", "domain_alignment_score"
        ]:
            feature_dict[key] = 0.0


    # -----------------------------
    # CSV features
    # -----------------------------
    if USE_CSV_FEATURES:

        feature_dict["skill_score"] = skill_score
        feature_dict["exp_score"] = exp_csv * EXP_WEIGHT
        feature_dict["qual_score"] = qual_csv * QUAL_WEIGHT

    '''
    else:
        feature_dict["skill_score"] = 0.0
        feature_dict["exp_score"] = 0.0
        feature_dict["qual_score"] = 0.0
    '''

    if USE_CSV_FEATURES:
        # Missing flags (VERY important) always include
        feature_dict["skill_missing"] = skill_missing
        feature_dict["exp_missing"] = exp_missing
        feature_dict["qual_missing"] = qual_missing

    # -----------------------------
    # Domain + Role Encoding (NEW)
    # -----------------------------

    # Detect domain and role
    #domain = detect_domain(jd, domain_config)
    #role = detect_role(jd, role_config)

    raw_domain = item.get("domain", None)

    if isinstance(raw_domain, str) and raw_domain.strip() != "":
        #print("Comput_features now going inside normalize_domain:", raw_domain)
        domain = normalize_domain(raw_domain)   # ✅ TRAINING
    else:
        domain = detect_domain(jd, domain_config)  # fallback

    '''
    print("RAW DOMAIN:", raw_domain)
    print("USING normalize_domain:", normalize_domain.__code__.co_filename)
    print("NORMALIZED DOMAIN:", domain)    
    '''

    raw_role = item.get("role", None)

    
    # 🔥 DEBUG (keep temporarily)
    #print("DEBUG ROLE:", raw_role)

    #print("RAW ROLE:", raw_role)
    #print("NORMALIZED ROLE:", normalize_role(raw_role))

    if isinstance(raw_role, str) and raw_role.strip() != "":
        role = normalize_role(raw_role)   # ✅ TRAINING
    else:
        role = detect_role(jd, role_config)  # ⚠️ fallback

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
    #base_features = list(feature_dict.values())
    base_features = [feature_dict[f] for f in FEATURE_ORDER]

    # One-hot encoding
    domain_vec = one_hot(domain_map.get(domain, domain_map["General"]), len(DOMAIN_LIST))
    role_vec = one_hot(role_map.get(role, role_map["Unknown"]), len(ROLE_LIST))

    # for debug 
    '''
    print("ROLE USED:", role)
    print("ROLE INDEX:", role_map.get(role, role_map["Unknown"]))
    print("ROLE VECTOR:", role_vec)

    
    print("Base:", len(base_features))
    print("Domain:", len(domain_vec))
    print("Role:", len(role_vec))
    '''

    # -----------------------------
    # Convert to tensor (UPDATED)
    # -----------------------------
    

    features = torch.tensor(
        base_features + domain_vec + role_vec,
        dtype=torch.float
    )

    # -----------------------------
    # 🔥 NORMALIZATION (CRITICAL)
    # -----------------------------
    #if features.numel() > 0:
    #    features = (features - features.mean()) / (features.std() + 1e-6)

    '''
    # -----------------------------
    #  DEBUG PRINT (BEST PART)
    # -----------------------------
    print("\n===== FEATURE DEBUG =====")
    for k, v in feature_dict.items():
        print(f"{k:25s}: {v}")
    print("=========================\n")
    '''

    final_features = base_features + domain_vec + role_vec

    '''
    if not hasattr(compute_features, "printed"):
        
        print("Feature length:", len(final_features))
        
        print("🔒 DOMAIN ORDER:", DOMAIN_LIST)
        print("🔒 ROLE ORDER:", ROLE_LIST)
        
        compute_features.printed = True
    '''

    return torch.tensor(final_features, dtype=torch.float)


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

        
        extra_features = compute_features(item, self.skill_dict, self.skill_graph)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'extra_features': extra_features,
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }