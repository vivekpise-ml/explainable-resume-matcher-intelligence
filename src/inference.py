import torch
import json

from transformers import BertTokenizer

from src.matcher_training import compute_features
from src.skill_extraction import extract_skills
# Commenting this importing from training, because importing causes the complete loading of the 
# all the resumes from training.py
from src.all_models import HybridModel  # IMPORTANT

from src.domain_role.domain_detector import detect_domain
from src.domain_role.role_mapper import detect_role, load_role_config, normalize_role
from src.domain_role.encoders import domain_map, role_map
from src.domain_role.domain_mapper import normalize_domain




# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "models/hybrid_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load skill files
# -----------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


domain_config = load_json("data/domain_role/domain_config.json")
role_config = load_json("data/domain_role/role_config.json")

# -----------------------------
# Load model
# -----------------------------
def load_model(skill_dict, skill_graph):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 🔥 Compute feature dimension dynamically
    dummy_item = {
        "resume": "dummy resume",
        "jd": "dummy jd",
        "skill_score": None,
        "experience_score": None,
        "qualification_score": None
    }

    dummy_features = compute_features(dummy_item, skill_dict, skill_graph)
    extra_dim = dummy_features.shape[0]

    model = HybridModel(extra_dim=extra_dim, num_labels=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    model.to(DEVICE)
    model.eval()

    return model, tokenizer


# -----------------------------
# Predict function
# -----------------------------
def predict(
    resume_text,
    jd_text,
    model,
    tokenizer,
    skill_dict,
    skill_graph
):
    
        
    #domain = detect_domain(jd_text, domain_config)
    raw_domain = detect_domain(jd_text, domain_config)
    domain = normalize_domain(raw_domain)

    #role = detect_role(jd_text, role_config)
    raw_role = detect_role(jd_text, role_config)
    role = normalize_role(raw_role)

    item = {
        "resume": resume_text,
        "jd": jd_text,
        "domain": domain,
        "role": role,
        "skill_score": None,
        "experience_score": None,
        "qualification_score": None
    }

    

    print("🔍 Detected Domain:", domain)
    print("🔍 Detected Role:", role)
    
    # -------- Tokenize --------
    encoding = tokenizer(
        resume_text,
        jd_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    # -------- Features --------
    extra_features = compute_features(item, skill_dict, skill_graph)
    extra_features = extra_features.unsqueeze(0).to(DEVICE)

    # debug 
    print("🔥 Inference feature size:", extra_features.shape)

    # -------- Inference --------
    with torch.no_grad():
        logits = model(input_ids, attention_mask, extra_features)
        probs = torch.softmax(logits, dim=1)

        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    return pred_class, confidence


# -----------------------------
# Explainability
# -----------------------------
def explain(resume_text, jd_text, skill_dict):

    resume_tech, resume_soft = extract_skills(resume_text, skill_dict)
    jd_tech, jd_soft = extract_skills(jd_text, skill_dict)

    # Convert to sets
    resume_skills = set(resume_tech) | set(resume_soft)
    jd_skills = set(jd_tech) | set(jd_soft)

    matched = resume_skills & jd_skills
    missing = jd_skills - matched

    # 🔥 NEW (tech vs soft breakdown)
    matched_tech = set(resume_tech) & set(jd_tech)
    missing_tech = set(jd_tech) - set(resume_tech)

    matched_soft = set(resume_soft) & set(jd_soft)
    missing_soft = set(jd_soft) - set(resume_soft)

    return {
        "matched": list(matched),
        "missing": list(missing),

        # 🔥 NEW fields
        "matched_tech": list(matched_tech),
        "missing_tech": list(missing_tech),
        "matched_soft": list(matched_soft),
        "missing_soft": list(missing_soft),
    }


# -----------------------------
# Class interpretation
# -----------------------------
def interpret_class(c):
    mapping = {
        0: "Poor Match",
        1: "Average Match",
        2: "Good Match",
        3: "Excellent Match"
    }
    return mapping.get(c, "Unknown")


# -----------------------------
# Main runner (for testing)
# -----------------------------
#def run_inference(resume_text, jd_text):
def run_inference(
    resume_text,
    jd_text,
    model,
    tokenizer,
    skill_dict,
    skill_graph
    ):

    print("🔥 NEW VERSION OF run_inference LOADED 🔥")
    #skill_dict = load_json("data/annotations/skill_dict.json")
    #skill_graph = load_json("data/annotations/skill_graph.json")

    #model, tokenizer = load_model(skill_dict, skill_graph)

    pred, confidence = predict(
        resume_text,
        jd_text,
        model,
        tokenizer,
        skill_dict,
        skill_graph
    )

    explanation = explain(resume_text, jd_text, skill_dict)

    return {
        "prediction": interpret_class(pred),
        "confidence": round(confidence, 3),
        "matched_skills": explanation["matched"],
        "missing_skills": explanation["missing"],

        # 🔥 ADD THESE
        "matched_tech": explanation["matched_tech"],
        "missing_tech": explanation["missing_tech"],
        "matched_soft": explanation["matched_soft"],
        "missing_soft": explanation["missing_soft"]
    }