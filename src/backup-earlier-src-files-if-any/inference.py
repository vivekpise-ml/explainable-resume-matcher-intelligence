import torch
import json
from src.skill_extraction import extract_skills
from src.feature_extraction import extract_structured_features


# -----------------------------
# Load Models
# -----------------------------
def load_ner_model():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except:
        print("spaCy model not found. Running without NER.")
        return None


def load_skill_graph(path):
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# Normalize Skill (IMPORTANT)
# -----------------------------
def normalize_skill(skill):
    skill = skill.lower().strip()

    mapping = {
        "cpp": "c++",
        "c plus plus": "c++",
        "s4": "s4 hana",
        "sap s4": "s4 hana"
    }

    return mapping.get(skill, skill)


# -----------------------------
# Graph-based Matching
# -----------------------------
def graph_based_match(jd_skills, resume_skills, skill_graph):
    matched = set()
    graph_matches = {}

    resume_set = set(resume_skills)

    for jd_skill in jd_skills:

        # Direct match
        if jd_skill in resume_set:
            matched.add(jd_skill)
            continue

        # Graph match
        if jd_skill in skill_graph:
            for rel in skill_graph[jd_skill]:
                if rel in resume_set:
                    matched.add(jd_skill)
                    graph_matches[jd_skill] = rel
                    break

    return matched, graph_matches


# -----------------------------
# Final Scoring Function (VERY IMPORTANT)
# -----------------------------
def compute_final_score(transformer_score, matched, total_jd, graph_matches):

    if total_jd == 0:
        skill_score = 0
    else:
        skill_score = len(matched) / total_jd

    graph_bonus = len(graph_matches) * 0.05

    final_score = (
        0.6 * transformer_score +
        0.4 * skill_score +
        graph_bonus
    )

    return min(final_score, 1.0), skill_score


# -----------------------------
# Main Inference
# -----------------------------
def run_inference(
    resume_text,
    jd_text,
    matcher_model,
    tokenizer,
    skill_dict,
    skill_graph,
    ner_model=None,
    row_data=None
):

    # -----------------------------
    # Safety check
    # -----------------------------
    if not resume_text.strip() or not jd_text.strip():
        return {"error": "Empty resume or JD"}

    # -----------------------------
    # Skill Extraction
    # -----------------------------
    resume_tech, resume_soft = extract_skills(
        resume_text, skill_dict, ner_model
    )

    jd_tech, jd_soft = extract_skills(
        jd_text, skill_dict, ner_model
    )

    # Normalize + Deduplicate
    resume_tech = list(set([normalize_skill(s) for s in resume_tech]))
    jd_tech = list(set([normalize_skill(s) for s in jd_tech]))

    # -----------------------------
    # Structured Features
    # -----------------------------
    structured_features = {}
    if row_data is not None:
        structured_features = extract_structured_features(row_data)

    # -----------------------------
    # Transformer Matching
    # -----------------------------
    inputs = tokenizer(
        resume_text,
        jd_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = matcher_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        transformer_score = probs[0][1].item()

    # -----------------------------
    # Graph-based Matching
    # -----------------------------
    matched_set, graph_matches = graph_based_match(
        jd_tech, resume_tech, skill_graph
    )

    missing_set = set(jd_tech) - matched_set

    # -----------------------------
    # Final Score
    # -----------------------------
    final_score, skill_score = compute_final_score(
        transformer_score,
        matched_set,
        len(jd_tech),
        graph_matches
    )

    # -----------------------------
    # Generate Remarks
    # -----------------------------
    remark = generate_remark(
    matched_set,
    missing_set,
    graph_matches
    )

    # -----------------------------
    # Output
    # -----------------------------
    return {
        "match_score": round(final_score, 3),

        "scores": {
            "transformer": round(transformer_score, 3),
            "skill_match": round(skill_score, 3)
        },

        "resume": {
            "tech_skills": resume_tech,
            "soft_skills": resume_soft
        },

        "jd": {
            "tech_skills": jd_tech,
            "soft_skills": jd_soft
        },

        "skill_analysis": {
            "matched": list(matched_set),
            "missing": list(missing_set),
            "graph_matches": graph_matches,
            "coverage": f"{len(matched_set)}/{len(jd_tech)}"
        },

        "remark" : remark,
        
        "features": structured_features
    }

# -------------------------------------------
#  Function to generate explainable Remarks
# -------------------------------------------
def generate_remark(matched, missing, graph_matches):

    if len(matched) == 0:
        return "Candidate does not match the job requirements."

    remark = []

    if len(matched) > 0:
        remark.append(f"Matches {len(matched)} required skills")

    if len(graph_matches) > 0:
        gm = ", ".join([f"{k} via {v}" for k, v in graph_matches.items()])
        remark.append(f"Indirect matches found ({gm})")

    if len(missing) > 0:
        miss = ", ".join(missing[:5])  # limit output
        remark.append(f"Missing key skills: {miss}")

    return ". ".join(remark)