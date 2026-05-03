import json

# -----------------------------
# Load skill dictionary
# -----------------------------
def load_skill_dict(path):
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# Extract skills - hybrid NER model
# -----------------------------
def extract_skills(text, skill_dict, nlp=None):
    text_lower = text.lower()

    tech_skills = []
    soft_skills = []

    # 1. Dictionary matching (PRIMARY)
    for skill, label in skill_dict.items():
        if skill in text_lower:
            if label == "TECH_SKILL":
                tech_skills.append(skill)
            elif label == "SOFT_SKILL":
                soft_skills.append(skill)

    # 2. NER (OPTIONAL - FILTERED)
    if nlp:
        doc = nlp(text)

        for ent in doc.ents:
            val = ent.text.lower()

            # Only keep useful entity types
            if ent.label_ in ["ORG", "PRODUCT"]:
                
                # Avoid duplicates
                if val not in tech_skills and len(val) > 2:
                    tech_skills.append(val)

    return list(set(tech_skills)), list(set(soft_skills))