import pandas as pd
import re
import json
from collections import Counter
import os


# -----------------------------
# Clean text
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    return text


# -----------------------------
# Extract structured skills (CRITICAL FIX)
# -----------------------------
def extract_structured_skills(skill_list):
    skills = []

    # whitelist for valid short skills
    VALID_SHORT_SKILLS = {"c", "r"}

    for text in skill_list:
        text = clean_text(text)

        # split by comma
        parts = re.split(r',|\n', text)

        for p in parts:
            p = p.strip()

            # remove "strong", "experience in", etc.
            p = re.sub(r'(strong|experience in|experience with)', '', p).strip()

            # FIX: allow valid short skills
            if len(p) > 1 or p in VALID_SHORT_SKILLS:
                skills.append(p)

    return skills


# -----------------------------
# Extract phrases (LIMITED n-grams)
# -----------------------------
def extract_phrases(text_list):
    phrases = []

    for text in text_list:
        text = clean_text(text)

        # split into words
        tokens = text.split()

        # ONLY bigrams (avoid trigrams explosion)
        for i in range(len(tokens)-1):
            phrase = tokens[i] + " " + tokens[i+1]

            # filter junk combinations
            if not any(w in phrase for w in ["experience", "strong", "good"]):
                phrases.append(phrase)

    return phrases


# -----------------------------
# Build skill dictionary
# -----------------------------
def build_skill_dict(csv_path):

    df = pd.read_csv(csv_path, header=1)
    print("Columns:", df.columns.tolist())

    skill_dict = {}

    # -----------------------------
    # 1. IT Skill Set (MOST IMPORTANT)
    # -----------------------------
    if "IT Skill set" in df.columns:
        structured_skills = extract_structured_skills(
            df["IT Skill set"].dropna().astype(str).tolist()
        )

        for skill in structured_skills:
            skill_dict[skill] = "TECH_SKILL"

    # -----------------------------
    # 2. Domain (keep FULL phrase)
    # -----------------------------
    if "Domain" in df.columns:
        for val in df["Domain"].dropna():
            skill = clean_text(val).strip()
            if len(skill) > 2:
                skill_dict[skill] = "TECH_SKILL"

    # -----------------------------
    # 3. Soft Skills
    # -----------------------------
    if "Soft skill" in df.columns:
        soft_skills = extract_structured_skills(
            df["Soft skill"].dropna().astype(str).tolist()
        )

        for skill in soft_skills:
            skill_dict[skill] = "SOFT_SKILL"

    # -----------------------------
    # 4. Optional: limited phrase extraction from Remark
    # -----------------------------
    if "Remark" in df.columns:
        phrases = extract_phrases(
            df["Remark"].dropna().astype(str).tolist()
        )

        counter = Counter(phrases)

        for phrase, freq in counter.most_common(50):
            if len(phrase) > 3:
                skill_dict[phrase] = "TECH_SKILL"

    return skill_dict


# -----------------------------
# Save dictionary
# -----------------------------
def save_skill_dict(skill_dict, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(skill_dict, f, indent=4)


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    csv_path = "data/Data.csv"
    output_file = "data/annotations/skill_dict.json"

    skill_dict = build_skill_dict(csv_path)
    save_skill_dict(skill_dict, output_file)

    print(f"Saved {len(skill_dict)} skills")