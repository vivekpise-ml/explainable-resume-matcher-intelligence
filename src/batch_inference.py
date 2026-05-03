import os
import json

from src.inference import run_inference, load_model

# -----------------------------
# Paths
# -----------------------------
JD_PATH = "sample_jd.txt"
RESUME_FOLDER = "batch_resumes"

SKILL_DICT_PATH = "data/annotations/skill_dict.json"
SKILL_GRAPH_PATH = "data/annotations/skill_graph.json"


# -----------------------------
# Load files
# -----------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


skill_dict = load_json(SKILL_DICT_PATH)
skill_graph = load_json(SKILL_GRAPH_PATH)

# Load model + tokenizer
model, tokenizer = load_model(skill_dict, skill_graph)


# Load JD once
with open(JD_PATH, "r", encoding="utf-8") as f:
    jd_text = f.read()


results = []

# -----------------------------
# Run batch inference
# -----------------------------
for file in os.listdir(RESUME_FOLDER):

    path = os.path.join(RESUME_FOLDER, file)

    with open(path, "r", encoding="utf-8") as f:
        resume_text = f.read()

    result = run_inference(
        resume_text,
        jd_text,
        model,
        tokenizer,
        skill_dict,
        skill_graph
    )

    results.append((file, result))


# -----------------------------
# Print results
# -----------------------------
for file, res in results:
    print("\n====================")
    print("Resume:", file)
    print("Prediction:", res["prediction"])
    print("Confidence:", res["confidence"])

    # 🔥 NEW (since inference updated)
    print("Matched Skills:", res["matched_skills"])
    print("Missing Skills:", res["missing_skills"])
    print("Matched Tech:", res["matched_tech"])
    print("Missing Tech:", res["missing_tech"])
    print("Matched Soft:", res["matched_soft"])
    print("Missing Soft:", res["missing_soft"])