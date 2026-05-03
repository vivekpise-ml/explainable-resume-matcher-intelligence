"""
=====================================================
Main Entry Point - Resume ↔ JD Matching System
=====================================================

This script performs end-to-end inference using:

✔ Hybrid Model (BERT + Structured Features)
✔ Dynamic Skill Extraction
✔ Skill Graph-based Matching
✔ Explainability (Matched & Missing Skills)

-----------------------------------------------------
Pipeline:
-----------------------------------------------------
1. Load skill dictionary & graph
2. Load trained Hybrid Model
3. Read Resume + JD
4. Compute features (same as training)
5. Predict match class (0–3)
6. Provide explainability

-----------------------------------------------------
Output:
-----------------------------------------------------
- Match Category (Poor → Excellent)
- Confidence Score
- Matched Skills
- Missing Skills

=====================================================
"""

# -----------------------------
# Imports
# -----------------------------
from src.inference import run_inference

import os


# -----------------------------
# Config
# -----------------------------
RESUME_PATH = "sample_resume.txt"
JD_PATH = "sample_jd.txt"


# -----------------------------
# Utility: Read text files
# -----------------------------
def read_text_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# Main execution
# -----------------------------
def main():

    print("\n====================================")
    print(" Resume ↔ JD Matching System ")
    print("====================================\n")

    # -----------------------------
    # Step 1: Load input files
    # -----------------------------
    print("[INFO] Loading resume and job description...\n")

    resume_text = read_text_file(RESUME_PATH)
    jd_text = read_text_file(JD_PATH)

    # -----------------------------
    # Step 2: Run inference
    # -----------------------------
    print("[INFO] Running hybrid model inference...\n")

    result = run_inference(resume_text, jd_text)

    # -----------------------------
    # Step 3: Display results
    # -----------------------------
    print("\n========== RESULT ==========\n")

    print(f"Prediction      : {result['prediction']}")
    print(f"Confidence      : {result['confidence']}")

    print("\n--- Skill Analysis ---")
    print(f"Matched Skills  : {result['matched_skills']}")
    print(f"Missing Skills  : {result['missing_skills']}")

    print("\n============================\n")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()