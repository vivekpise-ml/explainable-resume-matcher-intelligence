import pandas as pd
import os
import math

from docx import Document
import pdfplumber


def read_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])


def read_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def read_file(path):
    if path.endswith(".pdf"):
        return read_pdf(path)
    elif path.endswith(".docx"):
        return read_docx(path)
    else:
        return ""


# -----------------------------
# Safe getter for missing values
# -----------------------------
def safe_float(val):
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        return float(val)
    except:
        return None


# -----------------------------
# Load labels + structured data
# -----------------------------
def load_labels(csv_path):
    df = pd.read_csv(csv_path, header = 2)
    # For debugging which can be removed later
    df.columns = df.columns.str.strip()
    print("\nColumns:\n", df.columns)
    print("\nSample rows:\n", df.head(2))

    label_map = {}

    for _, row in df.iterrows():

        resume_id = str(row['Resume Files']).strip()
        jd_id = str(row['JD title ( Folder Name)']).strip()

        score = row['Matching Score']

        if pd.isna(score):
            continue  # skip bad rows

        if score >= 75:
            label = 3
        elif score >= 50:
            label = 2
        elif score >= 25:
            label = 1
        else:
            label = 0

        #print("Sample labels:", list(label_map.values())[:5]) # For debugging

        label_map[(resume_id, jd_id)] = {
            "label": int(label) if not pd.isna(label) else 0,

            "skill_score": safe_float(row.get('Main Skill set Matching Score')),
            "experience_score": safe_float(row.get('Min Years of Experience Matching Score')),
            "qualification_score": safe_float(row.get('Qualification Matching Score')),

            # 🔥 ADD THIS
            "role": row.get("Role"),

            "domain": row["Domain"]
        }

    # Debug
    print("LABEL MAP DOMAIN SAMPLE:", list(label_map.values())[:3])
    return label_map


# -----------------------------
# Create pairs
# -----------------------------
import os

def create_pairs(data_dir, label_map):
    pairs = []

    for (resume_id, jd_id), info in label_map.items():

        # -----------------------------
        # Locate JD folder
        # -----------------------------
        
        #jd_folder = os.path.join(data_dir, jd_id)

        jd_folder = None

        for folder in os.listdir(data_dir):
            if folder.lower() == jd_id.lower():
                jd_folder = os.path.join(data_dir, folder)
                break

        if jd_folder is None:
            print(f"[WARN] JD folder not found: {jd_id}")
            continue

        if not os.path.exists(jd_folder):
            print(f"[WARN] JD folder not found: {jd_folder}")
            continue

        files = os.listdir(jd_folder)

        # -----------------------------
        # Find JD file
        # -----------------------------
        jd_file = None
        for f in files:
            f_lower = f.lower()
            if f_lower.startswith(jd_id.lower()) and (f_lower.endswith(".pdf") or f_lower.endswith(".docx")):
                jd_file = f
                break

        # -----------------------------
        # Find Resume file
        # -----------------------------
        resume_file = None
        for f in files:
            f_lower = f.lower()

            # match pattern: candidateX_jdname
            if resume_id.lower() in f_lower and jd_id.lower() in f_lower:
                if f_lower.endswith(".pdf") or f_lower.endswith(".docx"):
                    resume_file = f
                    break

        # -----------------------------
        # Validate files found
        # -----------------------------
        if jd_file is None:
            print(f"[WARN] JD file not found in {jd_folder}")
            continue

        if resume_file is None:
            print(f"[WARN] Resume file not found for {resume_id} in {jd_folder}")
            continue

        # -----------------------------
        # Build full paths
        # -----------------------------
        jd_path = os.path.join(jd_folder, jd_file)
        resume_path = os.path.join(jd_folder, resume_file)

        # -----------------------------
        # Read text (you must have read_file defined)
        # -----------------------------
        '''
        try:
            jd_text = read_file(jd_path)
            resume_text = read_file(resume_path)
        except Exception as e:
            print(f"[ERROR] Reading file failed: {e}")
            continue
        '''
        # -----------------------------
        # Read text (you must have read_file defined)
        # -----------------------------
        try:
            #print(f"[DEBUG] Attempting to read JD file: {jd_path}")
            jd_text = read_file(jd_path)

            #print(f"[DEBUG] Attempting to read Resume file: {resume_path}")
            resume_text = read_file(resume_path)

        except Exception as e:
            print("\n[ERROR] Reading file failed:")
            print(f"        JD Path     : {jd_path}")
            print(f"        Resume Path : {resume_path}")
            print(f"        Resume ID   : {resume_id}")
            print(f"        JD ID       : {jd_id}")
            print(f"        Error       : {e}\n")
            continue

        # -----------------------------
        # Append pair
        # -----------------------------
        pairs.append({
            "resume": resume_text,
            "jd": jd_text,
            "label": info["label"],

            "skill_score": info.get("skill_score"),
            "experience_score": info.get("experience_score"),
            "qualification_score": info.get("qualification_score"),

            # 🔥 ADD THIS
            "role": info.get("role"),
            "domain": info.get("domain")
        })

    print(f"[INFO] Total valid pairs: {len(pairs)}")

    return pairs