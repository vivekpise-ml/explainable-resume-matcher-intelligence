import streamlit as st
import torch
import shap
import matplotlib.pyplot as plt

import os
import sys
import PyPDF2
import docx

import logging
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# 🔥 suppress transformers logs
logging.getLogger("transformers").setLevel(logging.ERROR)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract

from src.domain_role.domain_detector import detect_domain, load_domain_config
from src.domain_role.role_mapper import detect_role, load_role_config, normalize_role
from src.domain_role.domain_features import compute_domain_score
from src.skill_extraction import extract_skills
from src.domain_role.encoders import DOMAIN_LIST, ROLE_LIST

from src.domain_role.domain_mapper import normalize_domain

# DEBUG (you can remove later)
# print("PROJECT_ROOT:", PROJECT_ROOT)
# print("sys.path[0]:", sys.path[0])

from src.inference import run_inference, load_model
from src.matcher_training import compute_features
from src.skill_extraction import load_skill_dict
import json


# -----------------------------
# File Extraction
# -----------------------------
'''
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text
'''

def extract_text_from_pdf(file):
    text = ""

    try:
        file.seek(0)
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except:
        pass

    if len(text.strip()) < 300:
        print("⚠️ OCR fallback triggered in Streamlit")

        file.seek(0)
        images = convert_from_bytes(file.read())

        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)

    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    else:
        return file.read().decode("utf-8")


# -----------------------------
# Load resources
# -----------------------------
@st.cache_resource
def load_all():
    skill_dict = json.load(open("data/annotations/skill_dict.json"))
    skill_graph = json.load(open("data/annotations/skill_graph.json"))

    model, tokenizer = load_model(skill_dict, skill_graph)

    return model, tokenizer, skill_dict, skill_graph


# -----------------------------
# SHAP (structured features only)
# -----------------------------
def explain_features(model, features_tensor):

    X = features_tensor.unsqueeze(0).numpy()

    def model_wrapper(x):
        x_tensor = torch.tensor(x, dtype=torch.float)

        input_ids = torch.zeros((x_tensor.shape[0], 10), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, x_tensor)
            return logits.numpy()

    explainer = shap.Explainer(model_wrapper, X)
    shap_values = explainer(X)

    return shap_values


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Resume Matcher", layout="wide")

st.title("📄 Resume ↔ JD Matcher (Explainable AI)")

st.markdown("### 🎯 Match Score Interpretation")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Class 0", "Poor ❌")
col2.metric("Class 1", "Average ⚠️")
col3.metric("Class 2", "Good ✅")
col4.metric("Class 3", "Excellent 🏆")

model, tokenizer, skill_dict, skill_graph = load_all()

domain_config = load_domain_config()
role_config = load_role_config()

# -----------------------------
# Input (Upload-based)
# -----------------------------
st.subheader("📄 Upload Job Description")
jd_file = st.file_uploader("Upload JD (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])

st.subheader("📂 Upload Resume(s)")
resume_files = st.file_uploader(
    "Upload Resume(s)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)


# -----------------------------
# Run Matching
# -----------------------------
if st.button("Run Matching"):

    if not jd_file or not resume_files:
        st.warning("Please upload JD and at least one resume")
    else:

        jd_text = extract_text(jd_file)

        # Compute ONCE
        raw_domain = detect_domain(jd_text, domain_config)
        domain = normalize_domain(raw_domain)
        raw_role = detect_role(jd_text, role_config)
        role = normalize_role(raw_role)
        
        results = []

        # -----------------------------
        # Run inference for each resume
        # -----------------------------
        for file in resume_files:
            resume_text = extract_text(file)

            # -----------------------------
            # Domain + Role detection
            # -----------------------------
            
            resume_tech, resume_soft = extract_skills(resume_text, skill_dict)
            resume_skills = set(resume_tech) | set(resume_soft)

            domain_score = compute_domain_score(resume_skills, domain, domain_config)

            result = run_inference(
                resume_text,
                jd_text,
                model,
                tokenizer,
                skill_dict,
                skill_graph
            )

            results.append({
                "name": file.name,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "matched_skills": result["matched_skills"],
                "missing_skills": result["missing_skills"],

                # 🔥 NEW (if you added in explain)
                "matched_tech": result.get("matched_tech", []),
                "missing_tech": result.get("missing_tech", []),
                "matched_soft": result.get("matched_soft", []),
                "missing_soft": result.get("missing_soft", []),

                "resume_text": resume_text,

                "domain": domain,
                "role": role,
                "domain_score": round(domain_score, 3)
            })
        
    # 🔥 ADD THESE TWO LINES AT THE END (after results loop finishes)
    st.session_state["results"] = results
    st.session_state["jd_text"] = jd_text

# -----------------------------
# 🔥 DISPLAY RESULTS (NEW BLOCK)
# -----------------------------
if "results" in st.session_state:

    results = st.session_state["results"]
    jd_text = st.session_state["jd_text"]

    # -----------------------------
    # Ranking
    # -----------------------------
    results = sorted(results, key=lambda x: x["confidence"], reverse=True)

    st.subheader("🏆 Resume Ranking")

    for i, res in enumerate(results):

        st.markdown(f"### {i+1}. {res['name']}")
        #st.write(f"**Match:** {res['prediction']}")

        class_map_reverse = {
            "Poor Match": 0,
            "Average Match": 1,
            "Good Match": 2,
            "Excellent Match": 3
        }

        cls = class_map_reverse.get(res["prediction"], "?")

        #st.write(f"**Match:** {res['prediction']} (Class {cls})")
        #st.markdown(f"### 🔥 **Match: {result['prediction']} (Class {cls})**")
        if cls == 3:
            color = "green"
        elif cls == 2:
            color = "blue"
        elif cls == 1:
            color = "orange"
        else:
            color = "red"

        st.markdown(
            f"<h3 style='color:{color};'>🔥 Match: {res['prediction']} (Class {cls})</h3>",
            unsafe_allow_html=True
        )

        #st.write(f"**Confidence:** {res['confidence']}")
        st.markdown(
            f"<b>Confidence:</b> <span style='font-size:18px'>{res['confidence']}</span>",
            unsafe_allow_html=True
        )

        st.write(f"**Domain:** {res['domain']}")
        st.write(f"**Role:** {res['role']}")
        st.write(f"**Domain Match Score:** {res['domain_score']}")

        # -----------------------------
        # Skill Analysis
        # -----------------------------
        with st.expander("🧠 Skill Analysis"):

            st.write("**Matched Skills (All):**", res["matched_skills"])
            st.write("**Missing Skills (All):**", res["missing_skills"])

            st.write("### 🛠️ Tech Skills")
            st.write("Matched:", res["matched_tech"])
            st.write("Missing:", res["missing_tech"])

            st.write("### 🤝 Soft Skills")
            st.write("Matched:", res["matched_soft"])
            st.write("Missing:", res["missing_soft"])

        feature_names = [
                "exact_ratio",
                "related_ratio",
                "coverage_ratio",
                "resume_skill_count_norm",
                "jd_skill_count_norm",
                "exact_vs_related",
                "match_density",
                "balance_score"
            ]
        
        feature_names += [f"domain_{d}" for d in DOMAIN_LIST]
        feature_names += [f"role_{r}" for r in ROLE_LIST]

        # -----------------------------
        # SHAP (optional per resume)
        # -----------------------------
        with st.expander("📊 Feature Contribution (SHAP) - Slow"):

            if st.checkbox(f"Show SHAP for {res['name']}", key=res["name"]):

                item = {
                    "resume": res["resume_text"],
                    "jd": jd_text,
                    "domain": res["domain"], 
                    "role": res["role"],
                    "skill_score": None,
                    "experience_score": None,
                    "qualification_score": None
                }

                features = compute_features(item, skill_dict, skill_graph)

                shap_values = explain_features(model, features)

                # -----------------------------
                # SHAP (fixed for multi-class)
                # -----------------------------

                class_map = {
                    "Poor Match": 0,
                    "Average Match": 1,
                    "Good Match": 2,
                    "Excellent Match": 3
                }

                pred_class = class_map.get(res["prediction"], 0)

                # Compute SHAP
                with st.spinner("Computing SHAP explanation..."):
                    shap_values = explain_features(model, features)

                # 🔥 Extract correct single explanation
                single_explanation = shap_values[0][:, pred_class]

                # 🔥 Attach feature names
                single_explanation.feature_names = feature_names

                fig = plt.figure()
                shap.plots.waterfall(single_explanation, show=False)
                st.pyplot(fig)