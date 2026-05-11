
#  Explainable Resume Matcher Intelligence

An advanced NLP-based system for matching resumes with job descriptions using **hybrid semantic + structured intelligence**, enhanced with **domain awareness, role alignment, and explainability**.

---

##  Overview

This project goes beyond traditional resume matching by combining:

*  Transformer-based semantic understanding (BERT)
*  Skill-based feature engineering
*  Domain detection & scoring
*  Role normalization & alignment
*  Explainability through structured features

The system mimics real recruiter reasoning:

> Skills dominate, while domain and role provide contextual signals.

---

##  Key Features

### 1. Hybrid Matching Model

* Combines **BERT embeddings + engineered features**
* Learns both semantic similarity and structured alignment

### 2. Skill Intelligence

* Dictionary-based skill extraction (`skill_dict.json`)
* Skill relationship expansion (`skill_graph.json`)
* Tech vs Soft skill separation
* Skill gap analysis (matched vs missing)

### 3. Domain Intelligence

* Domain detection from JD (company-based)
* Domain normalization (e.g., Industrial Machine → Manufacturing)
* Domain scoring using:

  * skill overlap
  * domain knowledge (`core_skills`) from (`domain_config.json`)
  * (optionally) skill graph expansion

### 4. Role Intelligence

* Role detection from JD and matching with `role_config.json`
* Role normalization to fixed categories
* One-hot encoding for model input

### 5. Explainability

* Shows:

  * matched skills
  * missing skills
  * domain alignment
  * role alignment
* SHAP-based feature contribution (optional)

### 6. Streamlit UI

* Upload JD + multiple resumes
* Ranked output
* Visual explanation of matching

---

##  Core Insight (Key Learning)

> **Feature engineering and data consistency impact performance more than hyperparameter tuning.**

* Skills → primary signal
* Domain & Role → contextual signals
* Model → weighted decision layer

---

##  Project Structure

```
project/
│
├── app/
│   ├── streamlit_app.py
│   ├── streamlit_app_domain.py
│
├── data/
│   ├── annotations/
│   │   ├── skill_dict.json
│   │   ├── skill_graph.json
│   │
│   ├── domain_role/
│   │   ├── domain_config.json
│   │   ├── role_config.json
│   │
│   ├── raw/              # JD + Resume data
│   ├── sap_ta_abap_1_Data_new.csv          # labeled dataset
│
├── models/
│   ├── hybrid_model.pt
│
├── src/
│   ├── domain_role/
│   │   ├── domain_detector.py
│   │   ├── role_detector.py
│   │   ├── domain_mapper.py
│   │
│   ├── skill_extraction.py
│   ├── feature_extraction.py
│   ├── matcher_training.py
│   ├── training.py
│   ├── inference.py
│   ├── batch_inference.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

##  Setup

```bash
pip install -r requirements.txt

```
* For OCR reading installing following if environment is from conda in VS code

```bash
conda install -c conda-forge tesseract poppler
```
* otherwise from colab

```bash
!apt-get install -y poppler-utils tesseract-ocr
!pip install pytesseract pdf2image
```


---

## ▶ Running the Project

### 🔹 Train Model

```bash
python -m src.training
```

---

### 🔹 Run Inference (CLI)

```bash
python -m src.batch_inference
```

---

###  Run Streamlit UI

```bash
cd app
streamlit run streamlit_app_domain.py
```

---

##  Model Architecture

```
Resume + JD
      ↓
BERT Encoder (semantic similarity)
      ↓
Feature Engineering:
    - skill match ratios
    - domain encoding
    - role encoding
      ↓
Concatenation
      ↓
Classifier (shallow MLP)
      ↓
Prediction (Class 0–3)
```

---

## 🧪 Labels

| Class | Meaning         |
| ----- | --------------- |
| 0     | Poor Match      |
| 1     | Average Match   |
| 2     | Good Match      |
| 3     | Excellent Match |

---

##  Domain Scoring (Advanced)

Domain score is computed using:

* domain-specific skills (`core_skills`)
* overlap with resume skills
* optional semantic expansion via skill graph

---

##  Key Challenges Solved

* Noisy domain & role labels in dataset
* Inconsistent feature pipelines (training vs inference)
* Skill vs domain mismatch
* Overfitting with small dataset
* Explainability integration

---

##  Key Learnings

* Feature design > model complexity
* Data consistency across pipeline is critical
* Domain ≠ skills (needs separate modeling)
* Hybrid systems outperform pure NLP models

---

##  Future Improvements

* Better domain scoring using embeddings
* Improved soft skill extraction
* Ranking-based loss instead of classification
* Real-time recruiter feedback loop

---

---

##  Notes

* Raw resume/JD files are ignored via `.gitignore`
* Model trained on structured + semantic hybrid features
* Designed for explainable AI use-case

---

## git cloning

For git cloning.

* Basic Git workflow

* git clone https://github.com/vivekpise-ml/explainable-resume-matcher-intelligence.git
* git checkout -b feature-branch
* git add .
* git commit -m "feature added"
* git push origin feature-branch



