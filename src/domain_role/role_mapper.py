import json

def load_role_config(path="data/domain_role/role_config.json"):
    with open(path, "r") as f:
        return json.load(f)


def detect_role(jd_text, role_config):
    jd_text = jd_text.lower()

    for role, keywords in role_config.items():
        for word in keywords:
            if word in jd_text:
                return role

    return "Unknown"

def normalize_role(role_str):
    #print("🔥 USING NEW normalize_role")
    if role_str is None or not isinstance(role_str, str):
        return "Unknown"

    role_str = role_str.lower()

    # 🔥 SPLIT multiple roles
    tokens = [r.strip() for r in role_str.split(",")]

    for token in tokens:

        # -----------------------------
        # PRIORITY ORDER (IMPORTANT)
        # -----------------------------

        if "abap" in token:
            return "SAP ABAP Developer"

        if "sap" in token and "consultant" in token:
            return "SAP Technical Consultant"

        if "consultant" in token:
            return "SAP Technical Consultant"

        if "developer" in token:
            return "Backend Developer"

        if "analyst" in token:
            return "Backend Developer"

        if "frontend" in token:
            return "Frontend Developer"

        if "ml" in token or "machine learning" in token:
            return "ML Engineer"

        if "data" in token:
            return "Data Scientist"

    return "Unknown"