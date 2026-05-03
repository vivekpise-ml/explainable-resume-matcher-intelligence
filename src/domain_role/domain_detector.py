import json

def load_domain_config(path="data/domain_role/domain_config.json"):
    with open(path, "r") as f:
        return json.load(f)


'''
def detect_domain(jd_text, domain_config):
    jd_text = jd_text.lower()

    for domain, data in domain_config.items():
        for skill in data["core_skills"]:
            if skill in jd_text:
                return domain

    return "General"
'''

def detect_domain(jd_text, domain_config):
    jd_text = jd_text.lower()

    for domain, data in domain_config.items():
        for company in data["companies"]:
            if company in jd_text:
                return domain

    return "General"