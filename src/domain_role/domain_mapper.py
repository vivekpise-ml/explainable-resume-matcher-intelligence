import json

def normalize_domain(domain_str):

    #print("Domain String", domain_str)
    #print("🔥 normalize_domain ACTIVE")
    if domain_str is None or not isinstance(domain_str, str):
        return "General"

    domain_str = domain_str.lower()

    tokens = [d.strip() for d in domain_str.split(",")]

    for token in tokens:

        # Manufacturing cluster
        if any(x in token for x in ["manufacturing", "industrial", "machine", "plant"]):
            return "Manufacturing"

        # Automobile cluster
        if any(x in token for x in ["automobile", "vehicle", "automotive"]):
            return "Automobile"

        # Insurance
        if "insurance" in token:
            return "Insurance"

        # IT Services
        if any(x in token for x in ["sap", "software", "it", "cloud", "integration"]):
            return "IT Services"

    return "General"