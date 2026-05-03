# 🔒 ORDER MATTERS — DO NOT CHANGE AFTER TRAINING

DOMAIN_LIST = [
    "Manufacturing",
    "Automobile",
    "Insurance",
    "IT Services",
    "General"
]

ROLE_LIST = [
    "Data Scientist",
    "ML Engineer",
    "Frontend Developer",
    "Backend Developer",
    "SAP ABAP Developer",
    "SAP Technical Consultant",
    "Project Manager",
    "Unknown"
]

# Create mappings (single source of truth)
domain_map = {d: i for i, d in enumerate(DOMAIN_LIST)}
role_map = {r: i for i, r in enumerate(ROLE_LIST)}

def one_hot(index, size):
    vec = [0] * size
    vec[index] = 1
    return vec