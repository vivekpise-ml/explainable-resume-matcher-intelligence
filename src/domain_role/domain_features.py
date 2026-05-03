def compute_domain_score(resume_skills, domain, domain_config):
    domain_skills = set(domain_config.get(domain, {}).get("core_skills", []))

    # debug
    print("RESUME SKILLS:", resume_skills)
    print("DOMAIN:", domain)
    print("DOMAIN CORE SKILLS:", domain_skills)
    print("OVERLAP:", resume_skills & domain_skills)

    if not domain_skills:
        return 0.0

    overlap = len(resume_skills & domain_skills)
  

    # 🔥 softer normalization
    return overlap / max(1, len(resume_skills))