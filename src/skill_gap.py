def skill_gap_analysis(resume_skills, jd_skills, skill_graph=None):
    matched = []
    missing = []
    related = []

    for skill in jd_skills:
        if skill in resume_skills:
            matched.append(skill)
        elif skill_graph and skill in skill_graph:
            if any(s in resume_skills for s in skill_graph[skill]):
                related.append(skill)
            else:
                missing.append(skill)
        else:
            missing.append(skill)

    return matched, related, missing