import json

def load_skill_graph(path):
    with open(path, "r") as f:
        return json.load(f)


def check_related(skill, resume_skills, skill_graph):
    if skill in skill_graph:
        related_skills = skill_graph[skill]

        for rs in related_skills:
            if rs in resume_skills:
                return True

    return False