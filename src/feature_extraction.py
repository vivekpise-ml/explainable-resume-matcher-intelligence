import re

def parse_years(value):
    """
    Convert text like '2 years', '18 months', '1.5', etc. into float (years)
    """
    if value is None:
        return None

    value = str(value).lower()

    try:
        # direct float
        return float(value)
    except:
        pass

    # extract numbers
    match = re.search(r'(\d+(\.\d+)?)', value)
    if not match:
        return None

    num = float(match.group(1))

    # convert months → years
    if "month" in value:
        return num / 12

    return num


def extract_structured_features(row):
    features = {}

    # Experience
    features["min_experience"] = parse_years(
        row.get("min years of experience", None)
    )

    # Average tenure
    features["avg_tenure"] = parse_years(
        row.get("average tenure per company", None)
    )

    # Location
    features["location"] = str(
        row.get("location", "")
    ).strip().lower()

    # Qualification
    features["qualification"] = str(
        row.get("qualification", "")
    ).strip().lower()

    # Soft skills
    features["soft_skills"] = str(
        row.get("soft skill", "")
    ).strip().lower()

    return features