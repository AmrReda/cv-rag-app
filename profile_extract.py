import re

TECH_KEYWORDS = [
    "C#", ".NET", "ASP.NET", "Azure", "AWS", "GCP", "Docker", "Kubernetes",
    "SQL", "PostgreSQL", "MongoDB", "Microservices", "Event-driven",
    "Kafka", "RabbitMQ", "REST", "GraphQL", "React", "Vue", "TypeScript",
    "Leadership", "Team Lead", "Mentor", "Agile", "Scrum",
]


def extract_skills(text: str):
    """
    Extracts technical skills from the text based on
    a predefined list of keywords.

    Args:
        text (str): The text to search for skills.

    Returns:
        list: A sorted list of unique skills found in the text.
    """
    found = set()
    lower = text.lower()
    for kw in TECH_KEYWORDS:
        if kw.lower() in lower:
            found.add(kw)
    return sorted(found)


def infer_seniority(text: str):
    """
    Infers the seniority level of the candidate based on keywords in the text.
    Uses a naive heuristic approach.

    Args:
        text (str): The text to analyze.

    Returns:
        str: The inferred seniority level.
    """
    # naive heuristic
    if re.search(r"\b(Head of|Principal|Staff Engineer|Lead|Engineering Manager|Manager)\b", text, re.I):
        return "Senior / Lead level"
    if re.search(r"\b(Senior|Sr\.)\b", text, re.I):
        return "Senior"
    return "Mid-level or unspecified"


def build_profile_summary_markdown(text: str) -> str:
    """
    Builds a markdown summary of the candidate's profile.
    Includes inferred seniority and detected skills.

    Args:
        text (str): The raw text of the CV.

    Returns:
        str: A markdown formatted string summarizing the profile.
    """
    skills = extract_skills(text)
    seniority = infer_seniority(text)

    md = "### Candidate Summary\n"
    md += f"**Seniority guess:** {seniority}\n\n"
    md += "**Detected skills / tech keywords:**\n"
    if skills:
        md += "- " + "\n- ".join(skills)
    else:
        md += "_No obvious keywords detected_"
    return md
