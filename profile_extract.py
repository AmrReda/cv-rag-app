import re

TECH_KEYWORDS = [
    "C#", ".NET", "ASP.NET", "Azure", "AWS", "GCP", "Docker", "Kubernetes",
    "SQL", "PostgreSQL", "MongoDB", "Microservices", "Event-driven",
    "Kafka", "RabbitMQ", "REST", "GraphQL", "React", "Vue", "TypeScript",
    "Leadership", "Team Lead", "Mentor", "Agile", "Scrum",
]


def extract_skills(text: str):
    found = set()
    lower = text.lower()
    for kw in TECH_KEYWORDS:
        if kw.lower() in lower:
            found.add(kw)
    return sorted(found)


def infer_seniority(text: str):
    # naive heuristic
    if re.search(r"\b(Head of|Principal|Staff Engineer|Lead|Engineering Manager|Manager)\b", text, re.I):
        return "Senior / Lead level"
    if re.search(r"\b(Senior|Sr\.)\b", text, re.I):
        return "Senior"
    return "Mid-level or unspecified"


def build_profile_summary_markdown(text: str) -> str:
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
