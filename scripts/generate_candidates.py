#!/usr/bin/env python3
"""Generate synthetic candidate profiles for the job matchmaking engine.

This script creates 500 realistic candidate profiles with diverse roles,
skills, experience levels, and biographical narratives. Profiles are drawn
from 10 distinct professional domains including software engineering,
data science, cybersecurity, and more.

Usage:
    python scripts/generate_candidates.py

Output:
    data/candidates.json — 500 synthetic profiles in JSON format
"""

import json
import random
import os

random.seed(42)

# ---------------------------------------------------------------------------
# Skill pools organised by professional domain
# ---------------------------------------------------------------------------
SKILL_POOLS = {
    "software_engineering": [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "Go", "Rust",
        "Git", "Docker", "Kubernetes", "CI/CD", "Linux", "REST APIs",
        "GraphQL", "Microservices", "System Design", "Agile", "TDD", "OOP",
    ],
    "web_frontend": [
        "React", "Angular", "Vue.js", "Next.js", "HTML5", "CSS3",
        "Tailwind CSS", "Sass", "Webpack", "Redux", "Responsive Design",
        "Figma", "UI/UX Design", "Web Performance",
    ],
    "web_backend": [
        "Node.js", "Express.js", "Django", "Flask", "FastAPI", "Spring Boot",
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
        "RabbitMQ", "Kafka", "OAuth2", "API Design", "Database Design",
    ],
    "data_science": [
        "Python", "R", "SQL", "Pandas", "NumPy", "Scikit-learn",
        "TensorFlow", "PyTorch", "Data Visualization", "Matplotlib",
        "Statistical Analysis", "A/B Testing", "Feature Engineering",
        "Jupyter Notebooks", "Data Wrangling",
    ],
    "machine_learning": [
        "Python", "TensorFlow", "PyTorch", "Scikit-learn", "Hugging Face",
        "NLP", "Computer Vision", "Deep Learning", "MLOps", "MLflow",
        "Transformers", "Recommendation Systems", "Model Deployment",
        "Feature Engineering", "Hyperparameter Tuning",
    ],
    "data_engineering": [
        "Python", "SQL", "Apache Spark", "Apache Airflow", "dbt",
        "Snowflake", "BigQuery", "Databricks", "ETL Pipelines",
        "Data Warehousing", "Kafka", "Terraform", "Data Quality",
    ],
    "devops_cloud": [
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
        "Ansible", "Jenkins", "GitHub Actions", "Prometheus", "Grafana",
        "Linux", "Bash", "Infrastructure as Code", "SRE",
    ],
    "cybersecurity": [
        "Network Security", "Penetration Testing", "SIEM", "Incident Response",
        "Encryption", "IAM", "OWASP", "Python", "Wireshark", "Nmap",
        "Threat Modeling", "Cloud Security",
    ],
    "mobile_development": [
        "React Native", "Flutter", "Swift", "Kotlin", "iOS Development",
        "Android Development", "Firebase", "Mobile UI/UX", "REST APIs",
    ],
    "product_management": [
        "Product Strategy", "Roadmap Planning", "User Research",
        "A/B Testing", "SQL", "Jira", "Agile", "Scrum", "OKRs",
        "Competitive Analysis", "Wireframing", "Figma",
    ],
}

# ---------------------------------------------------------------------------
# Role → domain mapping (each role draws skills from two pools)
# ---------------------------------------------------------------------------
ROLES = {
    "Software Engineer": ["software_engineering", "web_backend"],
    "Senior Software Engineer": ["software_engineering", "web_backend"],
    "Full Stack Developer": ["web_frontend", "web_backend"],
    "Frontend Developer": ["web_frontend", "software_engineering"],
    "Backend Developer": ["web_backend", "software_engineering"],
    "Data Scientist": ["data_science", "machine_learning"],
    "Senior Data Scientist": ["data_science", "machine_learning"],
    "Machine Learning Engineer": ["machine_learning", "data_engineering"],
    "ML Engineer": ["machine_learning", "software_engineering"],
    "Data Engineer": ["data_engineering", "software_engineering"],
    "Data Analyst": ["data_science", "product_management"],
    "DevOps Engineer": ["devops_cloud", "software_engineering"],
    "Cloud Architect": ["devops_cloud", "software_engineering"],
    "Product Manager": ["product_management", "data_science"],
    "Security Engineer": ["cybersecurity", "software_engineering"],
    "Mobile Developer": ["mobile_development", "software_engineering"],
    "iOS Developer": ["mobile_development", "web_frontend"],
    "Android Developer": ["mobile_development", "web_frontend"],
    "AI Engineer": ["machine_learning", "software_engineering"],
    "NLP Engineer": ["machine_learning", "data_science"],
}

# ---------------------------------------------------------------------------
# Name generation pools
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "Alex", "Jordan", "Priya", "Arun", "Wei", "Yuki", "Fatima", "Omar",
    "Sofia", "Liam", "Emma", "Noah", "Lucas", "Mia", "Ethan", "Olivia",
    "Aisha", "Raj", "Chen", "Diego", "Elena", "Viktor", "Kai", "Zara",
    "Marcus", "Felix", "Luna", "Ravi", "Nina", "Tariq", "Hana", "Sage",
]
LAST_NAMES = [
    "Smith", "Johnson", "Patel", "Kumar", "Chen", "Wang", "Kim", "Park",
    "Garcia", "Martinez", "Brown", "Davis", "Lee", "Wilson", "Taylor",
    "Anderson", "Müller", "Johansson", "Fernandez", "O'Brien", "Rossi",
    "Dubois", "Hansen", "Singh", "Zhang", "Tanaka", "Berg", "Costa",
]

# ---------------------------------------------------------------------------
# Bio templates with variable slots
# ---------------------------------------------------------------------------
BIO_TEMPLATES = [
    "Experienced {role} with {years} years of expertise. Proficient in {skills_phrase} and passionate about building impactful solutions.",
    "Results-driven {role} bringing {years} years of hands-on experience. Skilled in {skills_phrase} with a strong focus on delivering quality.",
    "Detail-oriented {role} with {years}+ years in the industry. Core competencies include {skills_phrase}.",
    "{role} with a {years}-year track record of delivering high-quality work. Adept at {skills_phrase} and thrives in collaborative environments.",
    "Motivated {role} seeking new challenges after {years} years of professional growth. Expert in {skills_phrase}.",
    "Innovative {role} passionate about leveraging {skills_phrase} to solve complex problems. {years} years of experience.",
    "Dedicated {role} with {years} years of experience building scalable systems. Specializing in {skills_phrase}.",
]


def generate_candidate(cid: int) -> dict:
    """Generate a single synthetic candidate profile."""
    role = random.choice(list(ROLES.keys()))
    domains = ROLES[role]
    pool = list(set(s for d in domains for s in SKILL_POOLS[d]))
    skills = random.sample(pool, min(random.randint(5, 8), len(pool)))
    exp = random.choices(
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
        weights=[5, 8, 12, 15, 15, 12, 10, 8, 6, 5, 4],
        k=1,
    )[0]
    bio = random.choice(BIO_TEMPLATES).format(
        role=role, years=exp, skills_phrase=", ".join(skills[:3])
    )
    return {
        "id": cid,
        "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        "bio": bio,
        "skills": skills,
        "experience_years": exp,
        "desired_role": role,
    }


def main():
    """Generate 500 candidates and save to data/candidates.json."""
    os.makedirs("data", exist_ok=True)
    candidates = [generate_candidate(i) for i in range(500)]

    out_path = os.path.join("data", "candidates.json")
    with open(out_path, "w") as f:
        json.dump(candidates, f, indent=2)

    print(f"Generated {len(candidates)} candidates → {out_path}")
    sample = candidates[0]
    print(f"\nSample: {sample['name']} — {sample['desired_role']} ({sample['experience_years']}yr)")
    print(f"  Skills: {', '.join(sample['skills'])}")
    print(f"  Bio:    {sample['bio'][:120]}...")


if __name__ == "__main__":
    main()
