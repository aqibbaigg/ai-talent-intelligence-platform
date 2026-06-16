"""
app/services/skill_extractor.py
--------------------------------
3-tier skill extraction pipeline:

  Tier 1 — Keyword matching (fast, zero cost, high precision)
            Checks resume text against a curated skills database.

  Tier 2 — spaCy NER (catches skills not in our keyword list)
            Uses the en_core_web_lg model to find named entities,
            filtered for tech/skill-like terms.

  Tier 3 — LLM extraction (most accurate, used as enrichment)
            GPT-4o prompt added in Week 3. Stub included here.

Why keyword first?
  For a fixed tech skills list, keyword matching is ~100% precise
  and runs in milliseconds. NER and LLM add recall for edge cases.
  Running all three and deduplicating gives best coverage.
"""

import re
from typing import Optional
from loguru import logger


# ─────────────────────────────────────────────────────────────────
#  Curated skills database
# ─────────────────────────────────────────────────────────────────

SKILLS_DB = {
    # ── Programming languages ─────────────────────────────────────
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go",
    "Rust", "Kotlin", "Swift", "R", "Scala", "PHP", "Ruby", "Dart",
    "MATLAB", "Julia",

    # ── Web frameworks ────────────────────────────────────────────
    "React", "Vue", "Angular", "Next.js", "Nuxt.js", "Svelte",
    "FastAPI", "Flask", "Django", "Express", "NestJS", "Spring Boot",
    "Laravel", "Rails", "ASP.NET",

    # ── Databases ─────────────────────────────────────────────────
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "Cassandra",
    "DynamoDB", "Elasticsearch", "Neo4j", "InfluxDB", "Snowflake",
    "BigQuery", "Redshift",

    # ── ML / AI ───────────────────────────────────────────────────
    "TensorFlow", "PyTorch", "Keras", "scikit-learn", "XGBoost",
    "LightGBM", "NLTK", "spaCy", "Hugging Face", "LangChain",
    "OpenAI", "FAISS", "Sentence Transformers", "Transformers",
    "Computer Vision", "NLP", "Deep Learning", "Machine Learning",
    "Reinforcement Learning", "RAG", "LLM", "Fine-tuning",

    # ── Data ──────────────────────────────────────────────────────
    "Pandas", "NumPy", "Spark", "Hadoop", "Kafka", "Airflow",
    "dbt", "Power BI", "Tableau", "Looker", "Matplotlib", "Seaborn",

    # ── Cloud ─────────────────────────────────────────────────────
    "AWS", "GCP", "Azure", "EC2", "S3", "Lambda", "EKS", "GKE",
    "Cloud Run", "Firebase", "Supabase", "Vercel",

    # ── DevOps / Infrastructure ───────────────────────────────────
    "Docker", "Kubernetes", "Terraform", "Ansible", "CI/CD",
    "GitHub Actions", "Jenkins", "ArgoCD", "Helm", "Nginx",
    "Linux", "Bash", "Git",

    # ── APIs / Architecture ───────────────────────────────────────
    "REST", "GraphQL", "gRPC", "WebSocket", "Microservices",
    "Event-Driven", "CQRS", "DDD",

    # ── Testing ───────────────────────────────────────────────────
    "pytest", "Jest", "Cypress", "Selenium", "Playwright",
    "Unit Testing", "Integration Testing", "TDD",

    # ── Project management ────────────────────────────────────────
    "Agile", "Scrum", "Jira", "Confluence", "Notion",

    # ── Other popular skills ──────────────────────────────────────
    "SQL", "NoSQL", "API", "JSON", "YAML", "HTML", "CSS",
    "Figma", "Postman", "Swagger", "OpenAPI",
}

# Normalise once at import time for fast lookup
_SKILLS_LOWER = {s.lower(): s for s in SKILLS_DB}


# ─────────────────────────────────────────────────────────────────
#  Tier 1: Keyword matching
# ─────────────────────────────────────────────────────────────────

def extract_skills_keyword(text: str) -> list[str]:
    """
    Fast O(n) keyword scan.
    Uses word-boundary regex to avoid 'Java' matching inside 'JavaScript'.

    Returns canonical skill names (original casing from SKILLS_DB).
    """
    found = set()
    text_lower = text.lower()

    for skill_lower, skill_canonical in _SKILLS_LOWER.items():
        # Word boundary — won't match 'Python' inside 'CPython3'
        pattern = rf"\b{re.escape(skill_lower)}\b"
        if re.search(pattern, text_lower):
            found.add(skill_canonical)

    return sorted(found)


# ─────────────────────────────────────────────────────────────────
#  Tier 2: spaCy NER
# ─────────────────────────────────────────────────────────────────

_nlp = None     # lazy-loaded — spaCy model is ~750MB, load once

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_lg")
            logger.info("spaCy model loaded: en_core_web_lg")
        except OSError:
            logger.warning(
                "spaCy model not found. Run: python -m spacy download en_core_web_lg"
            )
            _nlp = None
    return _nlp


def extract_skills_spacy(text: str) -> list[str]:
    """
    NER-based extraction. Catches:
    - ORG entities (company/product names often = tech skills)
    - PRODUCT entities (framework names)
    - Custom pattern matching

    Filters out common false positives (countries, people names, etc.)
    """
    nlp = _get_nlp()
    if nlp is None:
        return []

    # Process only first 5000 chars for speed
    doc = nlp(text[:5000])

    # Common false positives to skip
    blacklist = {
        "india", "usa", "uk", "google", "amazon", "microsoft",
        "facebook", "apple", "university", "college", "institute",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    }

    found = set()
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART"):
            text_clean = ent.text.strip()
            if (
                len(text_clean) >= 2
                and text_clean.lower() not in blacklist
                and text_clean[0].isupper()
                and len(text_clean) < 40
            ):
                # Only add if it looks like a tech term (not a full sentence)
                if len(text_clean.split()) <= 3:
                    found.add(text_clean)

    return sorted(found)


# ─────────────────────────────────────────────────────────────────
#  Tier 3: LLM extraction stub (Week 3)
# ─────────────────────────────────────────────────────────────────

async def extract_skills_llm(text: str) -> list[str]:
    """
    LLM-based extraction using GPT-4o.
    Implemented in Week 3 when LangChain is added.

    Prompt:
        Extract all technical skills from the following resume.
        Return a JSON array of strings only.
        Example: ["Python", "FastAPI", "PostgreSQL"]
    """
    # Week 3 implementation goes here
    return []


# ─────────────────────────────────────────────────────────────────
#  Combined extractor
# ─────────────────────────────────────────────────────────────────

def extract_all_skills(text: str) -> list[str]:
    """
    Run Tier 1 + Tier 2 and merge results.
    Tier 3 (LLM) is async and called separately.

    Deduplication: case-insensitive, returns canonical names.
    """
    keyword_skills  = extract_skills_keyword(text)
    spacy_skills    = extract_skills_spacy(text)

    # Merge — prefer keyword canonical names, add novel spaCy finds
    all_skills = set(keyword_skills)
    keyword_lower = {s.lower() for s in keyword_skills}

    for skill in spacy_skills:
        if skill.lower() not in keyword_lower:
            all_skills.add(skill)

    result = sorted(all_skills)
    logger.debug(
        "Skills extracted — keyword: {}, spaCy: {}, total: {}",
        len(keyword_skills), len(spacy_skills), len(result),
    )
    return result
