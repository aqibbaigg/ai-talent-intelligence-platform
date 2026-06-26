"""
app/services/skill_extractor.py
--------------------------------
Railway-safe skill extraction pipeline.

Default:
- Keyword-based extraction only
- Does NOT load spaCy automatically

Optional:
- Enable spaCy by setting ENABLE_SPACY_SKILLS=true
"""

import os
import re
from loguru import logger


SKILLS_DB = {
    # Programming languages
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go",
    "Rust", "Kotlin", "Swift", "R", "Scala", "PHP", "Ruby", "Dart",
    "MATLAB", "Julia",

    # Web frameworks
    "React", "Vue", "Angular", "Next.js", "Nuxt.js", "Svelte",
    "FastAPI", "Flask", "Django", "Express", "NestJS", "Spring Boot",
    "Laravel", "Rails", "ASP.NET",

    # Databases
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "Cassandra",
    "DynamoDB", "Elasticsearch", "Neo4j", "InfluxDB", "Snowflake",
    "BigQuery", "Redshift",

    # ML / AI
    "TensorFlow", "PyTorch", "Keras", "scikit-learn", "Scikit-learn",
    "XGBoost", "LightGBM", "NLTK", "spaCy", "Hugging Face",
    "LangChain", "OpenAI", "FAISS", "Sentence Transformers",
    "Transformers", "Computer Vision", "NLP", "Deep Learning",
    "Machine Learning", "Reinforcement Learning", "RAG", "LLM",
    "Fine-tuning", "Generative AI", "Artificial Intelligence",

    # Data
    "Pandas", "NumPy", "Spark", "Hadoop", "Kafka", "Airflow",
    "dbt", "Power BI", "Tableau", "Looker", "Matplotlib",
    "Seaborn", "Excel", "Data Analysis", "Data Science",

    # Backend / APIs
    "REST API", "REST", "GraphQL", "gRPC", "WebSocket",
    "Microservices", "API", "JSON", "YAML", "OpenAPI", "Swagger",
    "SQLAlchemy", "Alembic", "Postman",

    # Cloud
    "AWS", "GCP", "Azure", "EC2", "S3", "Lambda", "EKS", "GKE",
    "Cloud Run", "Firebase", "Supabase", "Vercel", "Railway",

    # DevOps / Infrastructure
    "Docker", "Kubernetes", "Terraform", "Ansible", "CI/CD",
    "GitHub Actions", "Jenkins", "ArgoCD", "Helm", "Nginx",
    "Linux", "Bash", "Git", "GitHub",

    # Frontend / UI
    "HTML", "CSS", "Figma",

    # Testing / PM
    "pytest", "Jest", "Cypress", "Selenium", "Playwright",
    "Unit Testing", "Integration Testing", "TDD",
    "Agile", "Scrum", "Jira", "Confluence", "Notion",
}

_SKILLS_LOWER = {s.lower(): s for s in SKILLS_DB}
_nlp = None


def extract_skills_keyword(text: str) -> list[str]:
    """
    Fast keyword scan using safe boundary matching.
    """
    if not text:
        return []

    found = set()
    text_lower = text.lower()

    for skill_lower, skill_canonical in _SKILLS_LOWER.items():
        pattern = rf"(?<![a-zA-Z0-9]){re.escape(skill_lower)}(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.add(skill_canonical)

    return sorted(found)


def _spacy_enabled() -> bool:
    return os.getenv("ENABLE_SPACY_SKILLS", "false").lower() in {
        "1", "true", "yes", "y"
    }


def _get_nlp():
    """
    Load spaCy only when ENABLE_SPACY_SKILLS=true.
    This prevents Railway memory crashes during resume upload.
    """
    global _nlp

    if not _spacy_enabled():
        return None

    if _nlp is not None:
        return _nlp

    try:
        import spacy

        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded: en_core_web_sm")
            return _nlp
        except OSError:
            logger.warning("en_core_web_sm not found.")

        try:
            _nlp = spacy.load("en_core_web_lg")
            logger.info("spaCy model loaded: en_core_web_lg")
            return _nlp
        except OSError:
            logger.warning("en_core_web_lg not found.")

    except Exception as e:
        logger.warning("spaCy initialization failed: {}", e)

    _nlp = None
    return None


def extract_skills_spacy(text: str) -> list[str]:
    """
    Optional spaCy NER extraction.
    Disabled by default on Railway.
    """
    nlp = _get_nlp()
    if nlp is None:
        return []

    doc = nlp(text[:5000])

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
                and len(text_clean) < 40
                and len(text_clean.split()) <= 3
            ):
                found.add(text_clean)

    return sorted(found)


async def extract_skills_llm(text: str) -> list[str]:
    """
    Future LLM-based skill extractor.
    """
    return []


def extract_all_skills(text: str) -> list[str]:
    """
    Combined extractor.

    Railway default:
    - keyword extraction only

    Optional:
    - spaCy extraction if ENABLE_SPACY_SKILLS=true
    """
    keyword_skills = extract_skills_keyword(text)

    spacy_skills = []
    if _spacy_enabled():
        try:
            spacy_skills = extract_skills_spacy(text)
        except Exception as e:
            logger.warning("spaCy extraction failed: {}", e)

    all_skills = set(keyword_skills)

    for skill in spacy_skills:
        all_skills.add(skill)

    result = sorted(all_skills)

    logger.info(
        "Skills extracted — keyword={} spaCy={} total={}",
        len(keyword_skills),
        len(spacy_skills),
        len(result),
    )

    return result