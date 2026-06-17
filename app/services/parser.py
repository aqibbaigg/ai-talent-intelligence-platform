"""
app/services/parser.py
-----------------------
PDF parsing service.

Strategy (waterfall — tries best method first):
  1. pdfplumber  — best text extraction, handles tables
  2. PyPDF2      — fallback for encrypted/complex PDFs
  3. Raw bytes   — last resort, returns whatever text exists
"""

import re
import io
from typing import Optional
from datetime import datetime

import pdfplumber
import PyPDF2
from loguru import logger


# ─────────────────────────────────────────────────────────────────
#  Text extraction
# ─────────────────────────────────────────────────────────────────

def extract_text_pdfplumber(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)


def extract_text_pypdf2(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


def clean_text(text: str) -> str:
    text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(file_bytes: bytes, filename: str = "") -> str:
    text = ""
    try:
        text = extract_text_pdfplumber(file_bytes)
        if text and len(text.strip()) > 50:
            logger.debug("PDF parsed with pdfplumber: {} chars", len(text))
            return clean_text(text)
    except Exception as e:
        logger.warning("pdfplumber failed for {}: {}", filename, e)

    try:
        text = extract_text_pypdf2(file_bytes)
        if text and len(text.strip()) > 50:
            logger.debug("PDF parsed with PyPDF2 (fallback): {} chars", len(text))
            return clean_text(text)
    except Exception as e:
        logger.warning("PyPDF2 failed for {}: {}", filename, e)

    if not text or len(text.strip()) < 50:
        raise ValueError(
            f"Could not extract readable text from '{filename}'. "
            "The PDF may be scanned (image-based) or password protected."
        )
    return clean_text(text)


# ─────────────────────────────────────────────────────────────────
#  Field extraction
# ─────────────────────────────────────────────────────────────────

def extract_email(text: str) -> Optional[str]:
    pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    match = re.search(pattern, text)
    return match.group(0).lower() if match else None


def extract_phone(text: str) -> Optional[str]:
    pattern = r"""
        (?:\+?\d{1,3}[\s\-\.]?)?
        (?:\(?\d{2,4}\)?[\s\-\.]?)
        \d{3,4}[\s\-\.]?\d{4}
    """
    match = re.search(pattern, text, re.VERBOSE)
    return match.group(0).strip() if match else None


def extract_name(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:5]:
        words = line.split()
        if (
            2 <= len(words) <= 4
            and all(w[0].isupper() for w in words if w)
            and not any(c.isdigit() for c in line)
            and len(line) < 60
        ):
            return line
    return lines[0][:100] if lines else "Unknown"


# ─────────────────────────────────────────────────────────────────
#  Experience extraction  (fixed)
# ─────────────────────────────────────────────────────────────────

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

MONTH_NAMES = "|".join(MONTH_MAP.keys())   # jan|feb|mar|...


def extract_experience_years(text: str) -> Optional[float]:
    """
    Extract total experience duration from resume text.

    Handles all common date range formats:
      Month Year – Month Year    (Dec 2023 – May 2026)
      Month Year - Month Year    (Dec 2023 - May 2026)
      Month Year to Month Year   (Dec 2023 to May 2026)
      Month Year – Present       (Dec 2023 – Present)
      Year – Year                (2022 – 2024)
      Year – Present             (2022 – Present)
    """
    total_months = 0
    now = datetime.now()

    # ── Pattern 1: "Month Year … Month/Present Year?" ─────────────
    month_range_pattern = re.compile(
        rf"({MONTH_NAMES})\w*[\s.]+(\d{{4}})"          # start: Month YYYY
        r"[\s\-–—to/]+?"                                # separator
        rf"({MONTH_NAMES}|present)\w*[\s.]*(\d{{4}})?", # end: Month/Present YYYY?
        re.IGNORECASE,
    )

    for m in month_range_pattern.finditer(text):
        start_mon, start_yr, end_mon, end_yr = m.groups()
        try:
            start = datetime(int(start_yr), MONTH_MAP[start_mon[:3].lower()], 1)
            if end_mon.lower().startswith("present"):
                end = now
            else:
                end = datetime(int(end_yr), MONTH_MAP[end_mon[:3].lower()], 1)
            months = (end.year - start.year) * 12 + (end.month - start.month)
            total_months += max(months, 0)
        except Exception:
            continue

    if total_months > 0:
        logger.debug("Experience from month ranges: {} months", total_months)
        return round(total_months / 12, 1)

    # ── Pattern 2: "Year – Year" or "Year – Present" ──────────────
    year_range_pattern = re.compile(
        r"\b(\d{4})\s*[\-–—to]+\s*(present|\d{4})\b",
        re.IGNORECASE,
    )

    # Only look in experience/work sections to avoid education years
    experience_section = _extract_section(text, [
        "experience", "work history", "employment", "internship", "projects"
    ])
    search_text = experience_section if experience_section else text

    for m in year_range_pattern.finditer(search_text):
        start_yr, end_val = m.groups()
        try:
            start = datetime(int(start_yr), 1, 1)
            end   = now if end_val.lower() == "present" else datetime(int(end_val), 12, 31)
            # Skip ranges that look like education years (e.g. 2020–2024 in education section)
            if end.year - int(start_yr) > 6:
                continue
            months = (end.year - start.year) * 12 + (end.month - start.month)
            total_months += max(months, 0)
        except Exception:
            continue

    if total_months > 0:
        logger.debug("Experience from year ranges: {} months", total_months)
        return round(total_months / 12, 1)

    # ── Pattern 3: explicit "X years of experience" statement ─────
    explicit = re.search(
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience",
        text, re.IGNORECASE,
    )
    if explicit:
        return float(explicit.group(1))

    return None


def _extract_section(text: str, keywords: list[str]) -> Optional[str]:
    """Return lines between a section header and the next section header."""
    lines = text.splitlines()
    section_headers = [
        "education", "skills", "certifications", "projects",
        "experience", "work", "employment", "summary", "objective",
        "achievements", "publications", "languages",
    ]
    inside = False
    collected = []

    for line in lines:
        lower = line.lower().strip()
        if not inside:
            if any(kw in lower for kw in keywords) and len(lower) < 40:
                inside = True
        else:
            # Stop at next section header
            if any(h in lower for h in section_headers if h not in keywords) and len(lower) < 40:
                break
            collected.append(line)

    return "\n".join(collected) if collected else None


# ─────────────────────────────────────────────────────────────────
#  Education extraction  (fixed)
# ─────────────────────────────────────────────────────────────────

def extract_education(text: str) -> Optional[str]:
    """
    Extract education from resume.
    Prioritises degree lines, not surrounding prose.
    """
    degree_patterns = [
        r"b\.?\s*tech",
        r"bachelor",
        r"m\.?\s*tech",
        r"master",
        r"b\.?\s*e\.?",
        r"b\.?\s*sc",
        r"mca", r"bca", r"mba",
        r"phd", r"ph\.d", r"doctorate",
        r"engineering\s+computer",
        r"computer\s+science",
        r"information\s+technology",
    ]

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # First: find education section and collect institution + degree lines
    education_section = _extract_section(text, ["education", "academic", "qualification"])
    if education_section:
        section_lines = [l.strip() for l in education_section.splitlines() if l.strip()]
        # Pick lines that contain a degree keyword
        degree_lines = [
            l for l in section_lines
            if any(re.search(p, l, re.IGNORECASE) for p in degree_patterns)
        ]
        if degree_lines:
            return " | ".join(degree_lines[:3])
        # Fall back to first few lines of the section
        if section_lines:
            return " | ".join(section_lines[:3])

    # Fallback: scan all lines for degree patterns
    for line in lines:
        if any(re.search(p, line, re.IGNORECASE) for p in degree_patterns):
            return line

    return None


# ─────────────────────────────────────────────────────────────────
#  Certifications extraction
# ─────────────────────────────────────────────────────────────────

def extract_certifications(text: str) -> list[str]:
    cert_patterns = [
        r"AWS\s+(?:Certified\s+)?[\w\s]+(?:Associate|Professional|Specialty)",
        r"Google\s+(?:Cloud\s+)?(?:Certified|Professional)\s+[\w\s]+",
        r"Microsoft\s+(?:Certified\s+)?[\w\s]+",
        r"PMP(?:\s+Certified)?",
        r"CFA(?:\s+Level\s+[123])?",
        r"Certified\s+(?:Data\s+)?Scientist",
        r"TensorFlow\s+(?:Developer\s+)?Certificate",
        r"Kubernetes\s+(?:CKA|CKAD|CKS)",
        r"CISSP|CEH|OSCP",
        r"Scrum\s+Master|CSM",
    ]
    found = []
    for pattern in cert_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found.extend(matches)
    return list(set(m.strip() for m in found))