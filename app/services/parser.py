"""
app/services/parser.py
-----------------------
PDF parsing service.

Strategy (waterfall — tries best method first):
  1. pdfplumber  — best text extraction, handles tables
  2. PyPDF2      — fallback for encrypted/complex PDFs
  3. Raw bytes   — last resort, returns whatever text exists

Why two libraries?
  pdfplumber is more accurate but occasionally fails on
  edge-case PDFs. PyPDF2 handles more formats but produces
  messier output. Together they cover ~99% of real resumes.
"""

import re
import io
from pathlib import Path
from typing import Optional

import pdfplumber
import PyPDF2
from loguru import logger
from datetime import datetime


# ─────────────────────────────────────────────────────────────────
#  Text extraction
# ─────────────────────────────────────────────────────────────────

def extract_text_pdfplumber(file_bytes: bytes) -> str:
    """
    Primary extractor. pdfplumber is best for:
    - Multi-column resumes
    - Tables (skills table, experience table)
    - Headers and footers
    """
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)


def extract_text_pypdf2(file_bytes: bytes) -> str:
    """
    Fallback extractor. Works on PDFs that pdfplumber struggles with.
    """
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


def clean_text(text: str) -> str:
    """
    Normalise extracted text:
    - Collapse multiple blank lines
    - Remove non-printable characters
    - Normalise whitespace
    """
    # Remove non-printable characters (keep newlines and tabs)
    text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(file_bytes: bytes, filename: str = "") -> str:
    """
    Main entry point — tries pdfplumber first, PyPDF2 as fallback.

    Parameters
    ----------
    file_bytes : raw PDF bytes
    filename   : used only for logging

    Returns
    -------
    str — cleaned text content of the PDF

    Raises
    ------
    ValueError if both extractors fail or text is too short
    """
    text = ""

    # Try pdfplumber first
    try:
        text = extract_text_pdfplumber(file_bytes)
        if text and len(text.strip()) > 50:
            logger.debug("PDF parsed with pdfplumber: {} chars", len(text))
            return clean_text(text)
    except Exception as e:
        logger.warning("pdfplumber failed for {}: {}", filename, e)

    # Fallback to PyPDF2
    try:
        text = extract_text_pypdf2(file_bytes)
        if text and len(text.strip()) > 50:
            logger.debug("PDF parsed with PyPDF2 (fallback): {} chars", len(text))
            return clean_text(text)
    except Exception as e:
        logger.warning("PyPDF2 failed for {}: {}", filename, e)

    # Both failed
    if not text or len(text.strip()) < 50:
        raise ValueError(
            f"Could not extract readable text from '{filename}'. "
            "The PDF may be scanned (image-based) or password protected."
        )

    return clean_text(text)


# ─────────────────────────────────────────────────────────────────
#  Field extraction  (regex-based, fast, no ML needed)
# ─────────────────────────────────────────────────────────────────

def extract_email(text: str) -> Optional[str]:
    """Find first email address in text."""
    pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    match = re.search(pattern, text)
    return match.group(0).lower() if match else None


def extract_phone(text: str) -> Optional[str]:
    """Find first phone number — handles many international formats."""
    pattern = r"""
        (?:\+?\d{1,3}[\s\-\.]?)?   # country code
        (?:\(?\d{2,4}\)?[\s\-\.]?) # area code
        \d{3,4}[\s\-\.]?\d{4}      # number
    """
    match = re.search(pattern, text, re.VERBOSE)
    return match.group(0).strip() if match else None


def extract_name(text: str) -> str:
    """
    Heuristic: the candidate's name is usually the first non-empty
    line of the resume, in title case.
    Falls back to 'Unknown' if not confident.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:5]:         # check first 5 lines only
        # Name-like: 2-4 words, each capitalised, no numbers
        words = line.split()
        if (
            2 <= len(words) <= 4
            and all(w[0].isupper() for w in words if w)
            and not any(c.isdigit() for c in line)
            and len(line) < 60
        ):
            return line
    return lines[0][:100] if lines else "Unknown"



import re
from datetime import datetime
from typing import Optional


def extract_experience_years(text: str) -> Optional[float]:
    """
    Extract experience duration from resumes.

    Handles:
    Dec 2025 May 2026
    Dec 2025 - May 2026
    Dec 2025 – Present
    """

    pattern = re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})"
        r".{0,20}?"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Present)"
        r"\s*(\d{4})?",
        re.IGNORECASE,
    )

    matches = pattern.findall(text)

    print("EXPERIENCE MATCHES:", matches)

    if not matches:
        return None

    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    total_months = 0

    for start_month, start_year, end_month, end_year in matches:

        start_date = datetime(
            int(start_year),
            month_map[start_month.lower()],
            1,
        )

        if end_month.lower() == "present":
            end_date = datetime.now()
        else:
            end_date = datetime(
                int(end_year),
                month_map[end_month.lower()],
                1,
            )

        months = (
            (end_date.year - start_date.year) * 12
            + (end_date.month - start_date.month)
        )

        total_months += max(months, 0)

    return round(total_months / 12, 1)


def extract_education(text: str) -> Optional[str]:
    """
    Extract education section from resume.
    """

    education_keywords = [
        "education",
        "academic background",
        "qualification",
        "qualifications",
    ]

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for i, line in enumerate(lines):
        if any(keyword in line.lower() for keyword in education_keywords):

            section = []

            for next_line in lines[i + 1 : i + 6]:

                if len(next_line) < 3:
                    continue

                if next_line.lower() in [
                    "skills",
                    "experience",
                    "projects",
                    "certifications",
                ]:
                    break

                section.append(next_line)

            if section:
                return " | ".join(section)

    degree_patterns = [
        r"b\.?tech",
        r"bachelor",
        r"m\.?tech",
        r"master",
        r"b\.?e",
        r"mca",
        r"bca",
        r"mba",
        r"phd",
        r"doctorate",
    ]

    for line in lines:
        for pattern in degree_patterns:
            if re.search(pattern, line.lower()):
                return line

    return None


def extract_certifications(text: str) -> list[str]:
    """Find common professional certifications."""
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
