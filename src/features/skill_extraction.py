import re
import pandas as pd

SKILLS = {
    "python": ["python"],
    "sql": ["sql"],
    "java": ["java"],
    "javascript": ["javascript", "java script"],
    "typescript": ["typescript"],
    "r": ["r", "r programming", "r language"],
    "c++": ["c++"],
    "c#": ["c#", "c sharp"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "excel": ["excel", "microsoft excel"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "machine_learning": ["machine learning", "machine-learning"],
    "deep_learning": ["deep learning", "deep-learning"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "scikit_learn": ["scikit-learn", "sklearn", "scikit learn"],
    "leadership": ["leadership", "team leadership"],
    "management": ["management", "project management"]
}

COMPILED_SKILLS = {
    skill: [
        re.compile(rf"(?<!\w){re.escape(pattern.lower())}(?!\w)")
        for pattern in patterns
    ]
    for skill, patterns in SKILLS.items()
}

SKILL_EXPRESSIONS = {
    skill: "|".join(
        rf"(?<!\w){re.escape(pattern.lower())}(?!\w)"
        for pattern in patterns
    )
    for skill, patterns in SKILLS.items()
}

def normalize_text(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return f" {text.strip()} "

def contains_skill(text: str, patterns: list[re.Pattern[str]]) -> int:
    normalized = normalize_text(text)
    for expression in patterns:
        if expression.search(normalized):
            return 1
    return 0

def extract_skill_features(
    df: pd.DataFrame,
    text_columns: list[str] | None = None) -> pd.DataFrame:
    result = df.copy()
    if text_columns is None:
        text_columns = [
            column
            for column in [
                "title",
                "description",
                "skills_desc"]
            if column in result.columns]

    result["skill_text"] = (
        result[text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1))
    for skill, expression in SKILL_EXPRESSIONS.items():
        result[f"has_{skill}"] = result["skill_text"].str.contains(
            expression,
            case=False,
            na=False,
            regex=True,
        ).astype("int8")

    skill_columns = [
        column
        for column in result.columns
        if column.startswith("has_")
    ]

    result["skill_count"] = result[skill_columns].sum(axis=1)
    return result