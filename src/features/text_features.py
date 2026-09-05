import re
import numpy as np
import pandas as pd

TECHNICAL_KEYWORDS = {
    "python",
    "java",
    "sql",
    "machine learning",
    "deep learning",
    "tensorflow",
    "pytorch",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "pandas",
    "numpy",
    "scikit-learn"}

MANAGEMENT_KEYWORDS = {
    "management",
    "manager",
    "project management",
    "stakeholder",
    "strategy",
    "operations"}

LEADERSHIP_KEYWORDS = {
    "leadership",
    "leader",
    "lead",
    "mentoring",
    "mentor",
    "team lead"}

COMPILED_KEYWORDS = {
    keyword: re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)")
    for keyword in (
        TECHNICAL_KEYWORDS | MANAGEMENT_KEYWORDS | LEADERSHIP_KEYWORDS
    )
}


def keyword_count_series(
    text: pd.Series,
    keywords: set[str],
) -> pd.Series:
    counts = pd.DataFrame(
        {
            keyword: text.str.contains(
                COMPILED_KEYWORDS[keyword],
                na=False,
                regex=True,
            )
            for keyword in keywords
        },
        index=text.index,
    )
    return counts.sum(axis=1).astype("int16")

def clean_text(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\- ]", " ", text)
    return text.strip()

def word_count(text: str) -> int:
    return len(text.split())

def sentence_count(text: str) -> int:
    if not text:
        return 0
    sentences = re.split(r"[.!?]+", text)
    return len([sentence for sentence in sentences if sentence.strip()])

def keyword_count(text: str, keywords: set[str]) -> int:
    count = 0
    for keyword in keywords:
        if COMPILED_KEYWORDS[keyword].search(text):
            count += 1
    return count

def create_text_features(
    df: pd.DataFrame,
    text_columns: list[str] | None = None) -> pd.DataFrame:
    result = df.copy()
    if text_columns is None:
        text_columns = [
            column
            for column in ["title", "description", "skills_desc"]
            if column in result.columns]
    result["combined_text"] = (
        result[text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .map(clean_text))
    
    result["text_length"] = result["combined_text"].str.len()

    result["word_count"] = (result["combined_text"].map(word_count))
    
    result["sentence_count"] = (result["combined_text"].map(sentence_count))

    result["keyword_count"] = keyword_count_series(
        result["combined_text"],
        TECHNICAL_KEYWORDS | MANAGEMENT_KEYWORDS | LEADERSHIP_KEYWORDS,
    )
    result["technical_keyword_count"] = keyword_count_series(
        result["combined_text"],
        TECHNICAL_KEYWORDS,
    )
    result["management_keyword_count"] = keyword_count_series(
        result["combined_text"],
        MANAGEMENT_KEYWORDS,
    )
    result["leadership_keyword_count"] = keyword_count_series(
        result["combined_text"],
        LEADERSHIP_KEYWORDS,
    )

    return result