import pandas as pd

def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "description" not in df.columns and "snippet" in df.columns:
        df["description"] = df["snippet"]

    for col in ("title", "description", "skills_desc"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["title_length"] = df["title"].str.len()
    df["description_length"] = (df["description"].str.len())
    df["skills_length"] = (df["skills_desc"].str.len())
    df["description_word_count"] = (df["description"].str.split().str.len())
    df["skills_word_count"] = (df["skills_desc"].str.split().str.len())
    df["combined_text"] = (df["title"] + " " + df["description"] + " " + df["skills_desc"])
    return df