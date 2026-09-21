import pandas as pd

def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    text_cols = [
        "title",
        "description",
        "skills_desc"]

    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)
    df["title_length"] = df["title"].str.len()
    df["description_length"] = (df["description"].str.len())
    df["skills_length"] = (df["skills_desc"].str.len())
    df["description_word_count"] = (df["description"].str.split().str.len())
    df["skills_word_count"] = (df["skills_desc"].str.split().str.len())
    df["combined_text"] = (df["title"] + " " + df["description"] + " " + df["skills_desc"])
    return df

if __name__ == "__main__":
    df = pd.DataFrame({
        "title": ["Python Developer", "Data Scientist"],
        "description": ["Develop Python applications and APIs", "Build machine learning models"],
        "skills_desc": [ "Python, FastAPI", "Python, Machine Learning"]
    })

    result = add_text_features(df)
    print("\n=== ORIGINAL DATA ===")
    print(df)
    print("\n=== TEXT FEATURES ===")
    print(result)
    print("\n=== NEW FEATURES ===")
    print(
        result[
            [
                "title_length",
                "description_length",
                "skills_length",
                "description_word_count",
                "skills_word_count",
                "combined_text"
            ]
        ]
    )