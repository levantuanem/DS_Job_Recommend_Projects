from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_vectorizer(
    max_features: int = 10_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95
):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True
    )

    return vectorizer