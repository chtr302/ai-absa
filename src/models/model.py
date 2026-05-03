from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# TF-IDF config shared across both classifiers
_TFIDF = dict(
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    max_features=60_000,
    strip_accents="unicode",
)


def build(kind: str = "nb") -> Pipeline:
    """
    kind: "nb"  -> Multinomial Naive Bayes  (Laplace alpha=0.1)
          "lr"  -> Logistic Regression      (balanced class weights)
    """
    vec = TfidfVectorizer(**_TFIDF)
    if kind == "nb":
        clf = MultinomialNB(alpha=0.1)
    else:
        clf = LogisticRegression(
            C=5.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
        )
    return Pipeline([("tfidf", vec), ("clf", clf)])


def fit(pipeline: Pipeline, texts: list[str], labels: list[str]) -> Pipeline:
    return pipeline.fit(texts, labels)


def predict(pipeline: Pipeline, texts: list[str]) -> list[str]:
    return pipeline.predict(texts).tolist()


def predict_proba(pipeline: Pipeline, texts: list[str]) -> list[dict]:
    classes: list[str] = pipeline.classes_.tolist()
    return [
        dict(zip(classes, row))
        for row in pipeline.predict_proba(texts)
    ]
