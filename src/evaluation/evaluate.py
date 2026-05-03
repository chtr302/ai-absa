from datetime import date

from sklearn.metrics import accuracy_score, classification_report

from src.data.schemas import ABSATriplet, ABSADocument, BatchABSA, Sentiment


def metrics(y_true: list[str], y_pred: list[str]) -> tuple[float, str]:
    """Return (accuracy, classification_report_string)."""
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, digits=4)
    return acc, rep


def to_batch(
    texts: list[str],
    preds: list[str],
    model_name: str = "ai-absa-baseline-v1",
) -> BatchABSA:
    """Wrap predictions into BatchABSA Pydantic schema."""
    docs = [
        ABSADocument(
            raw_text=text,
            triplets=[
                ABSATriplet(
                    aspect="__DOC_LEVEL__",
                    opinion=None,
                    sentiment=Sentiment(label),
                )
            ],
            model_name=model_name,
        )
        for text, label in zip(texts, preds)
    ]
    return BatchABSA(
        results=docs,
        total_count=len(docs),
        execution_date=str(date.today()),
    )
