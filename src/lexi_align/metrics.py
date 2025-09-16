from typing import TypedDict

from lexi_align.models import TextAlignment


class Metrics(TypedDict):
    precision: float
    recall: float
    f_measure: float
    aer: float
    true_positives: int
    predicted: int
    gold: int


def calculate_metrics(
    predicted: TextAlignment, gold: TextAlignment, f_alpha: float = 0.5
) -> Metrics:
    """Calculate alignment metrics following standard word alignment evaluation.

    Args:
        predicted: The system-generated alignment
        gold: The gold-standard alignment (treated as both sure and possible alignments)
        f_alpha: Weight parameter for F-measure calculation (default: 0.5 for F1)

    Returns:
        dict containing precision, recall, f_measure, aer and alignment statistics

    Example:
        >>> from lexi_align.models import TextAlignment, TokenAlignment
        >>> pred = TextAlignment(alignment=[
        ...     TokenAlignment(source="the", target="le"),
        ...     TokenAlignment(source="cat", target="chat")
        ... ])
        >>> gold = TextAlignment(alignment=[
        ...     TokenAlignment(source="the", target="le"),
        ...     TokenAlignment(source="cat", target="chat"),
        ...     TokenAlignment(source="!", target="!")
        ... ])
        >>> metrics = calculate_metrics(pred, gold)
        >>> print(f"Precision: {metrics['precision']:.2f}")
        Precision: 1.00
        >>> print(f"Recall: {metrics['recall']:.2f}")
        Recall: 0.67
        >>> print(f"AER: {metrics['aer']:.2f}")
        AER: 0.20
    """
    # Convert alignments to sets of pairs
    A = {(a.source, a.target) for a in predicted.alignment}
    # In our case, S = P since we don't distinguish sure/possible alignments
    S = P = {(a.source, a.target) for a in gold.alignment}

    # Calculate intersection sizes
    a_intersect_p = len(A & P)
    a_intersect_s = len(A & S)

    # Calculate metrics following standard formulas
    precision = a_intersect_p / len(A) if A else 0.0
    recall = a_intersect_s / len(S) if S else 0.0

    # Calculate AER (Alignment Error Rate)
    aer = (
        1.0 - ((a_intersect_p + a_intersect_s) / (len(A) + len(S)))
        if (len(A) + len(S)) > 0
        else 1.0
    )

    # Calculate weighted F-measure
    if f_alpha < 0.0:
        f_measure = 0.0
    else:
        if precision > 0 and recall > 0:
            f_divident = (f_alpha / precision) + ((1.0 - f_alpha) / recall)
            f_measure = 1.0 / f_divident
        else:
            f_measure = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
        "aer": aer,
        "true_positives": a_intersect_s,  # Same as a_intersect_p in our case
        "predicted": len(A),
        "gold": len(S),
    }
