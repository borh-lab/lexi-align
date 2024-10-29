from lexi_align.models import TextAlignment
from typing import Set, Tuple


def get_alignment_pairs(alignment: TextAlignment) -> Set[Tuple[str, str]]:
    """Convert TextAlignment into a set of (source, target) token pairs.

    Args:
        alignment: TextAlignment object to convert

    Returns:
        Set of (source_token, target_token) tuples

    Example:
        >>> alignment = TextAlignment(alignment=[
        ...     TokenAlignment(source_token="cat", target_token="chat"),
        ...     TokenAlignment(source_token="the", target_token="le")
        ... ])
        >>> pairs = get_alignment_pairs(alignment)
        >>> sorted(pairs)  # Sort for consistent output
        [('cat', 'chat'), ('the', 'le')]
    """
    return {(a.source_token, a.target_token) for a in alignment.alignment}


def calculate_metrics(predicted: TextAlignment, gold: TextAlignment) -> dict:
    """Calculate Precision, Recall, and F1 score for token alignments.

    Args:
        predicted: The system-generated alignment
        gold: The gold-standard alignment

    Returns:
        dict containing precision, recall, and f1 scores

    Example:
        >>> pred = TextAlignment(alignment=[
        ...     TokenAlignment(source_token="the", target_token="le"),
        ...     TokenAlignment(source_token="cat", target_token="chat"),
        ...     TokenAlignment(source_token="is", target_token="est")
        ... ])
        >>> gold = TextAlignment(alignment=[
        ...     TokenAlignment(source_token="the", target_token="le"),
        ...     TokenAlignment(source_token="cat", target_token="chat"),
        ...     TokenAlignment(source_token="is", target_token="wrong")
        ... ])
        >>> metrics = calculate_metrics(pred, gold)
        >>> f"{metrics['precision']:.2f}"
        '0.67'
        >>> f"{metrics['recall']:.2f}"
        '0.67'
        >>> f"{metrics['f1']:.2f}"
        '0.67'
    """
    pred_pairs = get_alignment_pairs(predicted)
    gold_pairs = get_alignment_pairs(gold)

    # Calculate true positives (correct alignments)
    true_positives = len(pred_pairs.intersection(gold_pairs))

    precision = true_positives / len(pred_pairs) if pred_pairs else 0.0
    recall = true_positives / len(gold_pairs) if gold_pairs else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}
