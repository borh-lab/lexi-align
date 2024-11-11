from pydantic import BaseModel, Field
from typing import Optional


class TokenAlignment(BaseModel):
    source_token: str
    target_token: str


class TextAlignment(BaseModel):
    alignment: list[TokenAlignment] = Field(
        description="A list of (source_token, target_token) TokenAlignment objects representing the alignment between tokens in the source and target texts. The provided tokens are space-delimited strings and should not be further split. A token can be aligned to multiple tokens; in such cases, include multiple tuples with the same source_token paired with different target_tokens. Unaligned tokens (typically those with predominantly grammatical function) can be omitted from the alignment list. For disambiguation, if a token appears multiple times, a suffix is appended to it; reuse this suffix to ensure correct alignment."
    )


class AlignmentAttempt(BaseModel):
    """Records details of a single alignment attempt"""

    attempt_number: int
    messages_sent: list[dict]
    raw_response: Optional[TextAlignment]
    validation_passed: bool
    validation_errors: list[str]
    exception: Optional[str] = None


class AlignmentResult(BaseModel):
    """Enhanced result containing full diagnostic information"""

    alignment: Optional[TextAlignment]
    attempts: list[AlignmentAttempt]
