import json
import re
from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    TypedDict,
    Union,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import GenerateJsonSchema
from typing_extensions import Self

if TYPE_CHECKING:
    from lexi_align.adapters.base import LLMAdapter

from lexi_align.constants import UNALIGNED_MARKER as _UNALIGNED_MARKER
from lexi_align.text_processing import (
    MarkerGenerator,
    create_subscript_generator,
    remove_unique_one,
)

logger = getLogger(__name__)


def calculate_max_alignments(source_tokens: List[str], target_tokens: List[str]) -> int:
    """Calculate maximum number of alignments required based on token counts.

    Args:
        source_tokens: List of source language tokens
        target_tokens: List of target language tokens

    Returns:
        Maximum number of alignments required (1.5 * max(source_len, target_len)) + 1

    Example:
        >>> calculate_max_alignments(['the', 'cat'], ['le', 'chat'])
        4
        >>> calculate_max_alignments(['a'], ['un'])
        2
    """
    return max(1, int(max(len(source_tokens), len(target_tokens)) * 1.5) + 1)


class SpecialTokens(str, Enum):
    """Special tokens used in alignments."""

    UNALIGNED = "<unaligned>"
    SOURCE_SPECIFIC = "<source_specific>"
    TARGET_SPECIFIC = "<target_specific>"


UNALIGNED_MARKER = _UNALIGNED_MARKER


def _create_token_enum(name: str, tokens: list[str]) -> Type[Enum]:
    """Helper to generate a Token Enum with UNALIGNED and the given tokens."""
    values = {"UNALIGNED": SpecialTokens.UNALIGNED.value}
    values.update(
        {token.replace(" ", "_").replace("-", "_").upper(): token for token in tokens}
    )
    return Enum(f"{name}Tokens", values)  # type: ignore


def create_source_token_enum(tokens: list[str]) -> Type[Enum]:
    """Create an Enum class for source tokens including special markers.

    Args:
        tokens: List of source language tokens (already uniquified)

    Returns:
        Enum class containing tokens and special markers

    Example:
        >>> SourceTokens = create_source_token_enum(["the₁", "cat", "the₂"])
        >>> list(SourceTokens)  # doctest: +NORMALIZE_WHITESPACE
        [<SourceTokens.UNALIGNED: '<unaligned>'>,
         <SourceTokens.THE₁: 'the₁'>,
         <SourceTokens.CAT: 'cat'>,
         <SourceTokens.THE₂: 'the₂'>]
    """
    return _create_token_enum("Source", tokens)


def create_target_token_enum(tokens: list[str]) -> Type[Enum]:
    """Create an Enum class for target tokens including special markers.

    Args:
        tokens: List of target language tokens (already uniquified)

    Returns:
        Enum class containing tokens and special markers
    """
    return _create_token_enum("Target", tokens)


class ValidationErrorType(str, Enum):
    """Validation error types that automatically serialize to strings."""

    MISSING_SOURCE_ALIGNMENTS = (
        "MISSING_SOURCE_ALIGNMENTS"  # For tokens still needing alignment
    )
    MISSING_TARGET_ALIGNMENTS = (
        "MISSING_TARGET_ALIGNMENTS"  # For tokens still needing alignment
    )
    INVALID_SOURCE_TOKEN = "INVALID_SOURCE_TOKEN"  # For tokens not in mapping
    INVALID_TARGET_TOKEN = "INVALID_TARGET_TOKEN"  # For tokens not in mapping
    DUPLICATE_ALIGNMENT = (
        "DUPLICATE_ALIGNMENT"  # For tokens aligned multiple times when not allowed
    )
    OTHER = "OTHER"  # For unexpected errors


class ValidationErrorDict(TypedDict):
    type: ValidationErrorType
    message: str
    tokens: List[str]


class ChatMessageDict(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


def make_unique(
    tokens: List[str], marker_generator: Optional[MarkerGenerator] = None
) -> List[str]:
    """Add unique markers to disambiguate repeated tokens.

    Args:
        tokens: List of tokens to uniquify
        marker_generator: Optional marker generator (defaults to subscript)

    Returns:
        List of tokens with unique markers added to duplicates

    Raises:
        TypeError: If input is not a list of strings

    Example:
        >>> make_unique(["the", "cat", "the", "mat"])
        ['the₁', 'cat', 'the₂', 'mat']
        >>> from lexi_align.text_processing import create_underscore_generator
        >>> make_unique(["the", "cat", "the", "mat"], create_underscore_generator())
        ['the_1', 'cat', 'the_2', 'mat']
    """
    if not isinstance(tokens, list):
        raise TypeError("Input must be a list")

    if not all(isinstance(t, str) for t in tokens):
        raise TypeError("All tokens must be strings")

    # Use default subscript generator if none provided
    marker_generator = marker_generator or create_subscript_generator()

    # Strip existing markers and count base tokens
    base_tokens = [
        remove_unique_one(token, marker_generator.pattern) for token in tokens
    ]
    base_counts: Dict[str, int] = {}
    base_seen: Dict[str, int] = {}
    unique_tokens = []

    # First pass: count base token occurrences
    for base_token in base_tokens:
        base_counts[base_token] = base_counts.get(base_token, 0) + 1

    # Second pass: add markers
    for i, base_token in enumerate(base_tokens):
        if base_counts[base_token] > 1:
            count = base_seen.get(base_token, 0) + 1
            base_seen[base_token] = count
            unique_tokens.append(f"{base_token}{marker_generator.generate(count)}")
        else:
            unique_tokens.append(base_token)

    return unique_tokens


def create_token_mapping(
    tokens: List[str], marker_generator: Optional[MarkerGenerator] = None
) -> "TokenMapping":
    """Create a TokenMapping object for a list of tokens.

    Args:
        tokens: List of original tokens
        marker_generator: Optional marker generator (defaults to subscript)

    Returns:
        TokenMapping object containing original and uniquified tokens with position maps

    Example:
        >>> tokens = ["the", "cat", "the", "mat", "the"]
        >>> mapping = create_token_mapping(tokens)
        >>> mapping.original
        ['the', 'cat', 'the', 'mat', 'the']
        >>> mapping.uniquified
        ['the₁', 'cat', 'the₂', 'mat', 'the₃']
        >>> mapping.get_position('the₁')  # First 'the'
        0
        >>> mapping.get_position('the₂')  # Second 'the'
        2
        >>> mapping.get_position('the₃')  # Third 'the'
        4
        >>> mapping.get_uniquified('the')  # Gets first uniquified version
        'the₁'
    """
    # Use default subscript generator if none provided
    marker_generator = marker_generator or create_subscript_generator()

    # Create uniquified tokens
    uniquified = make_unique(tokens, marker_generator)

    # Create position mappings that track exact positions of uniquified tokens
    positions: dict[str, list[int]] = {}  # Maps base token to list of positions
    unique_positions: dict[str, int] = {}  # Maps uniquified token to its position

    # First build positions map for base tokens
    for i, token in enumerate(tokens):
        base_token = remove_unique_one(token, marker_generator.pattern)
        if base_token not in positions:
            positions[base_token] = []
        positions[base_token].append(i)

    # Then map uniquified tokens to their positions
    for i, (orig, uniq) in enumerate(zip(tokens, uniquified)):
        unique_positions[uniq] = i

    return TokenMapping(
        original=tokens,
        uniquified=uniquified,
        positions=positions,  # Now contains lists of positions for each base token
        unique_positions=unique_positions,
        marker_pattern=marker_generator.pattern,
    )


@dataclass
class TokenMapping:
    """Tracks relationships between original, uniquified, and normalized tokens."""

    original: List[str]  # Original tokens
    uniquified: List[str]  # Tokens with unique markers
    positions: Dict[str, List[int]]  # Position lists for original tokens
    unique_positions: Dict[str, int]  # Position map for uniquified tokens
    marker_pattern: re.Pattern  # Pattern used for markers

    @property
    def normalized_map(self) -> Dict[str, str]:
        """Map from normalized (no markers) to uniquified tokens."""

        return {
            remove_unique_one(token, self.marker_pattern): token
            for token in self.uniquified
        }

    def get_position(self, token: str, normalized: bool = True) -> int:
        """Get position of a token, optionally normalizing it first."""
        if not normalized:
            return self.unique_positions.get(token, -1)
        # If it's already a uniquified token, look it up directly
        if token in self.unique_positions:
            return self.unique_positions[token]
        # Otherwise normalize and return the first occurrence position (if any)
        base = remove_unique_one(token, self.marker_pattern)
        pos_list = self.positions.get(base)
        return pos_list[0] if pos_list else -1

    def get_uniquified(self, token: str) -> str:
        """Get uniquified version of a normalized token."""

        normalized = remove_unique_one(token, self.marker_pattern)
        # Find first uniquified version of this token
        for uniq in self.uniquified:
            if remove_unique_one(uniq, self.marker_pattern) == normalized:
                return uniq
        return token


class TokenAlignment(BaseModel):
    # We want the resulting JSON to be relatively compact so use 1-token field names
    source: str = Field(description="A token from the source text.")
    target: str = Field(description="A token from the target text.")


class TextAlignmentSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional step-by-step reasoning explaining alignment decisions, considering semantic equivalence, syntactic roles, idiomatic expressions, and structural differences between languages.",
    )
    alignment: List[TokenAlignment] = Field(
        description="A list of (source_token, target_token) TokenAlignment objects representing the alignment between tokens in the source and target texts. The provided tokens are space-delimited strings and should not be further split. A token can be aligned to multiple tokens; in such cases, include multiple tuples with the same source_token paired with different target_tokens. Unaligned tokens (typically those with predominantly grammatical function) can be omitted from the alignment list. For disambiguation, if a token appears multiple times, a suffix is appended to it; reuse this suffix to ensure correct alignment."
    )

    @classmethod
    def from_base_schema(cls, base: "TextAlignmentSchema") -> "TextAlignmentSchema":
        """Convert base schema to this schema type."""
        return cls(alignment=base.alignment)

    def to_base_schema(self) -> "TextAlignmentSchema":
        """Convert to base schema."""
        return TextAlignmentSchema(alignment=self.alignment)


class TextAlignment(TextAlignmentSchema):
    # These will not be serialized:
    source_mapping: Optional[TokenMapping] = Field(default=None, exclude=True)
    target_mapping: Optional[TokenMapping] = Field(default=None, exclude=True)
    source_enum: Optional[Type[Enum]] = Field(default=None, exclude=True)
    target_enum: Optional[Type[Enum]] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_and_sort_alignment(self) -> Self:
        """Ensure alignments are deduplicated and sorted while preserving special tokens."""

        # First deduplicate and filter invalid pairs
        unique_pairs = set()
        special_alignments = []
        regular_alignments = []
        duplicates_count = 0
        both_unaligned_count = 0

        # Get list of special token values for comparison
        special_tokens = {
            SpecialTokens.UNALIGNED.value,
            SpecialTokens.SOURCE_SPECIFIC.value,
            SpecialTokens.TARGET_SPECIFIC.value,
        }

        for align in self.alignment:
            pair = (align.source, align.target)

            # Filter both-unaligned pairs
            if align.source == UNALIGNED_MARKER and align.target == UNALIGNED_MARKER:
                both_unaligned_count += 1
                logger.warning(
                    "Filtering out invalid alignment with both source and target as <unaligned>"
                )
                continue

            # Check for duplicates
            if pair in unique_pairs:
                duplicates_count += 1
                continue

            unique_pairs.add(pair)

            # Check if either token is a special token
            if align.source in special_tokens or align.target in special_tokens:
                special_alignments.append(align)
            else:
                regular_alignments.append(align)

        # Log summary if any pairs were filtered
        if duplicates_count > 0:
            logger.warning(
                f"Filtered out {duplicates_count} duplicate alignment pair(s)"
            )
        if both_unaligned_count > 0:
            logger.warning(
                f"Filtered out {both_unaligned_count} both-unaligned pair(s)"
            )

        # Try to sort regular alignments if we have mappings
        if self.source_mapping and self.target_mapping:
            regular_alignments = self.sort_alignments(
                regular_alignments, self.source_mapping, self.target_mapping
            )

        # Combine special tokens with sorted regular alignments
        # Special tokens go at the end by convention
        self.alignment = regular_alignments + special_alignments

        return self

    @staticmethod
    def sort_alignments(
        alignments: list[TokenAlignment],
        source_mapping: TokenMapping,
        target_mapping: TokenMapping,
    ) -> list[TokenAlignment]:
        """Sort alignments by source position first, then target position."""
        if not alignments:
            return alignments

        # logger.debug("Sorting alignments:")
        for align in alignments:
            s_pos = source_mapping.get_position(align.source)
            t_pos = target_mapping.get_position(align.target)
            # logger.debug(
            #     f"Token: {align.source}->{align.target}, Positions: {s_pos},{t_pos}"
            # )

        # Get positions for each alignment
        alignments_with_pos = []
        for align in alignments:
            s_pos = source_mapping.get_position(align.source)
            t_pos = target_mapping.get_position(align.target)
            alignments_with_pos.append((s_pos, t_pos, align))

        # Sort by source position
        sorted_alignments = sorted(alignments_with_pos, key=lambda x: x[0])

        # logger.debug(
        #     f"Sorted positions: {[(s, t, align) for s, t, align in sorted_alignments]}"
        # )

        # Return alignments in sorted order
        return [align for _, _, align in sorted_alignments]

    @classmethod
    def from_token_alignments(
        cls,
        alignments: list[TokenAlignment],
        source_tokens: list[str],
        target_tokens: list[str],
        marker_generator: Optional[MarkerGenerator] = None,
        adapter: Optional["LLMAdapter"] = None,
    ) -> "TextAlignment":
        """Create a TextAlignment from a list of TokenAlignment objects and token lists."""
        # Create mappings with optional marker generator
        source_mapping = create_token_mapping(source_tokens, marker_generator)
        target_mapping = create_token_mapping(target_tokens, marker_generator)

        # Create schema with context
        schema_data = {"alignment": alignments}
        context = {
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
            "adapter": adapter,
        }

        # Handle dynamic schema if adapter supports it
        if adapter and adapter.supports_length_constraints():
            dynamic_schema = create_dynamic_alignment_schema(
                source_tokens, target_tokens, marker_generator
            )
            schema = dynamic_schema.model_validate(schema_data, context=context)
            # Convert back to base schema
            schema = schema.to_base_schema()
        else:
            schema = TextAlignmentSchema.model_validate(schema_data, context=context)

        # Create Enum classes for tokens using the same marker generator
        source_enum = create_source_token_enum(source_mapping.uniquified)
        target_enum = create_target_token_enum(target_mapping.uniquified)

        return cls(
            alignment=schema.alignment,
            source_mapping=source_mapping,
            target_mapping=target_mapping,
            source_enum=source_enum,
            target_enum=target_enum,
        )

    def sort_by_position(
        self,
        source_mapping: TokenMapping,
        target_mapping: TokenMapping,
    ) -> "TextAlignment":
        """Sort alignments by source position first, then target position."""
        special_vals = {
            SpecialTokens.UNALIGNED.value,
            SpecialTokens.SOURCE_SPECIFIC.value,
            SpecialTokens.TARGET_SPECIFIC.value,
        }
        regular = [
            a
            for a in self.alignment
            if a.source not in special_vals and a.target not in special_vals
        ]
        special = [
            a
            for a in self.alignment
            if a.source in special_vals or a.target in special_vals
        ]
        sorted_regular = self.sort_alignments(regular, source_mapping, target_mapping)
        return TextAlignment(
            alignment=sorted_regular + special,
            source_mapping=source_mapping,
            target_mapping=target_mapping,
        )

    def __eq__(self, other: object) -> bool:
        """Compare TextAlignment objects based on their alignments only.

        Args:
            other: Object to compare with

        Returns:
            True if alignments are equivalent, False otherwise
        """
        if not isinstance(other, TextAlignment):
            return NotImplemented

        # Convert alignments to sets of tuples for comparison
        self_pairs = {(a.source, a.target) for a in self.alignment}
        other_pairs = {(a.source, a.target) for a in other.alignment}

        return self_pairs == other_pairs

    def pairs(self) -> list[tuple[str, str]]:
        """Return [(source, target), ...] pairs for convenience.
        Example:
            >>> ta = TextAlignment(alignment=[TokenAlignment(source="a", target="b")])
            >>> ta.pairs()
            [('a', 'b')]
        """
        return [(a.source, a.target) for a in self.alignment]

    def get_alignment_positions(
        self,
        source_mapping: Optional[TokenMapping] = None,
        target_mapping: Optional[TokenMapping] = None,
    ) -> list[tuple[int, int]]:
        """Get alignment positions using token mappings. If mappings are not
        provided, defaults to the mappings stored on this instance."""

        sm = source_mapping or self.source_mapping
        tm = target_mapping or self.target_mapping
        if sm is None or tm is None:
            raise ValueError(
                "Token mappings are required (provide arguments or ensure TextAlignment has source_mapping/target_mapping)."
            )
        positions: list[tuple[int, int]] = []
        for align in self.alignment:
            s_pos = sm.get_position(align.source)
            t_pos = tm.get_position(align.target)
            if s_pos >= 0 and t_pos >= 0:
                positions.append((s_pos, t_pos))
        return sorted(positions)

    def get_aligned_tokens(self) -> tuple[set[str], set[str]]:
        """Get sets of uniquified aligned source and target tokens.

        Returns:
            Tuple of (source_tokens, target_tokens) sets
        """
        source_tokens = {align.source for align in self.alignment}
        target_tokens = {align.target for align in self.alignment}
        return source_tokens, target_tokens

    def compare_alignments(
        self,
        gold: "TextAlignment",
        source_mapping: TokenMapping,
        target_mapping: TokenMapping,
    ) -> Dict[str, Union[float, int]]:
        """Compare this alignment to a gold standard using position-based comparison."""
        pred_positions = set(
            self.get_alignment_positions(source_mapping, target_mapping)
        )
        gold_positions = set(
            gold.get_alignment_positions(source_mapping, target_mapping)
        )

        true_positives = len(pred_positions & gold_positions)
        precision = true_positives / len(pred_positions) if pred_positions else 0
        recall = true_positives / len(gold_positions) if gold_positions else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "predicted": len(pred_positions),
            "gold": len(gold_positions),
        }


def create_dynamic_alignment_schema(
    source_tokens: list[str],
    target_tokens: list[str],
    marker_generator: Optional[MarkerGenerator] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    tokens_are_uniquified: bool = False,
    use_reasoning: bool = False,
) -> Type[TextAlignmentSchema]:
    """Create a dynamic schema with token-specific validation.

    Args:
        source_tokens: List of source language tokens
        target_tokens: List of target language tokens
        marker_generator: Optional marker generator for unique markers
        min_length: Optional minimum length constraint for alignments
        max_length: Optional maximum length constraint for alignments
        tokens_are_uniquified: If True, assume provided tokens already include unique markers and do not re-uniquify.
        use_reasoning: If True, include reasoning field in schema

    Returns:
        A new TextAlignmentSchema subclass with token-specific validation
    """
    # Use default marker generator if none provided
    if marker_generator is None:
        marker_generator = create_subscript_generator()

    # Build enum token lists:
    # - If tokens_are_uniquified=True, assume tokens already carry unique markers and keep them.
    # - Otherwise, uniquify now (default behavior).
    if tokens_are_uniquified:
        unique_source = source_tokens
        unique_target = target_tokens
    else:
        unique_source = make_unique(source_tokens, marker_generator)
        unique_target = make_unique(target_tokens, marker_generator)

    # Create enums directly from the tokens plus special tokens
    source_enum = create_source_token_enum(unique_source)
    target_enum = create_target_token_enum(unique_target)

    # Calculate alignment constraints using original token counts if not provided
    if min_length is None:
        min_length = 0
    if max_length is None:
        max_length = calculate_max_alignments(source_tokens, target_tokens)

    # Create a new TokenAlignment model with constrained fields
    class DynamicTokenAlignment(TokenAlignment):
        """Dynamic token alignment with enum-based validation."""

        model_config = ConfigDict(use_enum_values=True, extra="forbid")
        source: source_enum  # type: ignore
        target: target_enum  # type: ignore

    # Create the schema class with the dynamic token alignment
    # Split into two definitions based on use_reasoning flag to properly handle field requirements
    if use_reasoning:

        class _DynamicTextAlignmentSchemaWithReasoning(TextAlignmentSchema):
            """Dynamic text alignment schema with required reasoning."""

            model_config = ConfigDict(extra="forbid")

            # Override parent's optional field with required field
            reasoning: str = Field(
                ...,  # Ellipsis makes it required (no default)
                min_length=50,
                max_length=2000,
                description="Provide detailed step-by-step reasoning about alignment decisions, considering: semantic equivalence, syntactic roles, idiomatic expressions, and structural differences between languages. Explain why each alignment choice was made. Minimum 50 characters required.",
            )

            if TYPE_CHECKING:
                alignment: list[TokenAlignment]
            else:
                alignment: list[DynamicTokenAlignment] = Field(
                    min_length=min_length,
                    max_length=max_length,
                )

            @model_validator(mode="after")
            def _filter_invalid_pairs(
                self,
            ) -> "_DynamicTextAlignmentSchemaWithReasoning":
                """Filter out invalid alignment pairs and log warnings."""
                seen: set[tuple[str, str]] = set()
                filtered: list[TokenAlignment] = []
                duplicates_count = 0
                both_unaligned_count = 0

                for a in self.alignment:
                    pair = (a.source, a.target)

                    # Check for both-unaligned pairs
                    if a.source == UNALIGNED_MARKER and a.target == UNALIGNED_MARKER:
                        both_unaligned_count += 1
                        logger.warning(
                            "Filtering out invalid alignment with both source and target as <unaligned>"
                        )
                        continue

                    # Check for duplicates
                    if pair in seen:
                        duplicates_count += 1
                        continue

                    seen.add(pair)
                    filtered.append(a)

                # Log summary if any pairs were filtered
                if duplicates_count > 0:
                    logger.warning(
                        f"Filtered out {duplicates_count} duplicate alignment pair(s)"
                    )
                if both_unaligned_count > 0:
                    logger.warning(
                        f"Filtered out {both_unaligned_count} both-unaligned pair(s)"
                    )

                self.alignment = filtered
                return self

            @classmethod
            def from_base_schema(
                cls, base: TextAlignmentSchema
            ) -> "_DynamicTextAlignmentSchemaWithReasoning":
                """Convert base schema to dynamic schema."""
                # Convert TokenAlignment objects to TokenAlignment to satisfy typing
                alignments_list: list[TokenAlignment] = [
                    TokenAlignment(source=a.source, target=a.target)
                    for a in base.alignment
                ]
                # When use_reasoning=True, we need to provide reasoning (use empty string as fallback)
                reasoning_text = (
                    getattr(base, "reasoning", None)
                    or "Alignment reasoning not provided."
                )
                return cls(alignment=alignments_list, reasoning=reasoning_text)

            def to_base_schema(self) -> TextAlignmentSchema:
                """Convert dynamic schema to base schema."""
                # Convert DynamicTokenAlignment objects back to TokenAlignment
                base_alignments = [
                    TokenAlignment(source=a.source, target=a.target)
                    for a in self.alignment
                ]
                return TextAlignmentSchema(
                    alignment=base_alignments, reasoning=self.reasoning
                )

        # Assign to common variable name for return
        DynamicTextAlignmentSchema = _DynamicTextAlignmentSchemaWithReasoning  # type: ignore[assignment]

    else:

        class _DynamicTextAlignmentSchemaWithoutReasoning(TextAlignmentSchema):
            """Dynamic text alignment schema with optional reasoning."""

            model_config = ConfigDict(extra="forbid")

            # Inherit parent's optional reasoning field (no override needed)

            @classmethod
            def model_json_schema(
                cls,
                by_alias: bool = True,
                ref_template: str = "#/$defs/{model}",
                schema_generator: Optional[Type[GenerateJsonSchema]] = None,
                mode: str = "validation",
                **kwargs: Any,
            ) -> dict[str, Any]:
                """Override to remove reasoning field from schema."""
                # Build kwargs for super call, only including schema_generator if not None
                super_kwargs: dict[str, Any] = {
                    "by_alias": by_alias,
                    "ref_template": ref_template,
                    "mode": mode,  # type: ignore[arg-type]
                    **kwargs,
                }
                if schema_generator is not None:
                    super_kwargs["schema_generator"] = schema_generator  # type: ignore[assignment]

                schema = super().model_json_schema(**super_kwargs)

                # Remove reasoning from properties and required fields
                schema.get("properties", {}).pop("reasoning", None)
                if "required" in schema and "reasoning" in schema["required"]:
                    schema["required"].remove("reasoning")
                return schema

            if TYPE_CHECKING:
                alignment: list[TokenAlignment]
            else:
                alignment: list[DynamicTokenAlignment] = Field(
                    min_length=min_length,
                    max_length=max_length,
                )

            @model_validator(mode="after")
            def _filter_invalid_pairs(
                self,
            ) -> "_DynamicTextAlignmentSchemaWithoutReasoning":
                """Filter out invalid alignment pairs and log warnings."""
                seen: set[tuple[str, str]] = set()
                filtered: list[TokenAlignment] = []
                duplicates_count = 0
                both_unaligned_count = 0

                for a in self.alignment:
                    pair = (a.source, a.target)

                    # Check for both-unaligned pairs
                    if a.source == UNALIGNED_MARKER and a.target == UNALIGNED_MARKER:
                        both_unaligned_count += 1
                        logger.warning(
                            "Filtering out invalid alignment with both source and target as <unaligned>"
                        )
                        continue

                    # Check for duplicates
                    if pair in seen:
                        duplicates_count += 1
                        continue

                    seen.add(pair)
                    filtered.append(a)

                # Log summary if any pairs were filtered
                if duplicates_count > 0:
                    logger.warning(
                        f"Filtered out {duplicates_count} duplicate alignment pair(s)"
                    )
                if both_unaligned_count > 0:
                    logger.warning(
                        f"Filtered out {both_unaligned_count} both-unaligned pair(s)"
                    )

                self.alignment = filtered
                return self

            @classmethod
            def from_base_schema(
                cls, base: TextAlignmentSchema
            ) -> "_DynamicTextAlignmentSchemaWithoutReasoning":
                """Convert base schema to dynamic schema."""
                # Convert TokenAlignment objects to TokenAlignment to satisfy typing
                alignments_list: list[TokenAlignment] = [
                    TokenAlignment(source=a.source, target=a.target)
                    for a in base.alignment
                ]
                return cls(alignment=alignments_list)

            def to_base_schema(self) -> TextAlignmentSchema:
                """Convert dynamic schema to base schema."""
                # Convert DynamicTokenAlignment objects back to TokenAlignment
                base_alignments = [
                    TokenAlignment(source=a.source, target=a.target)
                    for a in self.alignment
                ]
                return TextAlignmentSchema(alignment=base_alignments)

        # Assign to common variable name for return
        DynamicTextAlignmentSchema = _DynamicTextAlignmentSchemaWithoutReasoning  # type: ignore[assignment]

    logger.debug(
        "Created dynamic schema:\n%s",
        json.dumps(
            DynamicTextAlignmentSchema.model_json_schema(), indent=2, ensure_ascii=False
        ),
    )

    return DynamicTextAlignmentSchema


class AlignmentAttempt(BaseModel):
    """Records details of a single alignment attempt"""

    attempt_number: int
    messages_sent: list[ChatMessageDict]
    raw_response: Optional[TextAlignment]
    validation_passed: bool
    validation_errors: list[tuple[ValidationErrorType, str, list[str]]]
    exception: Optional[str] = None

    @property
    def validation_errors_typed(self) -> List["ValidationErrorDict"]:
        """
        Return validation errors as typed dictionaries.
        Example:
            >>> from lexi_align.models import AlignmentAttempt, ValidationErrorType
            >>> a = AlignmentAttempt(attempt_number=1, messages_sent=[], raw_response=None, validation_passed=False, validation_errors=[(ValidationErrorType.OTHER, "oops", ["x"])])
            >>> a.validation_errors_typed[0]["message"]
            'oops'
        """
        return [
            {"type": et, "message": msg, "tokens": toks}
            for et, msg, toks in self.validation_errors
        ]


class AlignmentResult(BaseModel):
    """Enhanced result containing full diagnostic information"""

    alignment: Optional[TextAlignment]
    attempts: list[AlignmentAttempt]
