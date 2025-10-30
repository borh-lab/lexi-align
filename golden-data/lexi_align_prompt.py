# lexi-align prompt
from lexi_align.models import TextAlignment, TokenAlignment

GUIDELINES = """
You are an expert Japanese↔Japanese alignment annotator.
Given space-delimited tokens for a source and target sentence, align them. Do not add, split, merge or normalize tokens.
Priorities:
1. Cover the source side as fully as possible;
2. Keep alignments monotonic in source order (do not jump backwards on the source side);
3. Prefer 1–1 alignment, but allow 1–1, 1–N, N–1, N–N;
4. Align content words first; function words (e.g. particles, adverbs) only when a clear counterpart exists;
5. Handle form shifts (e.g., proper noun ↔ pronoun with same role);
6. Align auxiliaries when present on both sides; if it exists on only one side, align it to the same token(s) as its main verb;
7. If duplicate token appear, align left-to-right (earliest feasible counterpart first).
8. If no counterpart exists, leave the token unaligned.
"""

EXAMPLES_JP_JP = [
    (
        "大勢の 人人が 胸壁に 凭れて、 下の 街路を 見下してゐるのを 見、 イッポリタは 驚きの 声を 揚げて 步みを 止めた。".split(),
        "イッポリタは、 大勢の 人が、 欄干から 乗りだすようにして、 下の 道路を 見おろしているのを 見て、 急に 立ちどまった。".split(),
        TextAlignment(
            alignment=[
                TokenAlignment(source="大勢の", target="大勢の"),
                TokenAlignment(source="人人が", target="人が、"),
                TokenAlignment(source="胸壁に", target="欄干から"),
                TokenAlignment(source="凭れて、", target="乗りだすようにして、"),
                TokenAlignment(source="下の", target="下の"),
                TokenAlignment(source="街路を", target="道路を"),
                TokenAlignment(source="見下してゐるのを", target="見おろしているのを"),
                TokenAlignment(source="見、", target="見て、"),
                TokenAlignment(source="イッポリタは", target="イッポリタは、"),
                TokenAlignment(source="步みを", target="立ちどまった。"),
                TokenAlignment(source="止めた。", target="立ちどまった。"),
            ]
        ),
    ),
    (
        "着くと 直ぐ、 どんどん 火を 焚か して お茶を 飲みませう。".split(),
        "宿へ ついたら、 火を うんと 焚いて、 お茶を のみましょうね".split(),
        TextAlignment(
            alignment=[
                TokenAlignment(source="着くと", target="ついたら、"),
                TokenAlignment(source="どんどん", target="うんと"),
                TokenAlignment(source="火を", target="火を"),
                TokenAlignment(source="焚か", target="焚いて、"),
                TokenAlignment(source="して", target="焚いて、"),
                TokenAlignment(source="お茶を", target="お茶を"),
                TokenAlignment(source="飲みませう。", target="のみましょうね"),
            ]
        ),
    ),
    (
        "彼は 夫の 監視してゐた 時分を、 殘り惜しさうに 思ひ返した。".split(),
        "そして、 彼女の 夫の 監視の 目が あった ときの方が よかったと、 また そう 考えて、 ある 種の 哀惜に 似た 気持ちを 抱くのだった。".split(),
        TextAlignment(
            alignment=[
                TokenAlignment(source="夫の", target="夫の"),
                TokenAlignment(source="監視してゐた", target="監視の"),
                TokenAlignment(source="監視してゐた", target="目が"),
                TokenAlignment(source="監視してゐた", target="あった"),
                TokenAlignment(source="時分を、", target="ときの方が"),
                TokenAlignment(source="殘り惜しさうに", target="ある"),
                TokenAlignment(source="殘り惜しさうに", target="種の"),
                TokenAlignment(source="殘り惜しさうに", target="哀惜に"),
                TokenAlignment(source="殘り惜しさうに", target="似た"),
                TokenAlignment(source="殘り惜しさうに", target="気持ちを"),
                TokenAlignment(source="思ひ返した。", target="抱くのだった。"),
            ]
        ),
    ),
]
