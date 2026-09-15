"""Character-level tokenizer with chat control tokens (F01).

The vocabulary is fixed and deterministic: six control tokens in the lowest ids,
then one id per printable ASCII character (space..tilde) plus newline, tab and
carriage return — 104 entries. Nothing is learned from the corpus.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

PAD, BOS, EOS, USER, ASSISTANT, SYSTEM = "<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|system|>"
CONTROL_TOKENS = [PAD, BOS, EOS, USER, ASSISTANT, SYSTEM]
CONTROL_DESCRIPTIONS = {
    PAD: "Padding: fills unused positions so a batch has one length; never trained on.",
    BOS: "Beginning of sequence: the first token of every training window and prompt.",
    EOS: "End of sequence: the model emits it to stop its reply; generation halts on it.",
    USER: "Marks the start of a user turn in the chat template.",
    ASSISTANT: "Marks the start of an assistant turn; the model's reply follows it.",
    SYSTEM: "Marks the system (persona) prompt at the start of a conversation.",
}
IGNORE_INDEX = -100

# Typographic characters common in the source datasets, mapped to ASCII before
# the generic NFKD fold. Anything still outside the vocabulary becomes a space.
_ASCII_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "′": "'", "″": '"',
    "–": "-", "—": "-", "‒": "-", "―": "-", "−": "-",
    "…": "...", " ": " ", " ": " ", "​": "",
    "•": "*", "×": "x", "÷": "/",
}


def normalize_text(text: str) -> tuple[str, int]:
    """Map text into the ASCII vocabulary; returns (text, n_replaced_with_space)."""
    out: list[str] = []
    replaced = 0
    for ch in text:
        if 32 <= ord(ch) < 127 or ch in "\n\t\r":
            out.append(ch)
            continue
        if ch in _ASCII_MAP:
            out.append(_ASCII_MAP[ch])
            continue
        folded = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii")
        folded = "".join(c for c in folded if 32 <= ord(c) < 127)
        if folded:
            out.append(folded)
        else:
            out.append(" ")
            replaced += 1
    return "".join(out), replaced


@dataclass(frozen=True)
class CharTokenizer:
    tokens: tuple[str, ...]

    @classmethod
    def build(cls) -> "CharTokenizer":
        chars = [chr(c) for c in range(32, 127)] + ["\n", "\t", "\r"]
        return cls(tuple(CONTROL_TOKENS + chars))

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def stoi(self) -> dict[str, int]:
        return _stoi(self.tokens)

    def id_of(self, token: str) -> int:
        return self.stoi[token]

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def control_ids(self) -> list[int]:
        return list(range(len(CONTROL_TOKENS)))

    def encode(self, text: str) -> list[int]:
        """Left-to-right scan; control-token literals win over their characters.

        Characters outside the vocabulary fall back to the space id. Callers that
        want typographic folding should run :func:`normalize_text` first.
        """
        stoi = self.stoi
        space = stoi[" "]
        ids: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            if text[i] == "<":
                for ctl in CONTROL_TOKENS:
                    if text.startswith(ctl, i):
                        ids.append(stoi[ctl])
                        i += len(ctl)
                        break
                else:
                    ids.append(stoi["<"])
                    i += 1
                continue
            ids.append(stoi.get(text[i], space))
            i += 1
        return ids

    def encode_plain(self, text: str) -> list[int]:
        """Encode content text character-by-character, never emitting control ids.

        Used for message bodies so a literal "<|user|>" typed inside a message
        cannot forge a role boundary.
        """
        stoi = self.stoi
        space = stoi[" "]
        return [stoi.get(ch, space) for ch in text]

    def decode(self, ids, skip_control: bool = False) -> str:
        n_ctl = len(CONTROL_TOKENS)
        parts = []
        for i in ids:
            i = int(i)
            if skip_control and i < n_ctl:
                continue
            parts.append(self.tokens[i])
        return "".join(parts)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"tokens": list(self.tokens), "n_control": len(CONTROL_TOKENS)}, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text())
        return cls(tuple(data["tokens"]))


_STOI_CACHE: dict[tuple[str, ...], dict[str, int]] = {}


def _stoi(tokens: tuple[str, ...]) -> dict[str, int]:
    table = _STOI_CACHE.get(tokens)
    if table is None:
        table = {t: i for i, t in enumerate(tokens)}
        _STOI_CACHE[tokens] = table
    return table


def format_chat(system: str, turns: list[tuple[str, str]]) -> str:
    """Render a conversation as template text (without BOS).

    ``turns`` is a list of (role, text) with role in {"user", "assistant"}.
    Assistant turns are closed by EOS. To build an inference prompt, end the
    list with a user turn and append ASSISTANT yourself (see ``chat_prompt``).
    """
    parts = [SYSTEM, system]
    for role, text in turns:
        if role == "user":
            parts += [USER, text]
        elif role == "assistant":
            parts += [ASSISTANT, text, EOS]
        else:
            raise ValueError(f"unknown role {role!r}")
    return "".join(parts)


def chat_prompt(system: str, user: str, history: list[tuple[str, str]] | None = None) -> str:
    """Canonical inference prompt: system + (history) + user + assistant marker."""
    turns = list(history or []) + [("user", user)]
    return format_chat(system, turns) + ASSISTANT
