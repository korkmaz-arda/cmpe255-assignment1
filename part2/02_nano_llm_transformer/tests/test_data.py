import json

import numpy as np
import pytest

from nanollama import kb
from nanollama.data import tinystories
from nanollama.data.records import Dialogue, norm_key
from nanollama.data.store import _open_final_split, open_training_split, read_manifest
from nanollama.data.windows import WindowStats, window_dialogue
from nanollama.tokenizer import ASSISTANT, BOS, SYSTEM, USER, CharTokenizer

TOK = CharTokenizer.build()


def _supervised_positions(w):
    """Label positions (1..len-1) flagged as targets."""
    return np.flatnonzero(w.target[1:]) + 1


def _assistant_stream(d: Dialogue) -> list[int]:
    out = []
    for role, text in d.turns:
        if role == "assistant":
            out += TOK.encode_plain(text) + [TOK.eos_id]
    return out


def test_parse_record_handles_field_order_and_missing_story():
    rec = "Summary: s\nStory: \nOnce upon a time.\nThe end.\nFeatures: Dialogue\nWords: a, b"
    f = tinystories.parse_record(rec)
    assert f["Story"] == "Once upon a time.\nThe end."
    assert f["Features"] == "Dialogue" and f["Words"] == "a, b"
    assert tinystories.parse_record("Summary: only a summary") is None
    instr = tinystories.instruction_from_fields(f)
    assert instr.index("Use these words") < instr.index("Story features") < instr.index("Summary")


def test_fitting_example_is_one_unchanged_window_with_assistant_only_targets():
    d = Dialogue("x", "kb", "sys", [("user", "hi"), ("assistant", "hello")])
    w = window_dialogue(d, TOK, 128, WindowStats())[0]
    expected = [TOK.id_of(BOS), TOK.id_of(SYSTEM)] + TOK.encode_plain("sys") + [TOK.id_of(USER)] + TOK.encode_plain("hi") \
        + [TOK.id_of(ASSISTANT)] + TOK.encode_plain("hello") + [TOK.eos_id]
    assert w.tokens.tolist() == expected
    assert [int(t) for t in w.tokens[_supervised_positions(w)]] == TOK.encode_plain("hello") + [TOK.eos_id]


def test_long_story_chunks_keep_prefix_and_supervise_each_character_once():
    story = "".join(f"Sentence {i} about a cat. " for i in range(120))
    d = Dialogue("s", "tinystories", "You tell stories.", [("user", "Write a story."), ("assistant", story)])
    stats = WindowStats()
    ws = window_dialogue(d, TOK, 256, stats)
    assert stats.chunked == 1 and len(ws) > 3
    prefix = [TOK.id_of(BOS), TOK.id_of(SYSTEM)] + TOK.encode_plain("You tell stories.") + [TOK.id_of(USER)] \
        + TOK.encode_plain("Write a story.") + [TOK.id_of(ASSISTANT)]
    supervised = []
    for w in ws:
        assert len(w.tokens) <= 257
        assert w.tokens[: len(prefix)].tolist() == prefix
        supervised += [int(t) for t in w.tokens[_supervised_positions(w)]]
    assert supervised == _assistant_stream(d)  # exactly once, in order


def test_long_conversation_windows_along_turns_and_keep_roles():
    turns = []
    for i in range(8):
        turns += [("user", f"Question {i}? " * 3), ("assistant", f"Answer {i}. " * 6)]
    d = Dialogue("c", "everyday", "sys", turns)
    stats = WindowStats()
    ws = window_dialogue(d, TOK, 256, stats)
    assert stats.turn_windowed == 1 and len(ws) > 1
    supervised = []
    for w in ws:
        assert w.tokens[0] == TOK.id_of(BOS) and w.tokens[1] == TOK.id_of(SYSTEM)
        first_role = next(int(t) for t in w.tokens if int(t) in (TOK.id_of(USER), TOK.id_of(ASSISTANT)))
        assert first_role == TOK.id_of(USER)
        supervised += [int(t) for t in w.tokens[_supervised_positions(w)]]
        # nothing inside a user turn is a target
        in_user = False
        for t, flag in zip(w.tokens.tolist(), w.target.tolist()):
            if t == TOK.id_of(USER):
                in_user = True
            elif t == TOK.id_of(ASSISTANT):
                in_user = False
            assert not (in_user and flag)
    assert supervised == _assistant_stream(d)


def test_unusable_example_is_counted_as_dropped():
    d = Dialogue("d", "kb", "s" * 250, [("user", "q"), ("assistant", "a" * 400)])
    stats = WindowStats()
    assert window_dialogue(d, TOK, 256, stats) == []
    assert stats.dropped == 1


def test_kb_split_is_entry_level_and_final_answers_never_trained():
    parts = kb.build_kb_splits()
    entries = kb.split_entries()
    keys = {r: {e.key for e in entries[r]} for r in kb.KB_ROLES}
    assert not (keys["train"] & keys["val"]) and not (keys["train"] & keys["test"]) and not (keys["val"] & keys["test"])
    train_answers = {norm_key(d.turns[1][1]) for d in parts["train"]}
    for role in ("val", "test"):
        assert all(norm_key(d.turns[1][1]) not in train_answers for d in parts[role])
    # the rephrasing diagnostic: answer seen, wording withheld
    train_queries = {norm_key(d.turns[0][1]) for d in parts["train"]}
    for d in parts["rephrase"]:
        assert norm_key(d.turns[1][1]) in train_answers
        assert norm_key(d.turns[0][1]) not in train_queries
    ids = [d.id for d in parts["train"]]
    assert len(ids) == len(set(ids))  # no dialogue emitted twice


def test_prepared_splits_are_disjoint_and_report_adds_up(prepared):
    manifest = read_manifest(prepared)
    for source in ("tinystories", "everyday", "kb"):
        owners = {}
        for split in ("train", "val"):
            ws, dl = open_training_split(prepared, split, source)
            owners[split] = {d.id for d in dl}
            stats = manifest["report"]["windows"][source][split]
            assert stats["examples"] == stats["fit_directly"] + stats["turn_windowed"] + stats["chunked"] + stats["dropped"]
            assert stats["windows"] == len(ws)
        ws, dl = _open_final_split(prepared, source)
        owners["test"] = {d.id for d in dl}
        assert not owners["train"] & owners["val"]
        assert not owners["train"] & owners["test"]
        assert not owners["val"] & owners["test"]
    rep = manifest["report"]
    ts_train = rep["windows"]["tinystories"]["train"]
    assert ts_train["windows"] > 0 and ts_train["chunked"] > 0 and ts_train["dropped"] == 0
    assert rep["tinystories"]["train_unparseable"] == 1  # the fixture record without a Story field
    assert rep["everyday"]["trailing_user_turn_trimmed"]["train"] == 10
    # dedup: the fixture's train stories never repeat val/test story text
    _, test_dl = _open_final_split(prepared, "tinystories")
    _, train_dl = open_training_split(prepared, "train", "tinystories")
    assert not {norm_key(d.turns[1][1]) for d in test_dl} & {norm_key(d.turns[1][1]) for d in train_dl}


def test_training_accessor_refuses_test_split(prepared):
    with pytest.raises(PermissionError):
        open_training_split(prepared, "test", "tinystories")


def test_live_chat_prompt_matches_training_serialization():
    """Inference prompts (with history) must be an exact prefix of the training window."""
    from nanollama.generate import build_prompt_ids

    d = Dialogue("c", "everyday", "You are NanoLlama, a small helpful assistant.", [
        ("user", "Hi"), ("assistant", "Hello! How can I help you today?"),
        ("user", "What is a “good” snack?"), ("assistant", "An apple."),
    ])
    w = window_dialogue(d, TOK, 1024, WindowStats())[0].tokens.tolist()
    for k in (0, 2):
        live = build_prompt_ids(TOK, d.system, d.turns[k][1], d.turns[:k])
        assert w[: len(live)] == live
        assert live[-1] == TOK.id_of(ASSISTANT)
    # the reply that follows is supervised and closed by EOS in training
    live = build_prompt_ids(TOK, d.system, d.turns[2][1], d.turns[:2])
    assert w[len(live):] == TOK.encode_plain("An apple.") + [TOK.eos_id]
