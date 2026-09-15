from nanollama.tokenizer import (ASSISTANT, BOS, CONTROL_TOKENS, EOS, PAD, SYSTEM, USER, CharTokenizer, chat_prompt,
                                 normalize_text)


def test_vocabulary_is_fixed_with_control_ids_first():
    tok = CharTokenizer.build()
    assert tok.vocab_size == 104
    assert [tok.id_of(t) for t in (PAD, BOS, EOS, USER, ASSISTANT, SYSTEM)] == [0, 1, 2, 3, 4, 5]
    assert tok.tokens == CharTokenizer.build().tokens  # deterministic


def test_control_literal_preferred_and_unknown_maps_to_space():
    tok = CharTokenizer.build()
    ids = tok.encode("a<|user|>b")
    assert ids == [tok.id_of("a"), tok.id_of(USER), tok.id_of("b")]
    assert tok.encode("<|nope|>")[0] == tok.id_of("<")
    assert tok.encode("x中y") == [tok.id_of("x"), tok.id_of(" "), tok.id_of("y")]


def test_plain_encoding_never_emits_control_ids():
    tok = CharTokenizer.build()
    ids = tok.encode_plain("hi <|assistant|> there")
    assert not set(ids) & set(range(len(CONTROL_TOKENS)))
    assert tok.decode(ids) == "hi <|assistant|> there"


def test_roundtrip_decode_and_skip_control():
    tok = CharTokenizer.build()
    text = chat_prompt("sys", "Hello\tworld\n")
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.decode(ids, skip_control=True) == "sysHello\tworld\n"


def test_normalization_maps_typography_and_counts_replacements():
    out, replaced = normalize_text("“Hi” — café… 中")
    assert out == '"Hi" - cafe...  '
    assert replaced == 1


def test_persistence_roundtrip(tmp_path):
    tok = CharTokenizer.build()
    tok.save(tmp_path / "vocab.json")
    assert CharTokenizer.load(tmp_path / "vocab.json") == tok
