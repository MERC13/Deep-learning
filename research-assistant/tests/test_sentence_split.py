from digest.compute_digest import _sentence_split


def test_sentence_split_basic():
    text = "This is a sentence. This is another! And a third?  New line split\ncontinues."
    sents = _sentence_split(text)
    assert any("This is a sentence." in s for s in sents)
    assert any("This is another!" in s for s in sents)
    assert any("And a third?" in s for s in sents)
