from nexora_rag.generation.rag_service import normalize_citations


def test_fullwidth_brackets_become_ascii():
    assert normalize_citations("cheaper\u30101\u3011.") == "cheaper[1]."


def test_multiple_citations():
    assert normalize_citations("No\u30101\u3011\u30102\u3011.") == "No[1][2]."


def test_extra_text_inside_brackets_is_dropped():
    assert normalize_citations("Text\u30101\u2020source\u3011.") == "Text[1]."


def test_ascii_citations_are_untouched():
    assert normalize_citations("Text [2].") == "Text [2]."