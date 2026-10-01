import pytest

from nexora_rag.guardrails.pii import mask_pii


def test_normal_answer_is_unchanged():
    text = "Employees get 20 days of annual leave [1]. Claims are due within 14 days [2]."
    assert mask_pii(text) == text


def test_email_is_masked():
    out = mask_pii("Contact hr@nexora.com for help.")
    assert "hr@nexora.com" not in out
    assert "<EMAIL_ADDRESS>" in out


def test_credit_card_is_masked():
    out = mask_pii("Card: 4111 1111 1111 1111")
    assert "4111" not in out


@pytest.mark.parametrize("secret", [
    "35202-1234567-1",
    "0300-1234567",
    "+92 300 1234567",
])
def test_pakistani_pii_is_masked(secret):
    out = mask_pii(f"Details: {secret} end")
    assert secret not in out


def test_empty_text():
    assert mask_pii("") == ""