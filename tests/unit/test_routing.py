import pytest

from nexora_rag.retrieval.routing import check_small_talk


@pytest.mark.parametrize(
    "message",
    [
        "hello",
        "Hello!",
        "HI",
        "hey there",
        "salam",
        "Assalam o Alaikum",
        "good morning",
        "thanks",
        "Thank you so much!",
        "shukriya",
        "bye",
        "Allah Hafiz",
    ],
)
def test_small_talk_gets_a_reply(message):
    assert check_small_talk(message) is not None


@pytest.mark.parametrize(
    "message",
    [
        "How many days of paid maternity leave are there?",
        "hello, how many maternity leave days?",
        "thanks, and what about G4?",
        "what is the hi-speed rail budget?",
        "aur G4 ke liye?",
        "",
    ],
)
def test_real_questions_go_to_rag(message):
    assert check_small_talk(message) is None