from onaai.training.reward import (
    answers_match,
    extract_answer,
    normalize_answer,
    verifiable_reward,
)


def test_extract_boxed_answer():
    assert extract_answer("<think>work</think>\n\\boxed{144}") == "144"


def test_extract_after_think_without_box():
    assert extract_answer("<think>reasoning here</think>\nThe answer is 42") == "The answer is 42"


def test_extract_plain():
    assert extract_answer("just 7") == "just 7"


def test_normalize_numbers():
    assert normalize_answer("1,000") == "1000"
    assert normalize_answer("5.0") == "5"
    assert normalize_answer(" $144$ ") == "144"
    assert normalize_answer("Yes.") == "yes"


def test_answers_match():
    assert answers_match("\\boxed{144}", "144")
    assert answers_match("the result is \\boxed{1,000}", "1000")
    assert not answers_match("\\boxed{7}", "8")


def test_verifiable_reward_binary():
    assert verifiable_reward("<think>..</think>\\boxed{15}", "15") == 1.0
    assert verifiable_reward("<think>..</think>\\boxed{16}", "15") == 0.0
