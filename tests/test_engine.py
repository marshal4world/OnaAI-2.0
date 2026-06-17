from onaai.engine import ReasoningEngine, ReasoningResult


def test_parse_think_and_boxed():
    raw = "<think>7^100 mod 13 cycles with period 12, 100 mod 12 = 4, 7^4=2401, mod13=9</think>\nThe answer is \\boxed{9}."
    result = ReasoningEngine.parse("q", raw)
    assert isinstance(result, ReasoningResult)
    assert result.answer == "9"
    assert "period 12" in result.reasoning


def test_parse_nested_boxed_braces():
    raw = "result: \\boxed{\\frac{1}{2}}"
    result = ReasoningEngine.parse("q", raw)
    assert result.answer == "\\frac{1}{2}"


def test_parse_last_boxed_wins():
    raw = "\\boxed{1} then corrected to \\boxed{42}"
    result = ReasoningEngine.parse("q", raw)
    assert result.answer == "42"


def test_parse_plain_text_without_markers():
    raw = "The answer is simply 5."
    result = ReasoningEngine.parse("q", raw)
    assert result.answer == "The answer is simply 5."
    assert result.reasoning == ""


def test_str_returns_answer():
    result = ReasoningEngine.parse("q", "\\boxed{7}")
    assert str(result) == "7"
