import pandas as pd

from scirex.retrieval.classify import (
    build_few_shot_block,
    build_prompt,
    parse_verdict,
    select_few_shot_examples,
)


def test_build_few_shot_block_formats_yes_no():
    examples = [("Title A", "Abstract A", True), ("Title B", "Abstract B", False)]
    block = build_few_shot_block(examples)
    assert "Answer: yes" in block
    assert "Answer: no" in block
    assert "Title A" in block and "Title B" in block


def test_build_few_shot_block_truncates_abstract():
    examples = [("T", "x" * 1000, True)]
    block = build_few_shot_block(examples, abstract_chars=10)
    assert "x" * 11 not in block


def test_build_prompt_includes_rubric_examples_and_target_paper():
    prompt = build_prompt("RUBRIC TEXT", "FEW SHOT BLOCK", "My Title", "My abstract")
    assert "RUBRIC TEXT" in prompt
    assert "FEW SHOT BLOCK" in prompt
    assert "My Title" in prompt
    assert "My abstract" in prompt


def test_parse_verdict_takes_last_yes_no_in_text():
    assert parse_verdict("The paper is about X. Answer: yes.") is True
    assert parse_verdict("hmm, tricky. no") is False


def test_parse_verdict_defaults_false_when_no_match():
    assert parse_verdict("I cannot determine this.") is False


def test_parse_verdict_ignores_case_and_punctuation():
    assert parse_verdict("YES!") is True


def test_select_few_shot_examples_returns_requested_counts():
    golden = pd.DataFrame({
        "title": [f"T{i}" for i in range(10)],
        "label": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    })
    few_shot = select_few_shot_examples(golden, n_positive=2, n_negative=2)
    assert len(few_shot) == 4
    assert (few_shot["label"] == 1).sum() == 2
    assert (few_shot["label"] == 0).sum() == 2