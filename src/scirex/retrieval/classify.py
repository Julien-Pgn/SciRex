"""Local LLM classification pass: prompt construction, model call, verdict parsing.

load_classifier imports transformers/torchao lazily, same reasoning as embed.py.
Everything else here (prompt/few-shot construction, verdict parsing, example
sampling) is pure and tested without touching a GPU.
"""

import re

import pandas as pd


def build_few_shot_block(examples: list[tuple[str, str, bool]], abstract_chars: int = 400) -> str:
    """Format labeled (title, abstract, is_relevant) examples for the prompt."""
    return "\n".join(
        f'Title: {title}\nAbstract: {abstract[:abstract_chars]}\nAnswer: {"yes" if is_relevant else "no"}\n'
        for title, abstract, is_relevant in examples
    )


def select_few_shot_examples(
    golden: pd.DataFrame, n_positive: int = 3, n_negative: int = 3, random_state: int = 42
) -> pd.DataFrame:
    """Sample a few labeled positives/negatives from a topic's golden set for the prompt.

    Whatever rows this returns must be excluded from evaluation afterward — never
    score the classifier on the examples it was shown.
    """
    positives = golden[golden["label"] == 1].sample(n_positive, random_state=random_state)
    negatives = golden[golden["label"] == 0].sample(n_negative, random_state=random_state)
    return pd.concat([positives, negatives])


def build_prompt(rubric: str, few_shot_block: str, title: str, abstract: str, abstract_chars: int = 800) -> str:
    """Assemble the classification prompt: topic-specific rubric + labeled examples + the target paper."""
    return f"""{rubric}
Examples:
{few_shot_block}
Now classify this paper. First, in one short sentence, name the paper's core technique. Then answer yes or no.

Title: {title}
Abstract: {abstract[:abstract_chars]}

Reasoning:"""


def parse_verdict(generated_text: str) -> bool:
    """Extract the yes/no verdict from the model's raw output.

    Takes the LAST yes/no found — robust to the model restating the question,
    or trailing punctuation/markdown around the final word.
    """
    matches = re.findall(r"\b(yes|no)\b", generated_text.strip().lower())
    return matches[-1] == "yes" if matches else False


def load_classifier(model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda:0"):
    """Load the local LLM classifier, int8 via torchao (bitsandbytes doesn't support CUDA 13).
    Not unit-tested — needs real GPU weights, exercised manually."""
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        quantization_config=TorchAoConfig(Int8WeightOnlyConfig()),
        torch_dtype="auto",
    )
    return tokenizer, model


def classify_paper(tokenizer, model, prompt: str, max_new_tokens: int = 60) -> bool:
    """Run one classification prompt through the LLM and parse its yes/no verdict."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return parse_verdict(text)