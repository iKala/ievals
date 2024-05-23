import re
import string
from ..answer_parser import match_response_choices

TRADITIONAL_COT = [
    r"答案為(.+?)",
    r"選項(.+?)是正確的",
    r"因此，選項(.+?)",
    r"答案是(.+?)",
]

SIMPLIFIED_COT = [
    r"答案为(.+?)",
    r"选项(.+?)是正确的",
    r"因此，选项(.+?)",
    r"答案是(.+?)"
]

class Evaluator:
    def __init__(self, choices, model_name, k=-1):
        self.choices = choices
        self.model_name = model_name
        self.k = k
        self.puncs = list(string.punctuation)

    def format_example(self, line, include_answer=True):
        example = line["question"]
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        example += "\n答案："
        if include_answer:
            example += f'{line["answer"]}\n\n'
        return example

    def generate_few_shot_prompt(self, subject, dev_df):
        prompt = f"以下是關於{subject}考試單選題，請選出正確的答案。\n\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += self.format_example(dev_df.iloc[i, :])
        return prompt

    def eval_subject(
        self, subject_name, test_df, dev_df=None, few_shot=False, save_result_dir=None
    ):
        pass

    def normalize_answer(self, s):
        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(self.puncs)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))

    def exact_match(self, pred, target):
        return self.normalize_answer(pred) == self.normalize_answer(target)

    def extract_ans(self, response_str: str):
        return match_response_choices(response_str, self.converter)

    def cot_match_response_choice(response_str: str, is_simplified=False):
        ans_list = re.findall(r"答案是(.+?)。", response_str)
        prompt_choices = TRADITIONAL_COT
        if is_simplified:
            prompt_choices = SIMPLIFIED_COT
        for prompt_regex in prompt_choices:
            ans_list = re.findall(prompt_regex, response_str)
            if len(ans_list) != 0:
                return ans_list
        # no answer found
        return []