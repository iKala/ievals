import os
import re
import logging
from time import sleep
import openai
import opencc
from tqdm import tqdm
from .evaluator import Evaluator


def replace_invalid_brackets(text):
    # Define a comprehensive dictionary for bracket replacements
    replacements = {
        "⟦": "[",  # Mathematical left white square bracket
        "⟧": "]",  # Mathematical right white square bracket
        "【": "[",  # Left black lenticular bracket
        "】": "]",  # Right black lenticular bracket
        "〖": "[",  # Left white lenticular bracket
        "〗": "]",  # Right white lenticular bracket
        "⦃": "{",  # Mathematical left white curly bracket
        "⦄": "}",  # Mathematical right white curly bracket
        "❨": "(",  # Medium left parenthesis ornament
        "❩": ")",  # Medium right parenthesis ornament
        "⟮": "(",  # Mathematical left flattened parenthesis
        "⟯": ")",  # Mathematical right flattened parenthesis
        "（": "(",
        "）": ")",
        # Add more replacements as needed
    }

    def replace_char(match):
        return replacements.get(
            match.group(0), match.group(0)
        )  # No default replacement

    pattern = re.compile("|".join(re.escape(key) for key in replacements.keys()))
    cleaned_text = pattern.sub(replace_char, text)

    return cleaned_text


class ChatGPT_Evaluator(Evaluator):
    def __init__(self, choices, k, api_key, model_name, switch_zh_hans=False):
        super(ChatGPT_Evaluator, self).__init__(choices, model_name, k)
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.converter = None
        self.switch_zh_hans = switch_zh_hans
        if switch_zh_hans:
            self.converter = opencc.OpenCC("t2s.json")

    def format_example(self, line, include_answer=True, cot=False):
        example = line["question"]
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'

        example += "\n答案："
        if include_answer:
            if cot:
                ans = line["answer"]
                content = "讓我們一步一步思考，\n" + line["explaination"] + f"\n所以答案是{ans}。"
                return [
                    {"role": "user", "content": example},
                    {"role": "assistant", "content": content},
                ]
            else:
                return [
                    {"role": "user", "content": example},
                    {"role": "assistant", "content": line["answer"]},
                ]
        else:
            return [
                {"role": "user", "content": example},
            ]

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = [
            {
                "role": "system",
                "content": f"你是一位專業的中文AI助理，以下是關於{subject}考試單選題，請選出正確的答案。",
            }
        ]
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            tmp = self.format_example(dev_df.iloc[i, :], include_answer=True, cot=cot)
            if i == 0:
                tmp[0]["content"] = (
                    f"以下是關於{subject}考試單選題，請選出正確的答案。\n\n" + tmp[0]["content"]
                )
                if self.converter:
                    tmp[0]["content"] = self.converter.convert(tmp[0]["content"])
                prompt += tmp

        return prompt

    def eval_subject(
        self,
        subject_name,
        test_df,
        dev_df=None,
        few_shot=False,
        save_result_dir=None,
        cot=False,
    ):
        correct_num = 0
        if save_result_dir:
            result = []
            score = []
        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(
                subject_name, dev_df, cot=cot
            )
        else:
            few_shot_prompt = [
                {
                    "role": "system",
                    "content": f"你是一位專業的中文AI助理，以下是關於{subject_name}考試單選題，請選出正確的答案。",
                }
            ]
        answers = list(test_df["answer"])
        for row_index, row in tqdm(
            test_df.iterrows(), total=len(test_df), dynamic_ncols=True
        ):
            question = self.format_example(row, include_answer=False)
            full_prompt = few_shot_prompt + question
            if not few_shot:
                full_prompt[-1]["content"] = (
                    f"以下是關於{subject_name}考試單選題，請選出正確的答案。\n\n"
                    + full_prompt[-1]["content"]
                )
            response = None
            timeout_counter = 0
            if self.converter:  # convert to simplified chinese
                for idx, prompt in enumerate(full_prompt):
                    full_prompt[idx]["content"] = self.converter.convert(
                        prompt["content"]
                    )

            while response is None and timeout_counter <= 30:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=full_prompt,
                        temperature=0.0,
                        max_tokens=800,
                    )
                except openai.InternalServerError:
                    for idx, prompt in enumerate(full_prompt):
                        full_prompt[idx]["content"] = replace_invalid_brackets(
                            prompt["content"]
                        )
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=full_prompt,
                        temperature=0.3,
                        max_tokens=800,
                    )
                except Exception as msg:
                    if "timeout=600" in str(msg):
                        timeout_counter += 1
                    logging.error(msg)
                    sleep(5)
                    continue
            if response == None:
                response_str = ""
            else:
                response_str = response.choices[0].message.content
            if cot:
                ans_list = self.cot_match_response_choice(
                    response_str, is_simplified=self.switch_zh_hans
                )

                if len(ans_list) == 0:
                    correct = 0
                else:
                    if self.exact_match(ans_list[-1], row["answer"]):
                        correct_num += 1
                        correct = 1
                    else:
                        correct = 0
            else:
                response_str = response_str.strip()
                if len(response_str) > 0:
                    ans_list = self.extract_ans(response_str)
                    if len(ans_list) > 0 and (ans_list[-1] == row["answer"]):
                        correct_num += 1
                        correct = 1
                    else:
                        correct = 0
                else:
                    correct = 0
            if save_result_dir:
                result.append(response_str)
                score.append(correct)
        correct_ratio = 100 * correct_num / len(answers)

        if save_result_dir:
            test_df["model_output"] = result
            test_df["correctness"] = score
            test_df.to_csv(
                os.path.join(save_result_dir, f"{subject_name}_val.csv"),
                encoding="utf-8",
                index=False,
            )
        return correct_ratio

    def eval_subject_v2(
        self,
        subject_name,
        test_df,
        dev_df=None,
        few_shot=False,
        save_result_dir=None,
        cot=False,
    ):
        correct_num = 0
        if save_result_dir:
            result = []
            score = []
        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(
                subject_name, dev_df, cot=cot
            )
        else:
            few_shot_prompt = [
                {
                    "role": "system",
                    "content": f"你是一位專業的中文AI助理，以下是關於{subject_name}考試單選題，請選出正確的答案。",
                }
            ]
        answers = list(test_df["answer"])
        for row_index, row in tqdm(
            test_df.iterrows(), total=len(test_df), dynamic_ncols=True
        ):
            question = self.format_example(row, include_answer=False)
            full_prompt = few_shot_prompt + question
            if not few_shot:
                full_prompt[-1]["content"] = (
                    f"以下是關於{subject_name}考試單選題，請選出正確的答案。\n\n"
                    + full_prompt[-1]["content"]
                )
            response = None
            timeout_counter = 0
            if self.converter:  # convert to simplified chinese
                for idx, prompt in enumerate(full_prompt):
                    full_prompt[idx]["content"] = self.converter.convert(
                        prompt["content"]
                    )

            while response is None and timeout_counter <= 30:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=full_prompt,
                        temperature=0.0,
                        max_tokens=800,
                    )
                except openai.InternalServerError:
                    for idx, prompt in enumerate(full_prompt):
                        full_prompt[idx]["content"] = replace_invalid_brackets(
                            prompt["content"]
                        )
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=full_prompt,
                        temperature=0.3,
                        max_tokens=800,
                    )
                except Exception as msg:
                    if "timeout=600" in str(msg):
                        timeout_counter += 1
                    logging.error(msg)
                    sleep(5)
                    continue
            if response == None:
                response_str = ""
            else:
                response_str = response.choices[0].message.content

            response_str = response_str.strip()
            if len(response_str) > 0:
                ans_list = self.llm_parsing_ans(response_str)
                if len(ans_list) > 0 and (ans_list[-1] == row["answer"]):
                    correct_num += 1
                    correct = 1
                else:
                    correct = 0
            else:
                correct = 0

            if save_result_dir:
                result.append(response_str)
                score.append(correct)
        correct_ratio = 100 * correct_num / len(answers)

        if save_result_dir:
            test_df["model_output"] = result
            test_df["correctness"] = score
            test_df.to_csv(
                os.path.join(save_result_dir, f"{subject_name}_val.csv"),
                encoding="utf-8",
                index=False,
            )
        return correct_ratio
