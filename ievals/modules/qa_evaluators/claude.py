import re
import os
import logging
from time import sleep
import opencc
import anthropic
from tqdm import tqdm
from .evaluator import Evaluator


class Claude_Evaluator(Evaluator):
    def __init__(self, choices, k, api_key, model_name, switch_zh_hans=False):
        super(Claude_Evaluator, self).__init__(choices, model_name, k)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        self.converter = None
        self.switch_zh_hans = switch_zh_hans
        if switch_zh_hans:
            self.converter = opencc.OpenCC("t2s.json")

        self.change_to_new_model = None
        if (
            "opus" in self.model_name
            or "sonnet" in self.model_name
            or "haiku" in self.model_name
        ):
            self.change_to_new_model = True

    def format_example(self, line, include_answer=True, cot=False):
        example = line["question"]
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'

        example += "\n答案："
        if include_answer:
            if cot:
                ans = line["answer"]
                content = "讓我們一步一步思考，\n" + line["explanation"] + f"\n所以答案是{ans}。"
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
                "content": f"你是一位專業的中文AI助理，以下是關於{subject}考試單選題，請直接選出正確的答案。",
            }
        ]
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            tmp = self.format_example(dev_df.iloc[i, :], include_answer=True, cot=cot)
            if i == 0:
                tmp[0]["content"] = (
                    f"以下是關於{subject}考試單選題，請直接選出正確的答案。\n\n" + tmp[0]["content"]
                )
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
                    "content": f"你是一位專業的中文AI助理，以下是關於{subject_name}考試單選題，請直接選出正確的答案。",
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
                    f"以下是關於{subject_name}考試單選題，請直接選出正確的答案。\n\n"
                    + full_prompt[-1]["content"]
                )
            response = None
            timeout_counter = 0
            text = ""
            for prompt in full_prompt:
                if prompt["role"] == "system":
                    text += anthropic.HUMAN_PROMPT + " " + prompt["content"]
                elif prompt["role"] == "user":
                    text += anthropic.HUMAN_PROMPT + " " + prompt["content"]
                elif prompt["role"] == "assistant":
                    text += anthropic.AI_PROMPT + " " + prompt["content"]
            text += anthropic.AI_PROMPT
            if self.converter:
                text = self.converter.convert(text)

            while response is None and timeout_counter <= 30:
                try:
                    if self.change_to_new_model:
                        response = self.client.messages.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": text,
                                }
                            ],
                            model=self.model_name,
                            max_tokens=1200 if cot else 400,
                        )
                    else:
                        response = self.client.completions.create(
                            prompt=text,
                            stop_sequences=[anthropic.HUMAN_PROMPT],
                            model=self.model_name,
                            temperature=0.1,
                            max_tokens_to_sample=1200 if cot else 400,
                        )
                except Exception as msg:
                    if "timeout=600" in str(msg):
                        timeout_counter += 1
                    logging.error(msg)
                    sleep(5)
                    continue
            if response == None:
                response_str = ""
            elif self.change_to_new_model:
                response_str = response.content[0].text
            else:
                response_str = response.completion

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
                if few_shot:
                    if len(response_str) > 0:
                        if self.exact_match(response_str, row["answer"]):
                            correct_num += 1
                            correct = 1
                        else:
                            ans_list = self.extract_ans(response_str)
                            if len(ans_list) > 0 and (ans_list[-1] == row["answer"]):
                                correct_num += 1
                                correct = 1
                            else:
                                correct = 0
                    else:
                        correct = 0
                else:
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
