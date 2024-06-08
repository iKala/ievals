import re
import os
import opencc
import logging
from tqdm import tqdm
from time import sleep
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
from .evaluator import Evaluator
from ..answer_parser import cot_match_response_choice
from ...helper import retry_with_exponential_backoff

vertexai.init(project=os.environ['GCP_PROJECT_NAME'], location="us-central1")

class Vertex_Evaluator(Evaluator):

    SAFETY_SETTINGS={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }

    def __init__(self, choices, k, api_key, model_name, switch_zh_hans=False):
        super(Vertex_Evaluator, self).__init__(choices, model_name, k)
        self.model = GenerativeModel(model_name)

        self.model_name = model_name
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
            prompt += tmp
        return prompt

    @retry_with_exponential_backoff
    def infer_with_backoff(self, prompt, max_tokens=1024, temperature=0.0, top_p=1, top_k=1):
        result = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": int(max_tokens),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k)
                },
                safety_settings=self.SAFETY_SETTINGS,
                stream=False
        ).candidates[0].content.parts[0].text
        res_info = {
            "input": prompt,
            "output": result,
            "num_input_tokens": self.model.count_tokens(prompt).total_tokens,
            "num_output_tokens": self.model.count_tokens(result).total_tokens,
            "logprobs": []  # NOTE: currently the Gemini API does not provide logprobs
        }
        return result, res_info


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
            text = []
            prev_role = ""
            for prompt in full_prompt:
                if prompt["role"] == "system":
                    text.append(prompt["content"] + "\n")
                elif prompt["role"] == "user":
                    if prev_role == "system":
                        text[-1] += "問題: " + prompt["content"] + "\n"
                    else:
                        text.append("問題: " + prompt["content"] + "\n")
                elif prompt["role"] == "assistant":
                    text.append(prompt["content"] + "\n")
                prev_role = prompt["role"]
            if self.converter:
                text = [self.converter.convert(seg) for seg in text]

            response_str, response = self.infer_with_backoff(text, max_tokens=512, temperature=0)

            if cot:
                ans_list = cot_match_response_choice(response_str,
                            is_simplified= self.switch_zh_hans)

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

