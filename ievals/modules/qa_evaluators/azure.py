import os
import re
import logging
from tqdm import tqdm
from time import sleep
from openai import AzureOpenAI
from .evaluator import Evaluator
import opencc



class Azure_Evaluator(Evaluator):
    def __init__(self, choices, k, api_key, model_name, switch_zh_hans=False):
        super(Azure_Evaluator, self).__init__(choices, model_name, k)
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=os.getenv("AZURE_OPENAI_VERSION", "2023-07-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.converter = None
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
                return [{"role": "user", "content": example}, {"role": "assistant", "content": content}]
            else:
                return [{"role": "user", "content": example}, {"role": "assistant", "content": line["answer"]}]
        else:
            return [
                {"role": "user", "content": example},
            ]

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = [{"role": "system", "content": f"你是一位專業的中文AI助理，以下是關於{subject}考試單選題，請選出正確的答案。"}]
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            tmp = self.format_example(dev_df.iloc[i, :], include_answer=True, cot=cot)
            if i == 0:
                tmp[0]["content"] = f"以下是關於{subject}考試單選題，請選出正確的答案。\n\n" + tmp[0]["content"]
                if self.converter:
                    tmp[0]["content"] = self.converter.convert(tmp[0]["content"])
            if len(prompt) + len(tmp) > 4096:
                break
            else:
                prompt += tmp

        return prompt

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False, save_result_dir=None, cot=False):
        correct_num = 0
        if save_result_dir:
            result = []
            score = []
        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)
        else:
            few_shot_prompt = [{"role": "system", "content": f"你是一位專業的中文AI助理，以下是關於{subject_name}考試單選題，請選出正確的答案。"}]
        answers = list(test_df["answer"])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df), dynamic_ncols=True):
            question = self.format_example(row, include_answer=False)
            full_prompt = few_shot_prompt + question
            if not few_shot:
                full_prompt[-1]["content"] = f"以下是關於{subject_name}考試單選題，請選出正確的答案。\n\n" + full_prompt[-1]["content"]
            
            if self.converter:
                converted = []
                for p in full_prompt:
                    p['content'] = self.converter.convert(p['content'])
                    converted.append(p)
                full_prompt = converted

            response = None
            timeout_counter = 0

            while response is None and timeout_counter <= 30:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name, messages=full_prompt, temperature=0.0
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
                ans_list = re.findall(r"答案是(.+?)。", response_str)
                if self.converter:
                    if len(ans_list) == 0:
                        ans_list = re.findall(r"答案为(.+?)。", response_str)
                    if len(ans_list) == 0:
                        ans_list = re.findall(r"选项(.+?)是正确的。", response_str)
                else:
                    if len(ans_list) == 0:
                        ans_list = re.findall(r"答案為(.+?)。", response_str)
                    if len(ans_list) == 0:
                        ans_list = re.findall(r"選項(.+?)是正確的。", response_str)

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
            test_df.to_csv(os.path.join(save_result_dir, f"{subject_name}_val.csv"), encoding="utf-8", index=False)
        return correct_ratio

    def extract_ans(self, response_str):
        pattern = [
            r"([A-D]). ",
            r"([A-D]).",
            r"^選([A-D])",
            r"^選項([A-D])",
            r"^选([A-D])",
            r"^选项([A-D])",
            r"答案是\s?選?項?\s?([A-D])",
            r"答案為\s?選?項?\s?([A-D])",
            r"答案應為\s?選?項?\s?([A-D])",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案選\s?選?項?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?選?項?\s?([A-D])",
            r"答案應該是:\s?選?項?\s?([A-D])",
            r"答案应该是:\s?选?项?\s?([A-D])",
            r"正確的一項是\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"答案為:\s?選?項?\s?([A-D])",
            r"答案應為:\s?選?項?\s?([A-D])",
            r"答案:\s?選?項?\s?([A-D])",
            r"答案是：\s?選?項?\s?([A-D])",
            r"答案應該是：\s?選?項?\s?([A-D])",
            r"答案為：\s?選?項?\s?([A-D])",
            r"答案應為：\s?選?項?\s?([A-D])",
            r"答案：\s?選?項?\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：\s?选?项?\s?([A-D])",
            r"答案应该是：\s?选?项?\s?([A-D])",
            r"答案为：\s?选?项?\s?([A-D])",
            r"答案应为：\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
        ]
        ans_list = []
        if response_str[0] in ["A", "B", "C", "D"]:
            ans_list.append(response_str[0])
        for p in pattern:
            if self.converter:
                p = self.converter.convert(p)
            if len(ans_list) == 0:
                ans_list = re.findall(p, response_str)
            else:
                break
        return ans_list


