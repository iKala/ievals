import os
import re
from time import sleep
from tqdm import tqdm
import opencc
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from .evaluator import Evaluator


class HF_Chat_Evaluator(Evaluator):
    def __init__(self, choices, k, api_key, model_name, switch_zh_hans=False):
        super(HF_Chat_Evaluator, self).__init__(choices, model_name, k)
        self.converter = None
        if switch_zh_hans:
            self.converter = opencc.OpenCC("t2s.json")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_auth_token=api_key
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=api_key,
        ).eval()
        self.model.generation_config.do_sample = False
        self.model.generation_config.repetition_penalty = 1.0

    def format_example(self, line, include_answer=True, cot=False):
        example = "問題：" + line["question"] + "\n\n"
        for choice in self.choices:
            example += f'{choice}. {line[f"{choice}"]}\n'

        example += "\n答案："
        history = []
        if include_answer:
            if cot:
                history.append(line["explaination"])
                example += "\n答案：" + line["answer"] + "\n\n"
            else:
                example += "\n答案：" + line["answer"] + "\n\n"
        else:
            example += "\n答案："
        if len(history) == 0:
            history = None
        return example, history

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = ""
        history_prompt = []
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            tmp, history = self.format_example(dev_df.iloc[i, :], cot=cot)
            if i == 0:
                tmp = f"以下是關於{subject}考試單選題，請選出正確的答案。\n\n" + tmp
            prompt += tmp
            if cot and len(history) > 0:
                history_prompt.extend(history)
        if not cot:
            history_prompt = None
        return prompt, history_prompt

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

        q_history = None
        if few_shot:
            few_shot_prompt, _ = self.generate_few_shot_prompt(
                subject_name, dev_df, cot=cot
            )
        else:
            few_shot_prompt = ""
        answers = list(test_df["answer"])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question, q_history = self.format_example(row, include_answer=False)
            full_prompt = few_shot_prompt + question
            response = None
            timeout_counter = 0
            text = ""

            if self.converter:
                full_prompt = self.converter.convert(full_prompt)

            while response is None and timeout_counter <= 30:
                try:
                    response, _ = self.model.chat(
                        self.tokenizer, full_prompt, history=q_history
                    )
                except Exception as msg:
                    if "timeout=600" in str(msg):
                        timeout_counter += 1
                    print(msg)
                    sleep(5)
                    continue

            if response == None:
                response_str = ""
            else:
                response_str = response
            if cot: # simplified chinese
                ans_list = re.findall(r"答案是(.+?)。", response_str)
                if self.converter: # simplified chinese
                    if len(ans_list) == 0:
                        ans_list = re.findall(r"答案为(.+?)", response_str)
                    if len(ans_list) == 0:
                        ans_list = re.findall(r"选项(.+?)是正确的", response_str)
                    if len(ans_list) == 0:
                        ans_list = re.findall(r"因此，选项(.+?)", response_str)
                else:
                    if len(ans_list) == 0:
                        ans_list = re.findall(r"答案為(.+?)", response_str)
                    if len(ans_list) == 0:
                        ans_list = re.findall(r"選項(.+?)是正確的", response_str)
                    if len(ans_list) == 0:
                        ans_list = re.findall(r"因此，選項(.+?)", response_str)

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
            r"答案選\s?選?項?\s?([A-D])",
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
