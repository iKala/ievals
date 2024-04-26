import os
import re
from time import sleep
from tqdm import tqdm
import opencc
from transformers import AutoModelForCausalLM, AutoTokenizer
from .evaluator import Evaluator
from ..answer_parser import cot_match_response_choice

class HF_Chat_Evaluator(Evaluator):
    def __init__(self, choices, k, api_key, model_name, switch_zh_hans=False):
        super(HF_Chat_Evaluator, self).__init__(choices, model_name, k)
        self.converter = None
        self.switch_zh_hans = switch_zh_hans
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
        if include_answer:
            if cot:
                answer = (
                    "讓我們一步一步思考，\n"
                    + line["explanation"]
                    + "\n所以答案是"
                    + line["answer"]
                    + "。\n\n"
                )
            else:
                answer = "\n答案：" + line["answer"] + "\n\n"
            m = (example, answer)
            return m
        return example

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        messages = []
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            tmp = self.format_example(dev_df.iloc[i, :], cot=cot)
            if i == 0:
                if isinstance(tmp, tuple):
                    tmp = (f"以下是關於{subject}考試單選題，請選出正確的答案。\n\n" + tmp[0], tmp[1])
                else:  # should be string
                    tmp = f"以下是關於{subject}考試單選題，請選出正確的答案。\n\n" + tmp
            messages.append(tmp)

        return messages

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
            history = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)
        else:
            history = []
        answers = list(test_df["answer"])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False, cot=cot)
            response = None
            timeout_counter = 0
            text = ""

            if self.converter:
                question = self.converter.convert(question)
                history = [self.converter.convert(hist) for hist in history]
            # better to check for history accuracy
            while response is None and timeout_counter <= 30:
                try:
                    response, _ = self.model.chat(
                        self.tokenizer, question, history=history
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
            if cot:  # simplified chinese
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

