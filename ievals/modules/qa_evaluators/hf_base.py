"""
    Note due to the nature of logits based inference (select the token for A, B, C, D)
    
    CoT in theory shouln't work because the response should not start to inference A-D tokens
    but the thought process. 
    If you want to use chain of thought, please use the hf_chat.py process    
"""
import os
import re
import logging
import opencc
import torch
import pandas as pd
import numpy as np
from time import sleep
from tqdm import tqdm
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from .evaluator import Evaluator


class Qwen_Evaluator(Evaluator):
    def __init__(self, choices, k, api_key, model_name, switch_zh_hans=False):
        super(Qwen_Evaluator, self).__init__(choices, model_name, k)
        self.converter = None
        if switch_zh_hans:
            self.converter = opencc.OpenCC("t2s.json")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            pad_token="<|extra_0|>",
            eos_token="<|endoftext|>",
            padding_side="left",
            trust_remote_code=True,
            use_auth_token=api_key,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            pad_token_id=self.tokenizer.pad_token_id,
            trust_remote_code=True,
            use_auth_token=api_key,
        ).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(
            model_name,
            pad_token_id=self.tokenizer.pad_token_id,
            trust_remote_code=True,
            token=api_key,
        )

    def format_example(self, line, include_answer=True, cot=False):
        example = "問題：" + line["question"]
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'

        example += "\n答案："

        if include_answer:
            if cot:
                example += (
                    "讓我們一步一步思考，\n"
                    + line["explanation"]
                    + "\n所以答案是"
                    + line["answer"]
                    + "。\n\n"
                )
            else:
                example += "\n答案：" + line["answer"] + "\n\n"
        else:
            if cot:
                example += "\n答案：讓我們一步一步思考，\n"
            else:
                example += "\n答案："
        return example

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = ""
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            tmp = self.format_example(dev_df.iloc[i, :], include_answer=True, cot=cot)

            if i == 0:
                tmp = f"以下是關於{subject}考試單選題，請選出正確的答案。\n\n" + tmp
            prompt += tmp
        return prompt

    def get_logits(self, inputs):
        input_ids = self.tokenizer(inputs, padding="longest")["input_ids"]
        input_ids = torch.tensor(input_ids, device=self.model.device)
        tokens = {"input_ids": input_ids}
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        outputs = self.model(input_ids, attention_mask=attention_mask)["logits"]
        logits = outputs[:, -1, :]
        log_probs = torch.nn.functional.softmax(logits, dim=-1)
        return log_probs, {"tokens": tokens}

    @torch.no_grad()
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
            total_scores = []
            result = []
            score = []

        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(
                subject_name, dev_df, cot=cot
            )
        else:
            few_shot_prompt = ""

        all_probs = {"prob_A": [], "prob_B": [], "prob_C": [], "prob_D": []}

        choices_ids = (
            torch.tensor(
                self.tokenizer("A")["input_ids"]
                + self.tokenizer("B")["input_ids"]
                + self.tokenizer("C")["input_ids"]
                + self.tokenizer("D")["input_ids"]
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        answers = list(test_df["answer"])
        for row_index, row in tqdm(
            test_df.iterrows(), total=len(test_df), dynamic_ncols=True
        ):
            question = self.format_example(row, include_answer=False, cot=cot)
            text = ""
            answer_list = []
            full_prompt = few_shot_prompt + question
            if "answer" in row:
                answer_list.append(row["answer"])
            input_info = None
            timeout_counter = 0

            if self.converter:
                text = self.converter.convert(text)

            while input_info is None and timeout_counter <= 30:
                try:
                    logits, input_info = self.get_logits([full_prompt])
                    softval = logits.gather(
                        1, choices_ids.expand(logits.size(0), -1)
                    ).softmax(1)
                    if softval.dtype in {torch.bfloat16, torch.float16}:
                        softval = softval.to(dtype=torch.float32)
                    probs = softval.detach().cpu().numpy()
                except Exception as msg:
                    if "timeout=600" in str(msg):
                        timeout_counter += 1
                    logging.error(msg)
                    sleep(5)
                    continue

            for i in range(len(probs)):
                for j, choice in enumerate(self.choices):
                    all_probs[f"prob_{choice}"].append(probs[i][j])
                    pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs[i])]

                    if answer_list != []:
                        correct = 1 if pred == answer_list[i] else 0

            if save_result_dir:
                result.append(pred)
                score.append(correct)
        correct_ratio = 100 * sum(score) / len(score)

        if save_result_dir:
            print(len(result), len(score))
            print(result[:5], score[:5])
            test_df["model_output"] = result
            test_df["correctness"] = score
            test_df.to_csv(
                os.path.join(save_result_dir, f"{subject_name}_val.csv"),
                encoding="utf-8",
                index=False,
            )
        return correct_ratio
