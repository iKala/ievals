# This code is modified from C-Eval Project: https://github.com/SJTU-LIT/ceval

import os
import re
from tqdm import tqdm
import opencc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from transformers import GenerationConfig
from .evaluator import Evaluator


class Mixtral_Evaluator(Evaluator):
    def __init__(
        self,
        choices,
        k,
        model_name,
        device="cuda",
        temperature=0.2,
        load_in_4bit=False,
        use_flash_attention_2=False,
        verbose=False,
        switch_zh_hans=False,
    ):
        super(Mixtral_Evaluator, self).__init__(choices, model_name, k)
        load_type = torch.float16
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        self.load_in_4bit = load_in_4bit
        self.use_flash_attention_2 = use_flash_attention_2
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name, legacy=True)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_compute_dtype=load_type,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config if self.load_in_4bit else None,
            torch_dtype=load_type,
            low_cpu_mem_usage=True,
            device_map="auto",
            attn_implementation="flash_attention_2"
            if self.use_flash_attention_2
            else "sdpa",
        )
        self.generation_config = GenerationConfig(
            temperature=temperature,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.1,
            max_length=4096,
        )

        self.sA_id = self.tokenizer.encode("A", add_special_tokens=False)[0]
        self.sB_id = self.tokenizer.encode("B", add_special_tokens=False)[0]
        self.sC_id = self.tokenizer.encode("C", add_special_tokens=False)[0]
        self.sD_id = self.tokenizer.encode("D", add_special_tokens=False)[0]
        self.A_id = self.tokenizer.encode("：A")[-1]
        self.B_id = self.tokenizer.encode("：B")[-1]
        self.C_id = self.tokenizer.encode("：C")[-1]
        self.D_id = self.tokenizer.encode("：D")[-1]
        self.converter = None
        if switch_zh_hans:
            self.converter = opencc.OpenCC("t2s.json")

    def eval_subject(
        self,
        subject_name,
        test_df,
        dev_df=None,
        few_shot=False,
        cot=False,
        save_result_dir=None,
        with_prompt=False,
        constrained_decoding=False,
        do_test=False,
    ):
        all_answers = {}
        if constrained_decoding is True:
            self.generation_config.output_scores = True
            self.generation_config.return_dict_in_generate = True
            self.generation_config.max_new_tokens = 1
            self.generation_config.top_p = 1.0
            self.generation_config.top_k = 0

        correct_num = 0
        if save_result_dir:
            result = []
            score = []
        if few_shot:
            if with_prompt:
                history = self.generate_mixtral_inst_few_shot_prompt(
                    subject_name, dev_df, cot=cot
                )
            else:
                history = self.generate_mixtral_few_shot_prompt(
                    subject_name, dev_df, cot=cot
                )
        else:
            history = ""
        answers = ["NA"] * len(test_df) if do_test is True else list(test_df["answer"])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(
                row, include_answer=False, cot=cot, with_prompt=with_prompt
            )
            instruction = question
            if with_prompt:
                prompt_template = "[INST] {instruction} [/INST]"

                instruction = prompt_template.format_map({"instruction": instruction})
            instruction = history + instruction
            inputs = self.tokenizer(instruction, return_tensors="pt")
            generation_output = self.model.generate(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                generation_config=self.generation_config,
                max_length=6400,
            )

            _, length = inputs.input_ids.shape
            if constrained_decoding is True:
                logits = generation_output.scores[0][0]

                logits = logits.float().cpu().detach()
                choices1_logits = logits[
                    [self.sA_id, self.sB_id, self.sC_id, self.sD_id]
                ]
                choices2_logits = logits[[self.A_id, self.B_id, self.C_id, self.D_id]]
                choicesAll_logits = (choices1_logits + choices2_logits).numpy()
                assert not (
                    np.any(np.isinf(choicesAll_logits))
                    or np.any(np.isnan(choicesAll_logits))
                )
                ans = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choicesAll_logits)]
                response_str = self.tokenizer.decode([logits.argmax(-1).item()])
            else:
                response_str = self.tokenizer.decode(
                    generation_output[0, length:], skip_special_tokens=True
                )
                ans = self.extract_answer(response_str)
            if ans == answers[row_index]:
                correct_num += 1
                correct = 1
            else:
                correct = 0
            if self.verbose is True:
                print(f"\n======={str(row_index)}=======")
                print(f"question: {question}\n")
                print(f"response: {response_str}\n")
                print(f"extracted answer: {ans}")
                print(f"ground truth: {answers[row_index]} \n")
            if save_result_dir:
                result.append(response_str)
                score.append(correct)

            all_answers[str(row_index)] = ans

        correct_ratio = 100 * correct_num / len(answers)

        if save_result_dir:
            test_df["model_output"] = result
            test_df["correctness"] = score
            test_df.to_csv(os.path.join(save_result_dir, f"{subject_name}_val.csv"))

        return correct_ratio

    def format_example(self, line, include_answer=True, cot=False, with_prompt=False):
        example = "問題：" + line["question"] + "\n\n"
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'

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
                example += "\n答案：" + line["answer"] + "\n\n"
        else:
            if with_prompt is False:
                if cot:
                    example += "\n答案：讓我們一步一步思考，\n1."
                else:
                    example += "\n答案："
            else:
                if cot:
                    example += "\n答案是什麼？讓我們一步一步思考，\n1."
                else:
                    example += "\n答案："
        return example

    def generate_mixtral_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = f"以下是關於{subject}考試單選題，請選出正確的答案。\n\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += self.format_example(
                dev_df.iloc[i, :], include_answer=True, cot=cot
            )

        return prompt

    def generate_mixtral_inst_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = f"以下是關於{subject}考試單選題，請選出正確的答案\n\n"
        prompt_template = "[INST] {instruction} [/INST]好的，我會結合{subject}相關知識回答"
        if self.converter:
            prompt = self.converter.convert(prompt)
            prompt_template = self.converter.convert(prompt_template)
        prompt = prompt_template.format_map({"instruction": prompt, "subject": subject})
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            line = dev_df.iloc[i, :]
            q = line["question"]
            if self.converter:
                q = self.converter.convert(q)
            for choice in self.choices:
                q += f'\n{choice}. {line[f"{choice}"]}'

            a = line["answer"]
            if self.converter:
                a = self.converter.convert(a)
            prompt += "[INST] " + q + "\n答案： [/INST]" + a + "\n"
        return prompt
