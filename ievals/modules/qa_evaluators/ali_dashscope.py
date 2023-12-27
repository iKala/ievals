import os
import re
import logging
from time import sleep
import opencc
from tqdm import tqdm
from http import HTTPStatus
try:
    import dashscope
    from dashscope import Generation
except ImportError as e:
    logging.error("dashscope API not supported, ignore this if you aren't using dashscope")
from .evaluator import Evaluator


class DashScope_Evaluator(Evaluator):
    """
        Completion endpoint for instruction based model
        qwen models
    """
    def __init__(self, choices, k, api_key, model_name, switch_zh_hans=False):
        super(DashScope_Evaluator, self).__init__(choices, model_name, k)
        dashscope.api_key = api_key
        assert model_name in set(Generation.Models.__dict__.values())
        self.model_name = model_name
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
            response = None
            timeout_counter = 0
            if self.converter:
                converted = []
                for p in full_prompt:
                    p['content'] = self.converter.convert(p['content'])
                    converted.append(p)
                full_prompt = converted

            text = ""
            for prompt in full_prompt:
                text += prompt["content"] + "\n"

            while response is None and timeout_counter <= 30:
                try:
                    response = Generation.call(model=self.model_name,
                            prompt=text)
                except Exception as msg:
                    if "timeout=600" in str(msg):
                        timeout_counter += 1
                    logging.error(msg)
                    sleep(5)
                    continue

            if response.status_code == HTTPStatus.OK:
                response_str = response.output.text
            else:
                response_str = ""

            if cot:
                ans_list = re.findall(r"答案是(.+?)。", response_str)
                if self.converter: # simplified chinese
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

if __name__ == "__main__":
    evaluator = DashScope_Evaluator([], 0, '', 'qwen-max')
    ans = evaluator.extract_ans("""C
\( \lim_{x \to \infty} (\sqrt{x^2 + 11x} - x) = ? \)
A. \( 37 \)
B. \( \frac{111}{5} \)
C. \( \frac{11}{2} \)
D. \( \frac{111}{4} \)
答案：
C
如果 \( u \) 和 \( v \) 正交，則 \( u \cdot v = 0 \)。\( S^\perp \) 是 \( \mathbb{R}^n \) 中所有與 \( S \) 中每個向量都正交的向量集合。考慮集合
\( S = \left\{ \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} \in \mathbb{R}^3 : x_1 - x_2 + x_3 = 0 \right\} \)。選擇以下正確的陳述。
A. \( S \) 是 \( \mathbb{R}^3 \) 的一個子空間且 \( \text{dim}S = 1 \)
B. \( \begin{bmatrix} 1 \\ -1 \\ 1 \end{bmatrix} \in S \)
C. 設 \( \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} = w + z \) 使得 \( w \in S \) 並且 \( z \in S^\perp \)，則 \( z = \begin{bmatrix} 1/3 \\ -1/3 \\ 1/3 \end{bmatrix} \)。
D. \( \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix} \in S^\perp \)
答案：
C
\( f(x) = \left\{
\begin{array}{ll}
\frac{x^4 - 1}{x^2 - 1}, & x \neq \pm1 \\
a, & x = 1 \\
b, & x = -1
\end{array}
\right. \)，若 \( f(x) \) 在 \( x=\pm1 \) 處連續，則 \( \frac{a}{b} \) 的值為何？
A. 2
B. 4
C. 3
D. 1
答案：
D
行使得 \( A = \begin{bmatrix} 1 & -3 & 4 & -2 & 5 \\ 2 & -6 & 9 & -1 & 8 \\ 2 & -6 & 9 & -1 & 9 \\ -1 & 3 & -4 & 2 & -5 \end{bmatrix} \), 則下列選項中何者為矩陣 A 的秩 (rank)？
A. 1
B. 2
C. 3
D. 4
答案：
C
2已知\(X\)和\(Y\)的聯合機率密度函數 (Joint probability density function) 為 \[ f_{X,Y}(x,y) = \begin{cases} 2, & 0 \leq y \leq x \leq 1 \\ 0, & 其他 \end{cases} \] 下列何者錯誤？
A. 在 0 ≤ x ≤1 ， fX( x) = 2 x
B. 條件機率密度函數 \[ f_{Y|X}(y|x) = \begin{cases} \frac{1}{x}, & 0 \leq y \leq x \leq 1 \\ 0, & 其他 \end{cases} \]
C. 條件機率密度函數 \[ f_{X|Y}(x|y) = \begin{cases} \frac{1}{y}, & 0 \leq y \leq x \leq 1 \\ 0, & 其他 \end{cases} \]
D. 在 0 ≤ y ≤1 ， fY( y) = 2(1 - y)
答案：
D""")
    print(ans)