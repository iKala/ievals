import re


def match_response_choices(response: str, converter=None):
    pattern = [
        r"correct answer is:\n\n\n([A-D]).",
        r"correct answer is:\n\n([A-D]).",
        r"correct answer is:\n([A-D]).",
        r"正確的答案應該是:.*?\b([A-D])\b",
        r"正确的答案应该是:.*?\b([A-D])\b",
        r"正確的選項應為:.*?\b([A-D])\b",
        r"正确的选项应为:.*?\b([A-D])\b",
        r"正確的答案是（([A-D])）",
        r"正确的答案是（([A-D])）",
        r"答案應該是:\s?選?項?\s?([A-D])",
        r"答案应该是:\s?选?项?\s?([A-D])",
        r"答案是:\s?選?項?\s?([A-D])",
        r"答案是:\s?选?项?\s?([A-D])",
        r"答案應為:\s?選?項?\s?([A-D])",
        r"答案应为:\s?选?项?\s?([A-D])",
        r"答案應該是:\s?選?項?\s?([A-D])",
        r"答案为:\s?选?项?\s?([A-D])",
        r"答案为：\s?选?项?\s?([A-D])",
        r"答案應為：\s?選?項?\s?([A-D])",
        r"答案应为：\s?选?项?\s?([A-D])",
        r"答案是：\s?选?项?\s?([A-D])",
        r"答案为\s?选?项?\s?([A-D])",
        r"答案选\s?选?项?\s?([A-D])",
        r"答案是\s?选?项?\s?([A-D])",
        r"答案是：\s?选?项?\s?([A-D])",
        r"答案为\s?选?项?\s?([A-D])",
        r"答案选\s?选?项?\s?([A-D])",
        r"答案是\s?选?项?\s?([A-D])",
        r"所以下列方程式的解是([A-D])",
        r"答案是\s?选?项?\s?([A-D])",
        r"答案为\s?选?项?\s?([A-D])",
        r"答案应为\s?选?项?\s?([A-D])",
        r"答案选\s?选?项?\s?([A-D])",
        r"答案是:\s?选?项?\s?([A-D])",
        r"答案应该是:\s?选?项?\s?([A-D])",
        r"正确的一项是\s?([A-D])",
        r"答案为:\s?选?项?\s?([A-D])",
        r"答案应为:\s?选?项?\s?([A-D])",
        r"答案:\s?选?项?\s?([A-D])",
        r"答案是：\s?选?项?\s?([A-D])",
        r"答案应该是：\s?选?项?\s?([A-D])",
        r"答案为：\s?选?项?\s?([A-D])",
        r"答案应为：\s?选?项?\s?([A-D])",
        r"答案：\s?选?项?\s?([A-D])",
        r"答案為:\s?([A-D])",
        r"答案应为：\s?([A-D])",
        r"答案為：\s?([A-D])",
        r"答案應該是:\s?([A-D])",
        r"正確答案為 \*\*([A-D])",
        r"正確答案為\(([A-D])\)",
        r"答案應為:\s?([A-D])",
        r"答案应为:\s?([A-D])",
        r"答案是 \*\*([A-D])",
        r"答案為 \*\*([A-D])",
        r"答案 ([A-D]) 正確",
        r"選項 ([A-D]) 正確",
        r"所以答案為([A-D])",
        r"答案：\(([A-D])\)",
        r"答案:\s?([A-D])",
        r"答案：\s?([A-D])",
        r"答案: ([A-D]) ",
        r"答案([A-D]) ",
        r"^選項([A-D])",
        r"^选项([A-D])",
        r"^選([A-D])",
        r"^选([A-D])",
        r"([A-D]). ",
        r"([A-D]).",
    ]
    ans_list = []
    if response[0] in ["A", "B", "C", "D"]:
        ans_list.append(response[0])
    for p in pattern:
        if converter is not None:  # for backward compatibility
            p = converter.convert(p)
        if len(ans_list) == 0:
            ans_list = re.findall(p, response)
        else:
            break
    return ans_list


def cot_answer_parser(response_str: str, llm_parser):
    """
    Experiment with LLM  parser for chain of thought reasoning based on the method original paper
    """
    prompt = (
        "You are an parser assistant, your task is to given the following context, parse out whether the context answers is A, B, C, D or unknown\n"
        + "Context:\n{}\n"
        + "Do not do any reasoning or answer anything else other than A, B, C, D or unknown"
    )
    answer = llm_parser(prompt.format(response_str))
    return [answer]
