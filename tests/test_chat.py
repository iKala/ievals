import os
from anthropic import Anthropic
from ievals.modules.answer_parser import cot_answer_parser

def test_cot_llm_parser():
    response_str = "答案：D. 以上皆是\n解釋：\n組織中常見的溝通媒介包括信件、會議、電子郵件等。這些媒介可以幫助組織內部的人"
    client = Anthropic(
        # This is the default and can be omitted
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    def llm_parser(response_str):
        message = client.messages.create(
            max_tokens=4096,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": response_str
                        }
                    ]
                }
            ],
            model="claude-3-haiku-20240307",
        )
        return message.content[0].text

    output  = cot_answer_parser(response_str, llm_parser)
    assert len(output) == 1
    assert output[0] == 'D'

test_cot_llm_parser()