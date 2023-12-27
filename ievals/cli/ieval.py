"""
    CLI for all models
    Support mode:
        if tgi service was used you must pass in IP and hostname

            if the service was found in model_config.csv you could skip providing the 4 tokens (user, assistant, system, eos)
            else you need to pass in the four token in args

"""
import os
import logging
import argparse
import pandas as pd
from datasets import load_dataset
from ievals.modules.qa_evaluators.tgi import TGI_Evaluator
from ievals.modules.qa_evaluators.gemini import Gemini_Evaluator
from ievals.modules.qa_evaluators.claude import Claude_Evaluator
from ievals.modules.qa_evaluators.azure import Azure_Evaluator
from ievals.modules.qa_evaluators.oai_complete import GPT_Evaluator
from ievals.modules.qa_evaluators.chatgpt import ChatGPT_Evaluator
try:
    from ievals.modules.qa_evaluators.hf_chat import HF_Chat_Evaluator
    from ievals.modules.qa_evaluators.hf_base import Qwen_Evaluator # we only use this for qwen base model
except ImportError as e:
    logging.info("huggingface and qwen models are not supported due to "+str(e))
from ievals.modules.qa_evaluators.ali_dashscope import DashScope_Evaluator
from ievals.exp_executer import run_exp

def get_model_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    up_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    df = pd.read_csv(os.path.join(up_dir, "model_config.csv"))
    df.fillna("", inplace=True)
    valid_model_names = df["model_name"].tolist()
    return valid_model_names, df

def get_tgi_prompt_config(model_name):
    valid_model_names, df = get_model_config()
    if model_name not in valid_model_names:
        return None, None
    prompt_config = df[df["model_name"] == model_name].iloc[0]
    prompt_config.pop('model_name')
    return prompt_config

def get_evaluator(model_name, series=""):
    if len(series):
        if series == 'azure':
            return Azure_Evaluator
        elif series == 'openai_chat':
            return ChatGPT_Evaluator
        elif series == 'openai_complete':
            return GPT_Evaluator
        elif series == 'gemini':
            return Gemini_Evaluator
        elif series == 'hf_chat': # implement the chat function
            return HF_Chat_Evaluator
        elif series == 'tgi': # implement the chat function
            return TGI_Evaluator

    l_model_name = model_name.lower()

    if 'gemini' in model_name:
        return Gemini_Evaluator
    if 'gpt-' in model_name: 
        # its possible to match gpt-3.5-instruct, 
        # but we don't really want to sacrifice more fixed params for that
        return ChatGPT_Evaluator
    elif 'claude' in model_name:
        return Claude_Evaluator
    elif 'Qwen' in model_name and 'chat' in l_model_name:
        return HF_Chat_Evaluator
    elif 'Qwen' in model_name and 'base' in l_model_name:
        return Qwen_Evaluator
    elif 'qwen' in model_name:
        return DashScope_Evaluator
    return TGI_Evaluator


def get_parser():
    parser = argparse.ArgumentParser(description="Run TMMLU+ evals")
    parser.add_argument("model", type=str, help="Name of the eval model")
    parser.add_argument("--series", type=str, default="")
    parser.add_argument("--dataset", type=str, default="ikala/tmmluplus")
    parser.add_argument("--choices", type=str, default="A,B,C,D")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--switch_zh_hans", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ip_addr", type=str, default="", help="IP:PORT for text-generation-inference server")

    parser.add_argument("--sys_token", type=str, default="", help="system prompt token")
    parser.add_argument("--usr_token", type=str, default="", help="user starting token")
    parser.add_argument("--ast_token", type=str, default="", help="assistant starting token")
    parser.add_argument("--eos_token", type=str, default="", help="end-of-sentence token usually its <|endoftext|> or </s>, but you have to verify from hf model tokenizer.json")


    parser.add_argument("--hf_cache", type=str, default="", help="huggingface cache")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    model_name = args.model
    if model_name == 'supported':
        valid_model_names, _ = get_model_config()
        print(valid_model_names)
        exit(0)

    valid_choices = args.choices.split(',')
    eval_cls = get_evaluator(model_name, args.series)
    if 'TGI' in str(eval_cls):
        if len(args.usr_token):
            prompt_config = {
                'systemMessageToken': args.sys_token,
                'userMessageToken': args.usr_token,
                'messageEndToken': args.eos_token,
                'assistantMessageToken': args.ast_token,
            }
        else:
            prompt_config = get_tgi_prompt_config(model_name)
        eval_ins = eval_cls(
            choices=valid_choices,
            k=args.top_k,
            ip_addr=args.ip_addr,
            model_name=model_name,
            switch_zh_hans=args.switch_zh_hans,
            **prompt_config
        )
    else:
        eval_ins = eval_cls(
            choices=valid_choices,
            k=args.top_k,
            api_key=args.api_key,
            model_name=model_name,
            switch_zh_hans=args.switch_zh_hans
        )
    postfix = model_name.split('/')[-1]
    if args.top_k > 0:
        postfix += f"_top_{args.top_k}"

    cache_path = None
    if args.cache:
        cache_path = '.cache'
        if args.top_k > 0:
            cache_path += f"_top_{args.top_k}"

    run_exp(eval_ins, model_name, args.dataset, few_shot= args.top_k > 0,
            cache_path='.cache', postfix_name=postfix)

if __name__ == "__main__":
    main()