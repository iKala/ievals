# iKala's Evaluator for Large Language Models

iEvals is a framework for evaluating chinese large language models (LLMs), especially performance in traditional chinese domain. Our goal was to provide an easy to setup and fast evaluation library for guiding the performance/use on existing chinese LLMs.

Currently, we only supports evaluation for TMMLU+ however in the future we are exploring more domain, ie knowledge extensive dataset (CMMLU, C-Eval) as well as context retrieval and multi-conversation dataset.


# Usage

```bash
ieval <model name> <series: optional> --top_k <numbers of incontext examples>
```

## OpenAI

```bash
ieval gpt-3.5-turbo-0613 --api_key "<Your OpenAI platform Key>" --top_k 5
```

## Gemini Pro

```bash
ieval gemini-pro  --api_key "<Your API Key from https://ai.google.dev/>" --top_k 5
```

Currently we do not support models from vertex AI yet. So PaLM (bison) series are not supported

## Anthropic (instant, v1.3, v2.0)

```bash
ieval claude-instant-1  --api_key "<Anthropic API keys>"
```

## Azure OpenAI Model

```bash
export AZURE_OPENAI_ENDPOINT="https://XXXX.azure.com/"
ieval <your azure model name>  azure --api_key "<Your API Key>" --top_k 5
```

We haven't experimented with instruction based model from azure yet, so for instruction based models, you will have to fallback to openai's models


# Coming soon

- Chain of Thought (CoT) with few shot

- Arxiv paper : detailed analysis on model interior and exterior relations

- More tasks

# Citation

```
@article{ikala2023eval,
  title={An Improved Traditional Chinese Evaluation Suite for Foundation Model},
  author={Tam, Zhi-Rui and Pai, Ya-Ting},
  journal={arXiv},
  year={2023}
}
```
