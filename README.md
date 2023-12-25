# iEvals : iKala's Evaluator for Large Language Models

<p align="center"> <img src="resources/ieval_cover.png" style="width: 50%; max-width: 400px" id="title-icon">       </p>


iEvals is a framework for evaluating chinese large language models (LLMs), especially performance in traditional chinese domain. Our goal was to provide an easy to setup and fast evaluation library for guiding the performance/use on existing chinese LLMs.

Currently, we only support evaluation for [TMMLU+](https://huggingface.co/datasets/ikala/tmmluplus), however in the future we are exploring more domain, ie knowledge extensive dataset (CMMLU, C-Eval) as well as context retrieval and multi-conversation dataset.


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


## Text generation inference

In order to reduce download friction, we recommend using [text-generation-inference](https://github.com/huggingface/text-generation-inference) for inferencing open-weight models

For example this would setup a simple tgi instance using docker

```bash
sudo docker run --gpus '"device=0"' \
    --shm-size 1g -p 8020:80 \
    -v /volume/saved_model/:/data ghcr.io/huggingface/text-generation-inference:1.1.0 \
    --max-input-length 4000 \
    --max-total-tokens 4096 \
    --model-id  GeneZC/MiniChat-3B
```
Note: For 5 shot settings, one might need to supply more than 5200 max-input-length to fit in the entire prompt

Once the server has warmed up, simply assign the models and IP:Port to the evaluation cli

```
ieval GeneZC/MiniChat-3B --ip_addr 0.0.0.0:8020
```

For custom models, you might need to provide tokens text for system, user, assistant and end of sentence.

```
ieval GeneZC/MiniChat-3B --ip_addr 0.0.0.0:8020 \
    --sys_token "<s> [|User|] " \
    --usr_token "<s> [|User|] " \
    --ast_token "[|Assistant|]" \
    --eos_token "</s>"
```

You can run `ieval supported` to check models which we have already included with chat prompt. (This feature will be deprecated once more models support format chat prompt function)


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
