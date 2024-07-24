# Supported models

Closed source API : OpenAI, Azure, Anthropic, Gemini

Open weights models : 
1. We rely on tgi for inference, checkout [**Text generation inference**](#text-generation-inference) for more details.
2. For Qwen models or other local models, please visit [**Qwen models or other local models**](#qwen-models-or-other-local-models) section.


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

```bash
ieval GeneZC/MiniChat-3B --ip_addr 0.0.0.0:8020
```

For custom models, you might need to provide tokens text for system, user, assistant and end of sentence.

```bash
ieval GeneZC/MiniChat-3B --ip_addr 0.0.0.0:8020 \
    --sys_token "<s> [|User|] " \
    --usr_token "<s> [|User|] " \
    --ast_token "[|Assistant|]" \
    --eos_token "</s>"
```

You can run `ieval supported` to check models which we have already included with chat prompt. (This feature will be deprecated once more models support format chat prompt function)


## OpenAI

```bash
ieval gpt-3.5-turbo-0613 --series openai_chat --api_key "<Your OpenAI platform Key>"
```

```bash
ieval gpt-3.5-turbo-instruct --series openai_complete --api_key "<Your OpenAI platform Key>" --top_k 5
```



## Gemini Pro
Use str parsing the answers -
```bash
ieval gemini-pro  --api_key "<Your API Key from https://ai.google.dev/>" --top_k 5
```

Use LLM parsing the answers -
```bash
API_KEY="<Gemini API Key>" ieval gemini-pro  --api_key "<Your API Key from https://ai.google.dev/>" --top_k 5
```

Currently we do not support models from vertex AI yet. So PaLM (bison) series are not supported

## Anthropic (instant, v1.3, v2.0, v3.0 Haiku, v3.0 Opus, v3.0 Sonnet)
Use str parsing the answers -
```bash
ieval claude-instant-1  --api_key "<Anthropic API keys>"
```

Use LLM parsing the answers -
```bash
API_KEY="<Gemini API Key>" ieval claude-instant-1  --api_key "<Anthropic API keys>"
```

## Azure OpenAI Model
Use str parsing the answers -
```bash
export AZURE_OPENAI_ENDPOINT="https://XXXX.azure.com/"
ieval <your azure model name> --series azure --api_key "<Your API Key>" --top_k 5
```

Use LLM parsing the answers -
```bash
API_KEY="<Gemini API Key>" ieval <your azure model name> --series azure --api_key "<Your API Key>" --top_k 5
```

We haven't experimented with instruction based model from azure yet, so for instruction based models, you will have to fallback to openai's models

## DashScope

Before using models from dashscope please install it via pypi

```bash
pip install dashscope==1.13.6
```

Once installed, you should be able to run:
Use str parsing the answers -
```bash
ieval <Your model name> --api_key "<Dash Scope API>"
```

Use LLM parsing the answers -
```bash
API_KEY="<Gemini API Key>" ieval <Your model name> --api_key "<Dash Scope API>"
```

Supported models : qwen-turbo, qwen-plus, qwen-max, qwen-plus-v1, bailian-v1

## Qwen models or other local models
Use str parsing the answers -
```bash
CUDA_VISIBLE_DEVICES=1 ieval Qwen/Qwen-7B-Chat
```

or 

```bash
CUDA_VISIBLE_DEVICES=1 ieval Qwen/Qwen-7B-Chat --series hf_chat
```

Use LLM parsing the answers -
```bash
API_KEY="<Gemini API Key>" CUDA_VISIBLE_DEVICES=1 ieval Qwen/Qwen-7B-Chat --series hf_chat
```

If the mentioned model is private you can pass in your huggingface read token via --api_key argument

## Reka

Before using models from reka please install it via pypi

```bash
pip install reka
```

Once installed, you should be able to run:

```bash
ieval <reka-flash, reka-flash, reka-edge> --api_key "<Reka API Key>"
```


## Groq

Before using models from Groq please install it via pypi

```bash
pip install groq
```

Once installed, you should be able to run:

```bash
ieval llama3-8b-8192 --series groq --api_key "<Groq API : gsk_XXXXX>"
```


## Together

Before using models from Together please install it via pypi

```bash
pip install together
```

Once installed, you should be able to run:

```bash
ieval meta-llama/Llama-3-70b-chat-hf --series together  --api_key "XXX"
```
