# iEvals : iKala's Evaluator for Large Language Models

<p align="center"> <img src="resources/ieval_cover.png" style="width: 50%; max-width: 400px" id="title-icon">       </p>


iEvals is a framework for evaluating chinese large language models (LLMs), especially performance in traditional chinese domain. Our goal was to provide an easy to setup and fast evaluation library for guiding the performance/use on existing chinese LLMs.

Currently, we only support evaluation for [TMMLU+](https://huggingface.co/datasets/ikala/tmmluplus), however in the future we are exploring more domain, ie knowledge extensive dataset (CMMLU, C-Eval) as well as context retrieval and multi-conversation dataset.


# Usage

```bash
ieval <model name> <series: optional> --top_k <numbers of incontext examples>
```

For more details please refer to [models](MODELS.md) section

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
