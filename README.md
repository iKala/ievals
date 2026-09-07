# iEvals : iKala's Evaluator for Large Language Models

<p align="center"> <img src="resources/ieval_cover.png" style="width: 50%; max-width: 400px" id="title-icon">       </p>


iEvals is a framework for evaluating chinese large language models (LLMs), especially performance in traditional chinese domain. Our goal was to provide an easy to setup and fast evaluation library for guiding the performance/use on existing chinese LLMs.

Currently, we only support evaluation for [TMMLU+](https://huggingface.co/datasets/ikala/tmmluplus), however in the future we are exploring more domain, ie knowledge extensive dataset (CMMLU, C-Eval) as well as context retrieval and multi-conversation dataset.

## Dataset version note

TMMLU+ now has a **v1.1** release: the question set was systematically re-verified, with 197 questions having their answer corrected and 539 questions removed for being defective or unresolvable. `ievals` currently loads whichever revision is tagged `main` on the dataset repo; see [Dataset Versions](https://huggingface.co/datasets/ikala/tmmluplus#dataset-versions) on the dataset page for what changed between v1.0 and v1.1.

# Installation

```bash
pip install git+https://github.com/ikala-corp/ievals.git
```

# Usage

```bash
ieval <model name> <series: optional> --top_k <numbers of incontext examples>
```

For more details please refer to [models](MODELS.md) section

# Citation

```
@article{ikala2023eval,
  title={An Improved Traditional Chinese Evaluation Suite for Foundation Model},
  author={Tam, Zhi-Rui and Pai, Ya-Ting},
  journal={arXiv},
  year={2023}
}
```

## About iKala

iKala helps enterprises make better, faster decisions by embedding AI and data at the core of their business. We support AI transformation by helping organizations move from data to decisions, delivering full AI solutions that combine their first-party data with iKala's intelligence built on billions of global social signals.

Headquartered in Taiwan with a global footprint, iKala serves over 1,000 enterprises and 50,000 brands across more than 190 countries, including Fortune 500 companies.

<img src="https://huggingface.co/datasets/patrick000517/Test/resolve/main/ikala_logo.png" alt="iKala logo" width="20" style="vertical-align: middle; border-radius: 4px;" /> iKala Official Website: [ikala.ai](https://ikala.ai)<br />
<img src="https://huggingface.co/datasets/patrick000517/Test/resolve/main/kolr_logo.png" alt="Kolr logo" width="20" style="vertical-align: middle; border-radius: 4px;" /> Kolr Official Website: [kolr.ai](https://kolr.ai)<br />
<img src="https://huggingface.co/datasets/patrick000517/Test/resolve/main/kuroma_logo.png" alt="Kuroma logo" width="20" style="vertical-align: middle; border-radius: 4px;" /> Kuroma Official Website: [kuroma.ai](https://kuroma.ai)
