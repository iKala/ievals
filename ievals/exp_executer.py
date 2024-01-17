import os
import logging
import pandas as pd
from datasets import load_dataset


def get_exp_setting(dataset):
    if 'ikala/tmmlu' in dataset.lower():
        from .settings import task_list, subject2name, subject2category

        return task_list, subject2name, subject2category
    elif '/cmmlu' in dataset.lower():
        from .cmmlu_settings import task_list, subject2name, subject2category
        return task_list, subject2name, subject2category
    elif '/c-eval' in dataset.lower():
        from .ceval_settings import task_list, subject2name, subject2category
        return task_list, subject2name, subject2category

    raise ValueError('dataset not supported')


def run_exp(
    evaluator,
    model_name,
    dataset,
    postfix_name="tgi",
    cache_path=".cache",
    split_name="test",
    few_shot=False,
    cot=False,
):
    model_name_path = model_name.replace("/", "_")
    save_result_dir = None

    if cache_path:
        os.makedirs(f"{cache_path}", exist_ok=True)
        os.makedirs(f"{cache_path}/{model_name_path}", exist_ok=True)
        save_result_dir = f"{cache_path}/{model_name_path}"

    task_list, subject2name, subject2category = get_exp_setting(dataset)
    postfix = model_name.split("/")[-1]
    prefix_name = dataset.split("/")[-1]
    result_cache = f"{prefix_name}_{postfix_name}.tsv"
    if os.path.exists(result_cache):
        logging.info(f"Found previous cache {result_cache}, skipping executed subjects")
        df = pd.read_csv(result_cache, delimiter="\t", header=None)
        df.columns = ["model_name", "subject", "score"]
        finished_subjects = df["subject"].tolist()
        task_list = [t for t in task_list if t not in finished_subjects]

    output_filename = ""
    # TODO: absract out the dataset-task logic, as this is likely
    #       limited under multi subject task only
    for task in task_list:
        zh_name = subject2name[task]
        test = load_dataset(dataset, task)[split_name]
        test_df = pd.DataFrame([dict(row) for row in test])
        dev = load_dataset(dataset, task)["train"]
        dev_df = pd.DataFrame([dict(row) for row in dev])

        accuracy = evaluator.eval_subject(
            zh_name,
            test_df,
            dev_df=dev_df,
            few_shot=few_shot,
            cot=cot,
            save_result_dir=f"{cache_path}/{model_name_path}",
        )

        with open(result_cache, "a") as fout:
            fout.write("{}\t{}\t{:.5f}\n".format(model_name, task, accuracy))

    df = pd.read_csv(result_cache, delimiter="\t", header=None)
    df.columns = ["model_name", "subject", "score"]
    for model_name in df["model_name"].unique():
        print(model_name)
