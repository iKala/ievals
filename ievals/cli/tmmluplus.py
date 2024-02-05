import os

import logging
import argparse
import pandas as pd
import numpy as np
from ievals.settings import categories, subject2category
def get_parser():
    parser = argparse.ArgumentParser(description="Run TMMLU+ score aggregate")
    parser.add_argument("result_file", type=str, help="Name of the eval model")

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    csv_filename = args.result_file
    df = pd.read_csv(csv_filename, delimiter='\t', header=None)

    inverted_categories = {}
    for big_cat, cats in categories.items():
        for cat in cats:
            inverted_categories[cat] = big_cat
    df.columns = ['model_name', 'subject', 'accuracy']
    model_scores = []
    for model_name in df.model_name.unique():
        results = {}
        total = 0
        for idx, row in df[df['model_name'] == model_name].iterrows():
            total += 1
            subject = row['subject']
            category = inverted_categories[subject2category[subject]]
            if category not in results:
                results[category] = []
            results[category].append(row['accuracy'])
        if total == 66:
            data = {'model_name': model_name}
            assert total == 66
            avg_scores = 0
            for category, scores in results.items():
                data[category] = np.mean(scores)
                avg_scores += np.mean(scores)
            data['Average'] = avg_scores/4
            model_scores.append(data)
        else:
            print(model_name, total)
            data = {'model_name': model_name}
            avg_scores = 0
            for category, scores in results.items():
                data[category] = np.mean(scores)
                avg_scores += np.mean(scores)
            data['Average'] = avg_scores/4
            print(data)
            print('------------')
    model_scores = sorted(model_scores, key=lambda x:x['Average'], reverse=True)

    for score in model_scores:
        print(score)
    return model_scores