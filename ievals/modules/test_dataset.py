from datasets import load_dataset
from tmmlu2.settings import task_list, categories, subject2category

if __name__ == "__main__":
    total = 0
    total_test, total_dev, total_validation = 0, 0, 0
    category_counts = {}
    for subject in task_list:
        print(subject)
        category = subject2category[subject]
        if category not in category_counts:
            category_counts[category] = 0
        test = load_dataset("ikala/tmmluplus", subject)["test"]
        dev = load_dataset("ikala/tmmluplus", subject)["train"]
        val = load_dataset("ikala/tmmluplus", subject)["validation"]
        print(len(val), len(test), len(dev))
        total += len(val) + len(test) + len(dev)
        total_test += len(test)
        total_dev += len(dev)
        total_validation += len(val)
        category_counts[category] += len(val) + len(test) + len(dev)
    print(total)
    print(category_counts)
    print(total_test, total_dev, total_validation)
