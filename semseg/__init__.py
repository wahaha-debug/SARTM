from tabulate import tabulate
from semseg import datasets


def show_datasets():
    dataset_names = datasets.__all__
    numbers = list(range(1, len(dataset_names)+1))
    print(tabulate({'No.': numbers, 'Datasets': dataset_names}, headers='keys'))
