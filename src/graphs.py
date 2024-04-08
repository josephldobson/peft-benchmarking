import os
import pickle

results_dir = 'src/results'
output_file = 'src/pickle_contents.txt'

with open(output_file, 'w') as output:

    if os.path.isdir(results_dir):
        output.write(f"The directory '{results_dir}' was found. Looking for pickle files...\n")
        files = os.listdir(results_dir)

        pickle_files = [file for file in files if file.endswith('.pickle')]

        if not pickle_files:
            output.write("No pickle files found in the directory.\n")
        else:
            for pickle_file in pickle_files:
                file_path = os.path.join(results_dir, pickle_file)

                try:
                    with open(file_path, 'rb') as handle:
                        data = pickle.load(handle)
                    output.write(f"Contents of {pickle_file}:\n{data}\n\n")
                except Exception as e:
                    output.write(f"Failed to read {pickle_file}: {e}\n")
    else:
        output.write(f"The directory '{results_dir}' does not exist or is not in the current directory.\n")

print(f"The contents of the pickle files have been written to {output_file}")

import matplotlib.pyplot as plt
import re
import os
from collections import defaultdict
import numpy as np
import seaborn as sns

output_dir = "src/method_graphs"
os.makedirs(output_dir, exist_ok=True)

subject_names = [
    'public_relations', 'elementary_mathematics', 'professional_accounting',
    'electrical_engineering', 'high_school_macroeconomics', 'moral_scenarios',
    'college_chemistry', 'econometrics', 'anatomy', 'business_ethics',
    'human_sexuality', 'high_school_microeconomics', 'astronomy',
    'conceptual_physics', 'international_law', 'professional_medicine',
    'machine_learning', 'security_studies', 'global_facts',
    'high_school_mathematics', 'prehistory', 'high_school_government_and_politics',
    'formal_logic', 'college_computer_science', 'high_school_psychology',
    'professional_psychology', 'virology', 'logical_fallacies',
    'high_school_us_history', 'high_school_european_history', 'high_school_geography',
    'high_school_statistics', 'college_medicine', 'high_school_world_history',
    'human_aging', 'high_school_chemistry', 'management', 'philosophy',
    'professional_law', 'us_foreign_policy', 'moral_disputes', 'world_religions',
    'jurisprudence', 'nutrition', 'high_school_computer_science', 'clinical_knowledge',
    'sociology', 'high_school_biology', 'medical_genetics', 'abstract_algebra',
    'marketing', 'college_mathematics', 'high_school_physics', 'miscellaneous',
    'college_biology', 'college_physics', 'computer_security'
]

flan_T5_base_values = [0.50, 0.195, 0.323, 0.250, 0.256, 0.240,
                       0.250, 0.50, 0.429, 0.545, 0.417, 0.423,
                       0.375, 0.192, 0.385, 0.258, 0.364, 0.370,
                       0.50, 0.310, 0.286, 0.524, 0.214, 0.273,
                       0.383, 0.406, 0.444, 0.50, 0.50, 0.50,
                       0.682, 0.435, 0.273, 0.385, 0.217, 0.409,
                       0.455, 0.324, 0.282, 0.364, 0.316, 0.211,
                       0.273, 0.485, 0.222, 0.379, 0.455, 0.250,
                       0.273, 0.273, 0.60, 0.091, 0.235, 0.302,
                       0.250, 0.727, 0.182]


def clean_subject_accuracies():
    with open('src/pickle_contents.txt', 'r') as file:
        contents = file.read()
    method_pattern = re.compile(r"flan-t5-base_(\w+)_\d+_subject-accs.pickle")
    acc_pattern = re.compile(r": \[\[\d+, \d+\], (\d+\.\d+)\]")

    methods = method_pattern.findall(contents)
    contents_per_method = re.split(method_pattern, contents)
    method_accs = defaultdict(list)
    contents_per_method = contents_per_method[1:]

    for i, method in enumerate(methods):
        odd_num = 2*i + 1
        acc_values = acc_pattern.findall(contents_per_method[odd_num])
        acc_values = [float(value) for value in acc_values if value]
        if acc_values:
            method_accs[method].extend(acc_values)

    return method_accs


def make_save_histograms(method_accs):
    method_accs['Flan-T5-Base'] = flan_T5_base_values

    output_dir = "src/method_graphs"
    os.makedirs(output_dir, exist_ok=True)

    sns.set(context='notebook', style='darkgrid')
    colors = sns.color_palette('rocket', n_colors=len(method_accs))

    for i, (method, accuracies) in enumerate(method_accs.items()):
        plt.figure(figsize=(10, 15))
        sns.barplot(x=accuracies, y=np.arange(len(accuracies)), orient='h', palette=[colors[i]])

        plt.title(f'Accuracy values for {method}', fontsize=16, style='italic')
        plt.yticks(np.arange(len(subject_names)), subject_names[:len(accuracies)], fontsize=12)
        plt.xlabel('Accuracy', fontsize=14)
        plt.ylabel('Subject', fontsize=14)

        plt.tight_layout(rect=[0, 0, 0.75, 1])

        file_path = os.path.join(output_dir, f'{method}_chart.pdf')
        plt.savefig(file_path)


def make_comparison__graph(method_accs):

    method_accs['Flan-T5-Base'] = flan_T5_base_values

    sns.set(context='notebook', style='darkgrid')
    fig, ax = plt.subplots(figsize=(12, 15))
    colors = sns.color_palette('rocket', n_colors=len(method_accs))

    for i, (method, accuracies) in enumerate(method_accs.items()):
        y_values = np.arange(len(subject_names))
        sns.scatterplot(ax=ax, x=accuracies, y=y_values, label=method, color=colors[i], s=30)

    ax.set_ylabel('Subject', fontsize=16, style='italic')
    ax.set_xlabel('Accuracy', fontsize=16, style='italic')
    ax.set_yticks(np.arange(len(subject_names)))
    ax.set_yticklabels(subject_names, fontsize=12)

    ax.legend(title='Methods', title_fontsize='10', fontsize='8', loc='upper right', bbox_to_anchor=(1, 1), frameon=True)

    ax.set_ylim(-1, len(subject_names))

    plt.tight_layout(rect=[0, 0, 0.75, 1])

    file_path = os.path.join(output_dir, 'method_accuracy_comparisons.pdf')
    plt.savefig(file_path)


def average_accuracies(method_accs):
    avg_accuracies = {method: sum(values) / len(values) for method, values in method_accs.items()}
    avg_accuracies['Flan-T5-Base'] = 0.337
    sorted_methods = sorted(avg_accuracies.items(), key=lambda x: x[1], reverse=True)
    methods, averages = zip(*sorted_methods)

    sns.set(context='notebook', style='darkgrid')
    plt.figure(figsize=(12, 6))

    sns.barplot(x=list(methods), y=list(averages), palette='rocket')

    plt.title('Average Accuracies by Method', fontsize=18, fontweight='bold')
    plt.xlabel('Method', fontsize=14, style='italic')
    plt.ylabel('Average Accuracy', fontsize=14, style='italic')
    plt.xticks(rotation=45, fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.75, 1])
    file_path = os.path.join(output_dir, 'average_accuracy_by_method.pdf')
    plt.savefig(file_path)


if __name__ == '__main__':
   method_accs = clean_subject_accuracies()
   make_save_histograms(method_accs=method_accs)
   make_comparison__graph(method_accs=method_accs)
   average_accuracies(method_accs=method_accs)
