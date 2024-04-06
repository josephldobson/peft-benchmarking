import matplotlib.pyplot as plt
import re
import os
from collections import defaultdict

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

    for method, accuracies in method_accs.items():
        plt.figure(figsize=(10, 15))  
        plt.barh(range(len(accuracies)), accuracies)
        plt.title(f'Accuracy Values for {method}')
        plt.yticks(range(len(accuracies)), subject_names[:len(accuracies)])
        plt.xlabel('Accuracy')
        plt.tight_layout() 
        file_path = os.path.join(output_dir, f'{method}_chart.png')
        plt.savefig(file_path)

def make_comparison__graph(method_accs):

    method_accs['Flan-T5-Base'] = flan_T5_base_values

    plt.figure(figsize=(12, 8))

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange', 'black']
    method_colors = dict(zip(method_accs.keys(), colors))

    for method, accuracies in method_accs.items():
        x_values = list(range(len(accuracies))) 
        y_values = accuracies  
        plt.scatter(x_values, y_values, label=method, color=method_colors[method], s=100)  

    plt.title('Accuracy Values Comparison Across Methods')
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(accuracies)), subject_names[:len(accuracies)], rotation=90)
    plt.legend()

    plt.tight_layout()  
    file_path = os.path.join(output_dir, 'method_accuracy_comparisons.png')
    plt.savefig(file_path)


def average_accuracies(method_accs):

    avg_accuracies = {method: sum(values) / len(values) for method, values in method_accs.items()}
    avg_accuracies['Flan-T5-Base'] = 0.337
    sorted_methods = sorted(avg_accuracies.items(), key=lambda x: x[1], reverse=True)
    methods, averages = zip(*sorted_methods)

    plt.figure(figsize=(10, 6))
    plt.bar(methods, averages)

    plt.title('Average Accuracies by Method')
    plt.xlabel('Method')
    plt.ylabel('Average Accuracy')
    plt.xticks(rotation=45)

    plt.tight_layout() 
    file_path = os.path.join(output_dir, 'average_accuracy_by_method.png')
    plt.savefig(file_path)
    plt.close()


if __name__ == '__main__':
   method_accs = clean_subject_accuracies()
   make_save_histograms(method_accs=method_accs)
   make_comparison__graph(method_accs=method_accs)
   average_accuracies(method_accs=method_accs)