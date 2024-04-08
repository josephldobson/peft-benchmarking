from numpy.random import test
import torch
import pandas as pd
import time
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, logging, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel, PeftConfig, get_peft_model, PromptTuningConfig, TaskType, LoraConfig, PrefixTuningConfig, PromptEncoderConfig
from utils import tokenize_function, get_peft_configuration, prepare_flan_datasets
from datasets import load_dataset, DatasetDict, load_from_disk
import numpy as np
import os
import pickle as pkl

def format_mmlu_example(example, incl_answer = False, five_shot=False):
    # Extracting the components of the example
    question = example['question']
    choices = example['choices']
    answer = example['answer']

    # Formatting the choices
    options = " ".join([f"{chr(65+i)}. {choice} " for i, choice in enumerate(choices)])
    formatted_example = f"{question} Pick the correct answer from the following options: {options}\nAnswer with A, B, C or D: "

    # Formatting the entire example
    if incl_answer:
        formatted_example = f"{formatted_example}{chr(65+answer)}"

    if five_shot:
        formatted_example = '\n\n'.join((five_shot, formatted_example))

    return {'formatted':formatted_example}


class CustomDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Extract the items you want to return
        item = self.ds[idx]
        formatted_text = item['formatted']
        answer = item['answer']
        return formatted_text, answer


def eval_mmlu(model_path, PEFT=True):
    """
    Evaluates a model on the MMLU dataset, using 5-shot prompting.

    Args:
        model_path (str): path to model
        tokenizer_path (str): path to model tokenizer
        PEFT (bool): whether to use PEFT configurations
    Returns:
        test_accuracy (float): model accuracy
        subject_acc (dict): dictionary with accuracy by subject, with values ([num_correct,total],accuracy)
    """
    np.random.seed(0)
    torch.manual_seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## load the pretrained model and tokenizer
    if PEFT:
        config = PeftConfig.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path,
                                                      #load_in_8bit=True,
                                                      )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_path).to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path
                                                      #load_in_8bit=True
                                                      ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    ## datasets
    mmlu_dataset = load_dataset("cais/mmlu", 'all', split='auxiliary_train')

    ## metrics
    subjects = set(mmlu_dataset['subject'])
    subject_acc = {label: [[0, 0],0] for label in subjects}

    for subject in subjects:
        subject_acc[subject][0][1] = len(mmlu_dataset.filter(lambda example: example['subject'] == subject))-5

    ## evaluation
    correct = 0
    total = len(mmlu_dataset) - 5*len(set(mmlu_dataset['subject']))

    for subject in subjects:
        print(f'\nEvaluating {subject}\n')
        # FLAN uses 5-shot prompting for MMLU, so we use that here
        subject_set = mmlu_dataset.filter(lambda example: example['subject'] == subject)
        dev_set, test_set = subject_set.select(range(5)), subject_set.select(range(5,len(subject_set)))
        formatted_egs = [format_mmlu_example(eg,incl_answer=True)['formatted'] for eg in dev_set]
        five_shot_text = "\n\n".join(formatted_egs)
        begin = time.time()

        input_texts = test_set.map(format_mmlu_example, fn_kwargs={"incl_answer": False, "five_shot": five_shot_text})

        input_texts = CustomDataset(input_texts.map(remove_columns=['subject','question','choices']))

        inputDL = DataLoader(input_texts, batch_size=16, shuffle=True, num_workers=0)

        for i, (prompts, answers) in enumerate(inputDL):

            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                output_ids = model.generate(input_ids=inputs.input_ids)

            output_answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            first_chars = [x.strip()[0].upper() if x else '' for x in output_answers]
            #print(first_chars)

            # map outputs to 0-3
            predicted_options = np.array([ord(x) for x in first_chars]) - ord('A')

            batch_acc = np.sum(predicted_options==answers.numpy())
            subject_acc[subject][0][0] += batch_acc

            # print(f'batch {i}, acc {batch_acc}')


        end = time.time()

        subject_acc[subject][1] = subject_acc[subject][0][0]/subject_acc[subject][0][1]
        print(f'Accuracy on {subject}: {subject_acc[subject][1]:.3f}; took {end-begin}s')

    test_accuracy = correct/total

    print(f"Accuracy on MMLU: {test_accuracy:.4f}")

    return test_accuracy, subject_acc

if __name__ == '__main__':
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    for PEFT_METHOD in [x"P_TUNING", "PREFIX_TUNING", "PROMPT_TUNING", "LORA"]:
        test_acc, subject_acc = eval_mmlu(f'models/google/flan-t5-base_{PEFT_METHOD}_1')

        # Forming the file paths
        acc_file_path = os.path.join(results_dir, f'flan-t5-base_{PEFT_METHOD}_1_MMLU-acc.pickle')
        subject_accs_file_path = os.path.join(results_dir, f'flan-t5-base_{PEFT_METHOD}_1_subject-accs.pickle')

        with open(acc_file_path, 'wb') as handle:
            pkl.dump(test_acc, handle)

        with open(subject_accs_file_path, 'wb') as handle:
            pkl.dump(subject_acc, handle)
