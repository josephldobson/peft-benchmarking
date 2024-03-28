import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, logging, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model, PromptTuningConfig, TaskType, LoraConfig, PrefixTuningConfig, PromptEncoderConfig
from utils import tokenize_function, get_peft_configuration, prepare_flan_datasets
from datasets import load_dataset, DatasetDict, load_from_disk
import numpy as np
import os

def format_mmlu_example(example, incl_answer = False):
    # Extracting the components of the example
    question = example['question']
    choices = example['choices']
    answer = example['answer']

    # Formatting the choices
    options = "\n".join([f"{chr(65+i)}: {choice}" for i, choice in enumerate(choices)])

    # Formatting the entire example
    if incl_answer:
        formatted_example = f"Question: {question}\nOptions:\n{options}\nAnswer: {chr(65+answer)}."
    else:
        formatted_example = f"Question: {question}\nOptions:\n{options}\nAnswer:"

    return formatted_example

def eval_mmlu(model_path, tokenizer_path, verbose=True):
    """
    Evaluates a model on the MMLU dataset, using 5-shot prompting.

    Args:
        model_path (str): path to model
        tokenizer_path (str): path to model tokenizer

    Returns:
        test_accuracy (float): model accuracy
        subject_acc (dict): dictionary with accuracy by subject, with values ([num_correct,total],accuracy)
    """
    ## load the pretrained model and tokenizer
    # model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    # tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    # model.eval()

    config = PeftConfig.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, return_dict=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, model_path)

    ## datasets

    mmlu_dataset = load_dataset("cais/mmlu", 'all', split='test')

    ## metrics

    subjects = set(mmlu_dataset['subject'])
    subject_acc = {label: [[0, 0],0] for label in subjects}

    for subject in subjects:
        subject_acc[subject][0][1] = len(mmlu_dataset.filter(lambda example: example['subject'] == subject))-5

    ## evaluation
    correct = 0
    total = len(mmlu_dataset) - 5*len(set(mmlu_dataset['subject']))

    for subject in subjects:

        if verbose:
            print(f'\nTesting on {subject}\n')

        # FLAN uses 5-shot prompting for MMLU, so we use that here
        subject_set = mmlu_dataset.filter(lambda example: example['subject'] == subject)
        dev_set, test_set = subject_set.select(range(5)), subject_set.select(range(5,len(subject_set)))
        formatted_egs = [format_mmlu_example(eg,incl_answer=True) for eg in dev_set]
        five_shot_text = "\n\n".join(formatted_egs)

        for example in test_set:

            question = example['question']
            subject = example['subject']
            choices = example['choices']
            answer = example['answer']

            # format inputs
            formatted_example = format_mmlu_example(example, incl_answer=False)
            input_text = '\n\n'.join((five_shot_text, formatted_example))
            inputs = tokenizer(input_text, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Ensure the model is in evaluation mode and torch.no_grad() is used for inference

            ## BROKEN -------------- (doesnt seem to be giving right output)
            model.eval()  # Make sure the model is in evaluation mode
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_length=10
                )

            output_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            first_char = output_answer.strip()[0].upper() if output_answer else ''
            ## -------------------
            # Assuming your model outputs the option letter (e.g., "A") as the answer
            predicted_option = ord(first_char) - ord('A')

            # Update the metric
            if predicted_option == answer:
                correct += 1
                subject_acc[subject][0][0] += 1

        subject_acc[subject][1] = subject_acc[subject][0][0]/subject_acc[subject][0][1]
        print(f'Accuracy on {subject}: {subject_acc[subject][1]:.3f}')

    test_accuracy = correct/total

    print(f"Accuracy on MMLU: {test_accuracy:.4f}")

    return test_accuracy, subject_acc

if __name__ == '__main__':
    eval_mmlu('peft-benchmarking/src/models/t5-small_LORA','peft-benchmarking/src/models/t5-small_LORA')
