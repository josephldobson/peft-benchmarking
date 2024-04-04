import os
import datasets
from peft import PromptTuningConfig, TaskType, LoraConfig, PrefixTuningConfig, PromptEncoderConfig, IA3Config, AdaLoraConfig
import pandas as pd


def get_peft_configuration(PEFT_METHOD, model):
    if PEFT_METHOD == "LORA":
        config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            lora_dropout=0.1,
            r=16,
            lora_alpha=8,
        )

    elif PEFT_METHOD == "DORA":
        config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            lora_dropout=0.1,
            r=16,
            lora_alpha=8,
            use_dora=True,
            target_modules=["q", "k", "v", "o", "wi", "wo"],
        )

    elif PEFT_METHOD == "PROMPT_TUNING":
        config = PromptTuningConfig(
            peft_type=PEFT_METHOD,
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=20
        )

    elif PEFT_METHOD == "PREFIX_TUNING":
        config = PrefixTuningConfig(
            peft_type=PEFT_METHOD,
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=20
        )

    elif PEFT_METHOD == "P_TUNING":
        config = PromptEncoderConfig(
            peft_type="P_TUNING",
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=20,
            encoder_hidden_size=128,
            encoder_num_layers = 2
        )

    elif PEFT_METHOD == "IA3":
        config = IA3Config(
            task_type=TaskType.SEQ_2_SEQ_LM,
        )

    elif PEFT_METHOD == "ADALORA":
        config = AdaLoraConfig(
            peft_type="ADALORA",
            task_type=TaskType.SEQ_2_SEQ_LM,
            lora_dropout=0.1,
            r=16,
            lora_alpha=8,
        )

    else:
        print("Invalid PEFT method")
        return None

    return config


def tokenize_function(tokenizer, model, x):
    question = x['question']
    choices = x['choices']

    ans = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    options = " ".join([f"{chr(65+i)}. {choice} " for i, choice in enumerate(choices)])
    input = f"{question} Pick the correct answer from the following options: {options}\nAnswer with A, B, C or D: "
    output = ans[x['answer']]

    tokenized_inputs = tokenizer(input, padding="max_length", truncation=True, max_length=model.config.max_length)
    tokenized_targets = tokenizer(output, padding="max_length", truncation=True, max_length=model.config.max_length)
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_targets["input_ids"],
    }


def prepare_flan_datasets(model, tokenizer):
    NUM_PROCS = os.cpu_count() - 1
    if os.path.exists("data/tokenized_testsets") and os.path.exists("data/tokenized_trainsets"):
        tokenized_trainsets = datasets.load_from_disk("data/tokenized_trainsets")
        tokenized_testsets = datasets.load_from_disk("data/tokenized_testsets")
        tokenized_valsets = datasets.load_from_disk("data/tokenized_valsets")
        return tokenized_trainsets, tokenized_testsets, tokenized_valsets

    else:
        trainset = datasets.load_dataset("cais/mmlu", "all", split='auxiliary_train+validation')
        testset = datasets.load_dataset("cais/mmlu", "all", split="test")
        devset = datasets.load_dataset("cais/mmlu", "all", split="dev")

        tokenized_trainsets = trainset.map(
            lambda x: tokenize_function(tokenizer, model, x),
            batched=False
        )

        tokenized_testsets = testset.map(
            lambda x: tokenize_function(tokenizer, model, x),
            batched=False
        )

        tokenized_valsets = devset.map(
            lambda x: tokenize_function(tokenizer, model, x),
            batched=False
        )

        tokenized_trainsets.save_to_disk("data/tokenized_trainsets")
        tokenized_testsets.save_to_disk("data/tokenized_testsets")
        tokenized_valsets.save_to_disk("data/tokenized_valsets")

        return tokenized_trainsets, tokenized_testsets, tokenized_valsets
