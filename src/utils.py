import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, logging
import datasets
from peft import PeftModel, PeftConfig, get_peft_model, PromptTuningConfig, TaskType, LoraConfig, PrefixTuningConfig, PromptEncoderConfig, IA3Config
import pandas as pd


def tokenize_function(tokenizer, model, x):
    tokenized_inputs = tokenizer(x['source'], padding="max_length", truncation=True, max_length=model.config.max_length)
    tokenized_targets = tokenizer(x['target'], padding="max_length", truncation=True, max_length=model.config.max_length)
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_targets["input_ids"],
    }

#TODO Confgure the PEFT methods with the right parameters
def get_peft_configuration(PEFT_METHOD, model):
    if PEFT_METHOD == "LORA":
        config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM
        )

    elif PEFT_METHOD == "PROMPT_TUNING":
        config = PromptTuningConfig(
            peft_type=PEFT_METHOD,
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=20,
        )

    elif PEFT_METHOD == "PREFIX_TUNING":
        config = PrefixTuningConfig(
            peft_type=PEFT_METHOD,
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=20,
        )

    elif PEFT_METHOD == "P_TUNING":
        config = PromptEncoderConfig(
            peft_type="P_TUNING",
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=20,
            encoder_hidden_size=128
        )

    elif PEFT_METHOD == "IA3":
        config = IA3Config(
            task_type=TaskType.SEQ_2_SEQ_LM
        )

    else:
        print("Invalid PEFT method")
        return None

    return config


def prepare_flan_datasets(model, tokenizer):
    NUM_PROCS = os.cpu_count() - 1
    if os.path.exists("data/tokenized_testsets"):
        tokenized_trainsets = datasets.load_from_disk("data/tokenized_trainsets")
        tokenized_testsets = datasets.load_from_disk("data/tokenized_testsets")
        return tokenized_trainsets, tokenized_testsets

    else:
        dataset = datasets.load_dataset("sordonia/flan-10k-flat", split="train[:1%]+train[-1%:]") # split = "train" for full dataset
        flan_dict = pd.read_csv("data/flan_collection_info.csv")

        multi_choice_qa_tasks_list = flan_dict.loc[flan_dict["Generic Task Category"] == "Multiple-Choice QA (no trivia knowledge required)"]["Specific Task Category"].drop_duplicates().tolist()
        multi_choice_qa_tasks_set = set(multi_choice_qa_tasks_list)
        mc_qa_dataset = dataset.filter(lambda r: r["task_name"] in multi_choice_qa_tasks_set, num_proc=NUM_PROCS)
        mc_qa_trainset = mc_qa_dataset.filter(lambda r: r["split"] == "train")
        mc_qa_testset = mc_qa_dataset.filter(lambda r: r["split"] == "test")

        tokenized_trainsets = mc_qa_trainset.map(
            lambda x: tokenize_function(tokenizer, model, x),
            batched=True
        )

        tokenized_testsets = mc_qa_testset.map(
            lambda x: tokenize_function(tokenizer, model, x),
            batched=True
        )
        tokenized_trainsets.save_to_disk("data/tokenized_trainsets")
        tokenized_testsets.save_to_disk("data/tokenized_testsets")

        return tokenized_trainsets, tokenized_testsets
