from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict, load_from_disk
from peft import PeftModel, PeftConfig, get_peft_model, PromptTuningConfig, TaskType, LoraConfig, PrefixTuningConfig, PromptEncoderConfig
from utils import tokenize_function, get_peft_configuration, prepare_flan_datasets
import pandas as pd
import os

def train_and_save(peft_method, model_name, batch_size, num_epochs):

    OUTPUT_DIR = "models/" + MODEL_NAME + PEFT_METHOD

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    peft_model = get_peft_configuration(PEFT_METHOD, model)
    tokenized_trainsets, tokenized_testsets = prepare_flan_datasets(model, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_trainsets,
        eval_dataset=tokenized_testsets,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)


if __name__ == '__main__':
    PEFT_METHOD = "PREFIX_TUNING"
    MODEL_NAME = "t5-small"
    BATCH_SIZE = 32
    NUM_EPOCHS = 4

    train_and_save(PEFT_METHOD, MODEL_NAME, BATCH_SIZE, NUM_EPOCHS)
