from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, logging
from datasets import load_dataset, DatasetDict, load_from_disk
from peft import PeftModel, PeftConfig, get_peft_model, PromptTuningConfig, TaskType, LoraConfig, PrefixTuningConfig, PromptEncoderConfig
from utils import tokenize_function, get_peft_configuration, prepare_flan_datasets
import pandas as pd
import os


def train_and_save(peft_method, model_name, batch_size, num_epochs):

    output_dir = "models/" + model_name + "_" + peft_method

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    config = get_peft_configuration(peft_method, model)

    if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

    peft_model = get_peft_model(model, config)
    tokenized_trainsets, tokenized_testsets = prepare_flan_datasets(model, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
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
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    for PEFT_METHOD in ["LORA", "PROMPT_TUNING", "PREFIX_TUNING", "P_TUNING", "IA3"]:
        MODEL_NAME = "t5-small"
        BATCH_SIZE = 128
        NUM_EPOCHS = 4

        train_and_save(PEFT_METHOD, MODEL_NAME, BATCH_SIZE, NUM_EPOCHS)
