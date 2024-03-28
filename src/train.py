from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, DatasetDict, load_from_disk
from peft import PeftModel, PeftConfig, get_peft_model, PromptTuningConfig, TaskType, LoraConfig, PrefixTuningConfig, PromptEncoderConfig
from utils import tokenize_function, get_peft_configuration, prepare_flan_datasets
import pandas as pd
import os


def train_and_save(peft_method, model_name, batch_size, num_epochs):

    output_dir = "models/" + model_name + "_" + peft_method

    tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=True, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    config = get_peft_configuration(peft_method, model)

    if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

    peft_model = get_peft_model(model, config)
    tokenized_trainsets, tokenized_testsets = prepare_flan_datasets(model, tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        per_device_train_batch_size=batch_size,
        learning_rate=1e-3,
        optim="adamw_torch",
        num_train_epochs=2,
        save_strategy="no",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_trainsets,
    )
    model.config.use_cache = False
    trainer.train()
    trainer.eval()

    trainer.model.push_to_hub(output_dir)


if __name__ == '__main__':
    for PEFT_METHOD in ["LORA"]:

    # for PEFT_METHOD in ["LORA", "PROMPT_TUNING", "PREFIX_TUNING", "P_TUNING", "IA3"]:
        MODEL_NAME = "t5-small"
        BATCH_SIZE = 128
        NUM_EPOCHS = 1

        train_and_save(PEFT_METHOD, MODEL_NAME, BATCH_SIZE, NUM_EPOCHS)
