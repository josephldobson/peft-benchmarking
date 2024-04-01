from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from datasets import load_dataset, DatasetDict, load_from_disk
from peft import PeftModel, PeftConfig, get_peft_model, PromptTuningConfig, TaskType, LoraConfig, PrefixTuningConfig, PromptEncoderConfig
from utils import tokenize_function, get_peft_configuration, prepare_flan_datasets
from gpu_usage_callback import GpuUsageCallback
import pandas as pd
import os
import time

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
        num_train_epochs=num_epochs,
        save_strategy="no",
    )

    training_start_time = time.time()
    trainer = Seq2SeqTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_trainsets,
        callbacks=[GpuUsageCallback]
    )
    model.config.use_cache = False
    trainer.train()
    training_duration = time.time() - training_start_time
    print(f"training completed in {(training_duration / 3600) :.2f} hours")

    pipe = pipeline(
        task="question-answering",
        model=model,
        tokenizer=tokenizer)

    model.print_trainable_parameters()

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    for PEFT_METHOD in ["LORA", "PROMPT_TUNING", "PREFIX_TUNING", "P_TUNING", "IA3", "ADALORA", "DORA"]:
        MODEL_NAME = "t5-small"
        BATCH_SIZE = 128
        NUM_EPOCHS = 1

        train_and_save(PEFT_METHOD, MODEL_NAME, BATCH_SIZE, NUM_EPOCHS)
