def preprocess_function(examples):
    inputs = []
    targets = []

    for question, choices, answer in zip(examples["question"], examples["choices"], examples["answer"]):
        input_text = f"question: {question} options: {', '.join(choices)}"
        target_text = f"{choices[answer]}"

        inputs.append(input_text)
        targets.append(target_text)

    return {"input_text": inputs, "target_text": targets}