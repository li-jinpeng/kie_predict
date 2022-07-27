import os
from dataclasses import dataclass, field

import numpy as np
from datasets import load_dataset

import transformers
from layoutlmft.data import DataCollatorForKeyValueExtraction
from layoutlmft.trainers import XfunSerTrainer
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

def my_predict(train_data_path,model_path,opath):

    datasets = load_dataset(
        os.path.abspath('/home/lijinpeng/kie_predict/shoppingdata.py'),
        keep_in_memory=True,
    )
    
    config = AutoConfig.from_pretrained(
        model_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        config=config,
    )

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8,
        padding=True,
        max_length=512,
    )

    trainer = XfunSerTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        #compute_metrics=compute_metrics,
    )
 

    predictions, labels, metrics = trainer.predict(datasets)
    predictions = np.argmax(predictions, axis=2)

    label_list = get_label_list(labels)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Save predictions
    output_test_predictions_file = os.path.join(opath, "test_predictions.txt")
    if trainer.is_world_process_zero():
        with open(output_test_predictions_file, "w") as writer:
            for prediction in true_predictions:
                writer.write(" ".join(prediction) + "\n")


