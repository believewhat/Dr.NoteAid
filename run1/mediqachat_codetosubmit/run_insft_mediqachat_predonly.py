#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import copy
from collections import defaultdict

import torch
import datasets
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric, concatenate_datasets

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


from sum_data_collator import DataCollatorForSumLanguageModeling
from templates import PATTERNS

example_separator = "\n* Another example below *\n"
x_y_delimiter = "\n* Given example above, solve this below *\n"


logger = logging.getLogger(__name__)


# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: str = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()



    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    assert training_args.per_device_train_batch_size == 1
    # task_names = ["mediqa"]
    # task_names = ["samsum", "mediqasum"]
    # task_names = ["cosmos_qa", "dart"]
    datasets_bytask = {}
    data_files = {}
    # data_files["train"] = f"./tmp/mediqachat_medinsgpt_hc_result_to_submit/test.json"
    # data_files["test"] = f"./tmp/mediqachat_medinsgpt_hc_result_to_submit/test.json"
    data_files["train"] = f"./tmp/mediqachat_medinsgpt_hc_result_to_submit/valid.json"
    data_files["test"] = f"./tmp/mediqachat_medinsgpt_hc_result_to_submit/valid.json"
    datasets_bytask["mediqa_predict"] = load_dataset("json", data_files=data_files, use_auth_token=True if model_args.use_auth_token else None)
    data_files = {}
    data_files["train"] = f"./tmp/mediqachat_medinsgpt_hc_result_to_submit/train.json"
    data_files["test"] = f"./tmp/mediqachat_medinsgpt_hc_result_to_submit/train.json"
    datasets_bytask["mediqa_finetune"] = load_dataset("json", data_files=data_files, use_auth_token=True if model_args.use_auth_token else None)
    
    secheader2trainidx = defaultdict(list)
    idx = 0
    for row in datasets_bytask['mediqa_finetune']["train"]:
        secheader2trainidx[row["section_header"]].append(idx)
        idx += 1
    task = "mediqa_finetune"
    sectionhead2indexs = {}
    num_fewshot_target = 5
    for section_header, indexes in secheader2trainidx.items():
        assert num_fewshot_target > 0
        result_strs = []
        if len(indexes) <= num_fewshot_target:
            subindexes = indexes
        else:
            subindexes = np.random.choice(indexes, size=num_fewshot_target, replace=False)
        sectionhead2indexs[section_header] = subindexes

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})
    tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    config.bos_token_id = tokenizer.bos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )
    initial_weights = f"{model_args.model_name_or_path}/pytorch_model.bin"
    model.load_state_dict(torch.load(initial_weights, map_location=torch.device("cpu")))

    model.resize_token_embeddings(len(tokenizer))
    model.bos_token_id = tokenizer.bos_token_id
    model.pad_token_id = tokenizer.pad_token_id

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    # if model.config.decoder_start_token_id is None:
    #     raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )


    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    max_source_length = data_args.max_source_length 
    max_total_len = max_source_length + max_target_length 
    padding_input = "max_length" if data_args.pad_to_max_length else "longest"
    padding_label = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )
        
    # prelena = []
    # aftlena = []
    # for a in train_dataset:
    #     prelena.append(len(a["input_ids"]))
    #     aftlena.append(len(a["labels"]))
    # prelena = np.array(prelena)
    # aftlena = np.array(aftlena)
    # print(prelena.mean())
    # print(prelena.std())
    # print(np.percentile(prelena, 50))
    # print(np.percentile(prelena, 95))
    # print(prelena.max())
    # print()
    # print(aftlena.mean())
    # print(aftlena.std())
    # print(np.percentile(aftlena, 50))
    # print(np.percentile(aftlena, 95))
    # print(aftlena.max())

    def find_prompt_example(sub_dataset, task_template):
        result_strs = []
        for idx in range(len(sub_dataset)-1):
            examples = sub_dataset[idx]
            input = task_template[0].format(**examples)
            output = task_template[1].format(**examples)
            result_strs.append(input + output + example_separator)
        examples = sub_dataset[-1]
        input = task_template[0].format(**examples)
        output = task_template[1].format(**examples)
        result_strs.append(input + output + x_y_delimiter)
        return result_strs

    def preprocess_function_test(examples, index, task_template, datasets_all):
        num_fewshot_target = 5
        subindexes = sectionhead2indexs[examples["section_header"]]
        subindexes = list(set(subindexes) - set([index]))
        sub_dataset = datasets_all.select(subindexes)
        prompts_fewshot = find_prompt_example(sub_dataset, task_template)
        input = task_template[0].format(**examples)
        src = tokenizer(" ".join(prompts_fewshot) + input, add_special_tokens=True,
                                        is_split_into_words=False)['input_ids']
        idx = 0
        while len(src) > max_source_length:
            idx += 1
            if idx == num_fewshot_target:
                src = tokenizer(input, add_special_tokens=True, truncation=True, max_length=max_source_length,
                                        is_split_into_words=False)['input_ids']
                break
            src = tokenizer(" ".join(prompts_fewshot[idx:]) + input, add_special_tokens=True,
                                        is_split_into_words=False)['input_ids']            
        
        num_fewshot_actual = num_fewshot_target - idx
        
        sent = src + [tokenizer.bos_token_id]
        return {
            "input_ids": torch.tensor(sent, dtype=torch.long),
            "src_sent": torch.tensor(sent, dtype=torch.long),
            "num_fewshot_actual": num_fewshot_actual
        }

    def preprocess_function_train(examples, index, task_template, datasets_all):
        num_fewshot_target = 5
        subindexes = sectionhead2indexs[examples["section_header"]]
        subindexes = list(set(subindexes) - set([index]))
        sub_dataset = datasets_all.select(subindexes)
        prompts_fewshot = find_prompt_example(sub_dataset, task_template)
        input = task_template[0].format(**examples)
        output = task_template[1].format(**examples)

        tgt = tokenizer(output, add_special_tokens=True, truncation=True, max_length=max_target_length,
                                        is_split_into_words=False)['input_ids']
        max_source_length = max_total_len - len(tgt) - 2
        src = tokenizer(" ".join(prompts_fewshot) + input, add_special_tokens=True,
                                        is_split_into_words=False)['input_ids']
        idx = 0
        while len(src) > max_source_length:
            idx += 1
            if idx == num_fewshot_target:
                src = tokenizer(input, add_special_tokens=True, truncation=True, max_length=max_source_length,
                                        is_split_into_words=False)['input_ids']
                break
            src = tokenizer(" ".join(prompts_fewshot[idx:]) + input, add_special_tokens=True,
                                        is_split_into_words=False)['input_ids'] 
        
        num_fewshot_actual = num_fewshot_target - idx

        sent = src + [tokenizer.bos_token_id] + tgt + [tokenizer.eos_token_id]
        sep_idx = sent.index(tokenizer.bos_token_id) + 1
        label = copy.deepcopy(sent)
        label[:sep_idx] = [-100] * sep_idx
        src_sent = sent[:sep_idx - 1]
        tgt_sent = sent[sep_idx - 1:]

        return {"input_ids": torch.tensor(sent, dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long),
                "src_sent": torch.tensor(src_sent, dtype=torch.long),
                "tgt_sent": torch.tensor(tgt_sent, dtype=torch.long),
                "num_fewshot_actual": num_fewshot_actual
        }

    if training_args.do_predict:
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):     
            pred_datasets = []
            task = "mediqa_predict"
            hf_dataset = datasets_bytask[task]
            template = PATTERNS[task]
            for a_template in template:
                # train
                pred_dataset = hf_dataset["test"]
                pred_dataset = pred_dataset.map(
                    preprocess_function_test,
                    with_indices=True,
                    batched=False,
                    num_proc=data_args.preprocessing_num_workers,
                    fn_kwargs={"task_template": a_template, "datasets_all":datasets_bytask["mediqa_finetune"]["train"]}
                )
                pred_datasets.append(pred_dataset)
            predict_dataset = concatenate_datasets(pred_datasets)
            del pred_datasets
    del datasets_bytask

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSumLanguageModeling(tokenizer=tokenizer, label_pad_token_id=label_pad_token_id)


    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions_all = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions_all = [pred.strip() for pred in predictions_all]
                save_path_a = os.path.join(training_args.output_dir, "generated_all")
                if not os.path.exists(save_path_a):
                    os.makedirs(save_path_a)
                for ind, pred in enumerate(predictions_all):
                    output_prediction_file = os.path.join(save_path_a, f"generated_all_{ind:04d}.txt")
                    with open(output_prediction_file, "w") as writer:
                        writer.write(pred)
                
                predictions = tokenizer.batch_decode(
                    predict_dataset["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                save_path_a = os.path.join(training_args.output_dir, "generated_predictions")
                if not os.path.exists(save_path_a):
                    os.makedirs(save_path_a)
                output_prediction_file = os.path.join(save_path_a, f"generated_predictions.csv")
                tosave_id = []
                tosave_header = []
                tosave_note = []
                for ind, pred in enumerate(predictions):
                    tosave_id.append(predict_dataset[ind]["id"])
                    tosave_header.append(predict_dataset[ind]["section_header"])
                    tosave_note.append(predictions_all[ind][len(pred):])
                df = pd.DataFrame({"TestID": tosave_id, "SystemOutput1": tosave_header, "SystemOutput2":tosave_note})
                df.to_csv(output_prediction_file, index=False)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang


    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()