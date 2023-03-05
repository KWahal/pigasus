import os

from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
from transformers import PegasusConfig, PegasusModel
import torch
from datasets import load_dataset
from tqdm import tqdm
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
import tensorflow as tf


def tune_transformer(num_samples=8, gpus_per_trial=0, smoke_test=False):
    # Change these as needed.
    model_name = "google/pegasus-large"

    # Download and cache tokenizer, model, and features
    print("Downloading and caching Tokenizer")
    tokenizer = PegasusTokenizer.from_pretrained(model_name)

    # Triggers tokenizer download to cache
    print("Downloading and caching pre-trained model")
    def get_model():
        return PegasusForConditionalGeneration.from_pretrained(model_name)

    # Download data.
    dataset = load_dataset('billsum', split="ca_test")

    # doc_data = dataset["text"] 
    train_data = dataset[:865] + dataset[1051:] # combine train + val
    test_data = dataset[865:1051]

    print(train_data[0:5])

    # sum_data = dataset["summary"]
    # train_label = sum_data[:865] + sum_data[1051:] # combine train + val
    # test_label = sum_data[865:1051]

    # TODO: process w tokenizer??? for data ^^

    # TODO: find pegasus trainargs
    training_args = TrainingArguments(
        output_dir=".",
        learning_rate=1e-5,  # config
        do_train=True,
        do_eval=True,
        no_cuda=gpus_per_trial <= 0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=2,  # config
        max_steps=-1,
        per_device_train_batch_size=16,  # config
        per_device_eval_batch_size=16,  # config
        warmup_steps=0,
        weight_decay=0.1,  # config
        logging_dir="./logs",
        skip_memory_metrics=True,
        report_to="none",
    )

    # Takes in tokenized generated summary and tokenized actual summary, then computes cross entropy
    def compute_metrics(p: EvalPrediction):
        loss = tf.losses.softmax_cross_entropy(p.predictions, p.label_ids)
        metrices = glue_compute_metrics(task_name, preds, p.lanbels_ids)
        return loss

    trainer = Trainer(
        model_init=get_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics(), # TODO: complete this func
    )

    tune_config = {
        # "per_device_train_batch_size": 32,
        # "per_device_eval_batch_size": 32,
        # "num_train_epochs": tune.choice([2, 3, 4, 5]),
        # "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
        'learning_rate': tune.grid_search([]),
        'label_smoothing_factor': 0.1,
        'num_train_epochs': tune.grid_search([90000, 100000, 115000, 120000]), #Number of steps???
        'per_gpu_train_batch_size': tune.grid_search([256, 512]) 
    }

    # scheduler = PopulationBasedTraining(
    #     time_attr="training_iteration",
    #     metric="eval_acc",
    #     mode="max",
    #     perturbation_interval=1,
    #     hyperparam_mutations={
    #         "weight_decay": tune.uniform(0.0, 0.3),
    #         "learning_rate": tune.uniform(1e-5, 5e-5),
    #         "per_device_train_batch_size": [16, 32, 64],
    #     },
    # )

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=["eval_acc", "eval_loss", "epoch", "training_iteration"],
    )

    trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        n_trials=num_samples,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        # scheduler=scheduler,
        keep_checkpoints_num=1,
        checkpoint_score_attr="training_iteration",
        stop={"training_iteration": 1} if smoke_test else None,
        progress_reporter=reporter,
        local_dir="~/ray_results/",
        name="tune_transformer_pbt",
        log_to_file=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        default=True,
        action="store_true",
        help="Finish quickly for testing",
    )
    args, _ = parser.parse_known_args()

    ray.init()

    if args.smoke_test:
        tune_transformer(num_samples=1, gpus_per_trial=0, smoke_test=True)
    else:
        # You can change the number of GPUs here:
        tune_transformer(num_samples=8, gpus_per_trial=1)