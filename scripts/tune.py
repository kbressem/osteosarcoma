import argparse
import calendar
import logging
import os
import sys
import time
from datetime import datetime
from warnings import warn

import monai
import optuna
from ignite.engine import Events
from trainlib.trainer import SegmentationTrainer
from trainlib.utils import load_config

parser = argparse.ArgumentParser(description="Train a segmentation model.")
parser.add_argument("--config", type=str, required=True, help="path to the base config file")
parser.add_argument("--debug", action="store_true", required=False, help="run in debug mode")
args = parser.parse_args()

config_fn = args.config

config = load_config(config_fn)
if args.debug:
    config.debug = True
monai.utils.set_determinism(seed=config.seed)

if config.debug:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    config.transforms.prob = 1.0


def replace_key(d: dict, name: str, replacement: str) -> dict:
    "replaces the name of a key in a dict"
    entries = d.pop(name)
    d.update({replacement: entries})


def objective(trial):

    config = load_config(config_fn)
    if args.debug:
        config.debug = True
    # learning rate
    config.optimizer.Adam.lr = trial.suggest_float("learning_rate", 1e-4, 0.01, log=True)

    # weigth decay
    config.optimizer.Adam.weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.1, log=True)

    # replace keys in config... better version of this is coming soon
    # optimizer
    optim = trial.suggest_categorical("optimizer", ["SGD", "Adam", "Novograd"])
    replace_key(config.optimizer, "Adam", optim)

    # loss function
    loss = trial.suggest_categorical("loss_function", ["DiceLoss", "DiceCELoss", "DiceFocalLoss"])
    replace_key(config.loss, "DiceLoss", loss)

    # activation_function_model
    config.model.act = trial.suggest_categorical(
        "activation_function", ["PRELU", "RELU", "MISH", "SWISH"]
    )

    # drop_out
    config.model.dropout = trial.suggest_float("drop_out", 0.0, 0.2)

    # norm type
    config.model.norm = trial.suggest_categorical("norm_type", ["BATCH", "INSTANCE"])

    # augmentation probability
    config.model.dropout = trial.suggest_float("prob_augment", 0.0, 0.3)

    # num_res_units_model
    config.model.num_res_units = trial.suggest_int("num_res_units", 4, 8)

    # scheduler
    scheduler = trial.suggest_categorical("scheduler", ["fit_one_cycle", "None"])

    # input size
    size = trial.suggest_categorical("input_size", [64, 96, 128])
    config.transforms.train.RandCropByPosNegLabeld.spatial_size = [size, size, size]
    config.transforms.train.Resized.spatial_size = [size, size, size]

    # cache_dir
    current_gmt = time.gmtime()
    time_stamp = calendar.timegm(current_gmt)
    time_stamp = str(datetime.fromtimestamp(time_stamp))
    config.data.cache_dir = os.path.join(config.data.cache_dir, time_stamp)

    if not config.overwrite:
        warn("Enable overwrite in config if you intend to run multiple sessions")

    trainer = SegmentationTrainer(
        progress_bar=True,
        early_stopping=True,
        metrics=["MeanDice", "HausdorffDistance", "SurfaceDistance"],
        save_latest_metrics=True,
        config=config,
    )
    pruning_handler = optuna.integration.PyTorchIgnitePruningHandler(
        trial, "val_mean_dice", trainer
    )
    trainer.evaluator.add_event_handler(Events.COMPLETED, pruning_handler)

    if scheduler == "fit_one_cycle":
        trainer.fit_one_cycle()

    trainer.run()
    return trainer.evaluator.state.metrics["val_mean_dice"]


if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        storage="sqlite:///sarcoma_mr.db",
        study_name="sarcoma_tune",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    loaded_study = optuna.load_study(study_name="sarcoma_tune", storage="sqlite:///sarcoma.db")
    with open("best_trial.txt", "a+") as f:
        params = loaded_study.best_params
        f.write(f"val_mean_dice: {loaded_study.best_value}\n")
        for key in params.keys():
            f.write(f"{key}: {params[key]}\n")
        f.write("\n\n")
