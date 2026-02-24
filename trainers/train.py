import sys

import os

# path problem.
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import pandas as pd
import collections
import argparse
import warnings
import sklearn.exceptions
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter


from utils import fix_randomness, starting_logs, AverageMeter
from abstract_trainer import AbstractTrainer

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # Optional K-Fold settings (off by default for backward compatibility).
        self.use_kfold = getattr(args, "use_kfold", False)
        self.num_folds = getattr(args, "num_folds", 5)
        self.kfold_shuffle = getattr(args, "kfold_shuffle", True)
        self.kfold_seed = getattr(args, "kfold_seed", 0)
        if self.use_kfold and self.num_folds < 2:
            raise ValueError("num_folds must be >= 2 when --use_kfold is enabled.")

        # TensorBoard logging settings.
        self.use_tensorboard = getattr(args, "use_tensorboard", True)
        self.tensorboard_dir = getattr(args, "tensorboard_dir", None)
        self.tb_writer = None
        self.tb_step = 0

        # Logging
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description,
                                        f"{self.run_description}")
        os.makedirs(self.exp_log_dir, exist_ok=True)

    def _maybe_init_tensorboard(self):
        if not self.use_tensorboard or self.tb_writer is not None:
            return

        if self.tensorboard_dir:
            log_dir = os.path.join(self.home_path, self.tensorboard_dir, self.experiment_description, self.run_description)
        else:
            log_dir = os.path.join(self.exp_log_dir, "tensorboard")

        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir)

    def _log_tensorboard_step(self, scenario, run_value, metrics, risks, fold_id=None):
        if self.tb_writer is None:
            return

        metric_names = ["mse", "mae", "r2"] if self.task == "regression" else ["acc", "f1_score", "auroc"]
        tag_prefix = f"{scenario}/run_{run_value}" if fold_id is None else f"{scenario}/run_{run_value}/fold_{fold_id}"

        for name, value in zip(metric_names, metrics):
            self.tb_writer.add_scalar(f"{tag_prefix}/metrics/{name}", float(value), self.tb_step)
        self.tb_writer.add_scalar(f"{tag_prefix}/risk/src_risk", float(risks[0]), self.tb_step)
        self.tb_writer.add_scalar(f"{tag_prefix}/risk/trg_risk", float(risks[1]), self.tb_step)

        for key, meter in self.pre_loss_avg_meters.items():
            self.tb_writer.add_scalar(f"{tag_prefix}/pretrain/{key}", float(meter.avg), self.tb_step)
        for key, meter in self.loss_avg_meters.items():
            self.tb_writer.add_scalar(f"{tag_prefix}/adapt/{key}", float(meter.avg), self.tb_step)

        self.tb_step += 1

    def _log_tensorboard_summary(self, table_results, table_risks, results_columns, risks_columns):
        if self.tb_writer is None:
            return

        for column in results_columns[2:]:
            numeric_col = pd.to_numeric(table_results[column], errors="coerce")
            self.tb_writer.add_scalar(f"summary/metrics/{column}_mean", float(numeric_col.mean()), self.tb_step)
            self.tb_writer.add_scalar(f"summary/metrics/{column}_std", float(numeric_col.std()), self.tb_step)
        for column in risks_columns[2:]:
            numeric_col = pd.to_numeric(table_risks[column], errors="coerce")
            self.tb_writer.add_scalar(f"summary/risks/{column}_mean", float(numeric_col.mean()), self.tb_step)
            self.tb_writer.add_scalar(f"summary/risks/{column}_std", float(numeric_col.std()), self.tb_step)

        self.tb_writer.flush()

    def _build_source_fold_loader(self, source_dataset, train_indices):
        return DataLoader(
            dataset=Subset(source_dataset, train_indices),
            batch_size=self.hparams["batch_size"],
            shuffle=self.dataset_configs.shuffle,
            drop_last=self.dataset_configs.drop_last,
            num_workers=4,
            pin_memory=True
        )

    def _get_fold_splits(self, run_id, source_dataset):
        num_samples = len(source_dataset)
        if num_samples < self.num_folds:
            raise ValueError(f"Source train samples ({num_samples}) are fewer than num_folds ({self.num_folds}).")
        random_state = self.kfold_seed + run_id if self.kfold_shuffle else None
        splitter = KFold(n_splits=self.num_folds, shuffle=self.kfold_shuffle, random_state=random_state)
        return splitter.split(range(num_samples))

    def train(self):
        self._maybe_init_tensorboard()

        # table with metrics
        if getattr(self.dataset_configs, 'task', 'classification') == 'regression':
            results_columns = ["scenario", "run", "mse", "mae", "r2"]
        else:
            results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)

        # table with risks
        risks_columns = ["scenario", "run", "src_risk", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)

        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Load data
                self.load_data(src_id, trg_id)

                if not self.use_kfold:
                    # Logging
                    self.logger, self.scenario_log_dir = starting_logs(
                        self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, run_id
                    )
                    # Average meters
                    self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                    self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                    # Train model
                    non_adapted_model, last_adapted_model, best_adapted_model = self.train_model()

                    # Save checkpoint
                    self.save_checkpoint(
                        self.home_path, self.scenario_log_dir, non_adapted_model, last_adapted_model, best_adapted_model
                    )

                    # Calculate risks and metrics
                    metrics = self.calculate_metrics()
                    risks = self.calculate_risks()

                    # Append results to tables
                    scenario = f"{src_id}_to_{trg_id}"
                    table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                    table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)
                    self._log_tensorboard_step(scenario, run_id, metrics, risks)
                else:
                    source_dataset = self.src_train_dl.dataset
                    fold_splits = self._get_fold_splits(run_id, source_dataset)

                    for fold_id, (train_indices, _) in enumerate(fold_splits):
                        fold_run_id = f"{run_id}_fold_{fold_id}"

                        # Logging
                        self.logger, self.scenario_log_dir = starting_logs(
                            self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, fold_run_id
                        )
                        self.logger.debug(
                            f"K-Fold enabled: fold {fold_id + 1}/{self.num_folds}, "
                            f"train_samples={len(train_indices)}, total_samples={len(source_dataset)}"
                        )

                        # Replace source train loader with the current fold split.
                        self.src_train_dl = self._build_source_fold_loader(source_dataset, train_indices)

                        # Average meters
                        self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                        self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                        # Train model
                        non_adapted_model, last_adapted_model, best_adapted_model = self.train_model()

                        # Save checkpoint
                        self.save_checkpoint(
                            self.home_path, self.scenario_log_dir, non_adapted_model, last_adapted_model, best_adapted_model
                        )

                        # Calculate risks and metrics
                        metrics = self.calculate_metrics()
                        risks = self.calculate_risks()

                        # Keep table schema unchanged by encoding fold into run id.
                        scenario = f"{src_id}_to_{trg_id}"
                        table_results = self.append_results_to_tables(table_results, scenario, fold_run_id, metrics)
                        table_risks = self.append_results_to_tables(table_risks, scenario, fold_run_id, risks)
                        self._log_tensorboard_step(scenario, run_id, metrics, risks, fold_id=fold_id)

        self._log_tensorboard_summary(table_results, table_risks, results_columns, risks_columns)

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, results_columns)
        table_risks = self.add_mean_std_table(table_risks, risks_columns)

        # Save tables to file
        self.save_tables_to_file(table_results, 'results')
        self.save_tables_to_file(table_risks, 'risks')


if __name__ == "__main__":
    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='experiments_logs', type=str,
                        help='Directory containing all experiments')
    parser.add_argument('--run_description', default=None, type=str, help='Description of run, if none, DA method name will be used')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='TERSE', type=str, help='Methods Settings.')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'./data', type=str, help='Path containing dataset')
    parser.add_argument('--dataset', default='HAR', type=str, help='Dataset Settings, such as HAR.')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='TemporalSpatialNN_new', type=str, help='Spatial-temporal feature encoder')



    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=3, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cpu", type=str, help='cpu or cuda')
    parser.add_argument('--gpu_id', default=0, type=str, help='gpu id.')
    parser.add_argument('--use_kfold', action='store_true', help='Enable K-Fold over source train split.')
    parser.add_argument('--num_folds', default=5, type=int, help='Number of folds when K-Fold is enabled.')
    parser.add_argument('--kfold_seed', default=0, type=int, help='Base random seed for fold generation.')
    parser.add_argument('--kfold_shuffle', action='store_true', default=True,
                        help='Shuffle source dataset before K-Fold splitting (default: True).')
    parser.add_argument('--no_kfold_shuffle', action='store_false', dest='kfold_shuffle',
                        help='Disable shuffling before K-Fold splitting.')
    parser.add_argument('--use_tensorboard', action='store_true', default=True,
                        help='Enable TensorBoard logging (default: True).')
    parser.add_argument('--no_tensorboard', action='store_false', dest='use_tensorboard',
                        help='Disable TensorBoard logging.')
    parser.add_argument('--tensorboard_dir', default=None, type=str,
                        help='Optional custom TensorBoard log root (relative to project root).')

    args = parser.parse_args()

    trainer = Trainer(args)
    try:
        trainer.train()
    finally:
        if getattr(trainer, "tb_writer", None) is not None:
            trainer.tb_writer.close()
