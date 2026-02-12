# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import gc
import logging
import os
import sys
import time
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
import xformers.profiler
from omegaconf import OmegaConf
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import lr_scheduler

import wandb
from apps.main.transformer import (
    LMTransformer,
    LMTransformerArgs,
    build_fsdp_grouping_plan,
    get_no_recompute_ops,
    get_num_flop_per_token,
    tp_parallelize,
)
from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint
from lingua.data import (
    DataArgs,
    PackTokensState,
    build_dataloader_from_args,
    init_dataloader_state_from_args,
)
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    check_model_value_range,
    clean_env,
    dist_mean_dict,
    get_device_mesh,
    get_is_master,
    get_world_size,
    init_signal_handler,
    parallelize_model,
    requeue_slurm_job,
    setup_env,
    setup_torch_distributed,
)
from lingua.logger import init_logger
from lingua.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    get_num_params,
)
from lingua.optim import OptimArgs, build_optimizer
from lingua.probe import AutoProbeD
from lingua.profiling import ProfilerArgs, maybe_run_profiler
from lingua.stool import StoolArgs, launch_job
from lingua.tokenizer import build_tokenizer

logger = logging.getLogger()


@dataclass
class EarlyStoppingArgs:
    """Configuration for early stopping based on validation loss."""

    enabled: bool = False
    patience: int = 5
    min_delta: float = 0.0
    metric: str = "nll_per_token"
    mode: str = "min"  # 'min' -> lower is better, 'max' -> higher is better


@dataclass
class EarlyStoppingState:
    """State for early stopping tracker (saved in checkpoints)."""

    best_metric: Optional[float] = None
    steps_without_improvement: int = 0
    should_stop: bool = False
    best_step: int = 0


class EarlyStoppingTracker:
    """Tracks validation metrics and determines when to stop training early."""

    def __init__(self, args: EarlyStoppingArgs):
        self.args = args
        self.best_metric: Optional[float] = None
        self.steps_without_improvement: int = 0
        self.should_stop: bool = False
        self.best_step: int = 0

    def update(self, metrics: Dict[str, Any], current_step: int) -> bool:
        """Update the tracker with new validation metrics.

        Args:
            metrics: Dictionary of validation metrics (can be nested by source)
            current_step: Current training step

        Returns:
            True if training should stop, False otherwise
        """
        if not self.args.enabled:
            return False

        # Extract the monitored metric - average across all validation sources
        metric_values = []
        for source_name, source_metrics in metrics.items():
            if isinstance(source_metrics, dict) and self.args.metric in source_metrics:
                metric_values.append(source_metrics[self.args.metric])

        if not metric_values:
            logger.warning(
                f"Early stopping metric '{self.args.metric}' not found in validation results"
            )
            return False

        current_metric = sum(metric_values) / len(metric_values)

        # Check if this is an improvement
        is_improvement = False
        if self.best_metric is None:
            is_improvement = True
        elif self.args.mode == "min":
            is_improvement = current_metric < (self.best_metric - self.args.min_delta)
        else:  # mode == "max"
            is_improvement = current_metric > (self.best_metric + self.args.min_delta)

        if is_improvement:
            self.best_metric = current_metric
            self.steps_without_improvement = 0
            self.best_step = current_step
            logger.info(
                f"Early stopping: New best {self.args.metric}={current_metric:.6f} at step {current_step}"
            )
        else:
            self.steps_without_improvement += 1
            logger.info(
                f"Early stopping: No improvement for {self.steps_without_improvement}/{self.args.patience} eval steps. "
                f"Best {self.args.metric}={self.best_metric:.6f} at step {self.best_step}"
            )

        if self.steps_without_improvement >= self.args.patience:
            self.should_stop = True
            logger.info(
                f"Early stopping triggered! No improvement for {self.args.patience} eval steps. "
                f"Best {self.args.metric}={self.best_metric:.6f} at step {self.best_step}"
            )
            return True

        return False

    def state_dict(self) -> Dict[str, Any]:
        return {
            "best_metric": self.best_metric,
            "steps_without_improvement": self.steps_without_improvement,
            "should_stop": self.should_stop,
            "best_step": self.best_step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.best_metric = state_dict.get("best_metric")
        self.steps_without_improvement = state_dict.get("steps_without_improvement", 0)
        self.should_stop = state_dict.get("should_stop", False)
        self.best_step = state_dict.get("best_step", 0)


@dataclass
class TrainArgs:
    name: str = "lingua"
    dump_dir: str = "/scratch/jeisner1/ssaigan1/lingua/dump_dir"

    seed: int = 42

    # Number of gradient accumulation steps
    # Total batch size is batch_size*grad_acc_steps
    grad_acc_steps: int = 8

    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None

    # Nb optimizer steps to take
    steps: int = 1000

    data: DataArgs = field(default_factory=DataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: LMTransformerArgs = field(default_factory=LMTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)
    early_stopping: EarlyStoppingArgs = field(default_factory=EarlyStoppingArgs)

    # If set to None, eval is run locally otherwise it launches a new job with the given number of gpus
    async_eval_gpus: Optional[int] = None
    eval: Optional[Any] = None


@dataclass
class TrainState(Stateful):
    step: int  # Nb of steps taken by the optimizer
    acc_step: int  # Nb of accumulation steps done since last optimizer step
    scheduler: lr_scheduler.LambdaLR
    data_loader_state: PackTokensState
    early_stopping_state: Optional[Dict[str, Any]] = (
        None  # State for early stopping tracker
    )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "data_loader_state": self.data_loader_state,
            "scheduler": self.scheduler.state_dict(),
            "early_stopping_state": self.early_stopping_state,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.data_loader_state = PackTokensState(**state_dict["data_loader_state"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.early_stopping_state = state_dict.get("early_stopping_state")


def validate_train_args(args: TrainArgs, output_size: int):
    if args.model.vocab_size < 0:
        logger.info(f"Setting model output size to {output_size}")
        args.model.vocab_size = output_size
    assert args.model.vocab_size == output_size, (
        "Vocab size should be the same as output size"
    )

    assert args.dump_dir, "Dump dir not set"
    logger.info(f"Experiment name: {args.name}")
    logger.info(f"Creating dump dir at: {args.dump_dir}")
    args.dump_dir = os.path.join(args.dump_dir, args.name)
    if args.checkpoint.path is None:
        logger.info(
            f"Setting checkpoint path to {str(Path(args.dump_dir) / 'checkpoints')}"
        )
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    for source in args.data.sources:
        data_path = os.path.join(args.data.root_dir, source)
        assert os.path.exists(data_path), f"{data_path} doesn't exist"
    print(
        f"world size: {get_world_size()}, desired world size: {args.distributed.dp_replicate * args.distributed.dp_shard * args.distributed.tp_size}"
    )
    if (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        != get_world_size()
    ):
        assert get_world_size() % args.distributed.dp_shard == 0
        args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard

        assert args.distributed.dp_replicate % args.distributed.tp_size == 0
        args.distributed.dp_replicate = (
            args.distributed.dp_replicate // args.distributed.tp_size
        )

        logger.warning(
            f"Setting Data Parallel size to {args.distributed.dp_replicate * args.distributed.dp_shard}"
        )
        assert (
            args.distributed.dp_replicate
            * args.distributed.dp_shard
            * args.distributed.tp_size
            == get_world_size()
        )

        if args.distributed.fsdp_type == "no_shard":
            assert (
                args.distributed.dp_shard == 1
                and args.distributed.dp_replicate == get_world_size()
            )

    args.model.max_seqlen = args.data.seq_len

    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    assert args.probe_freq != args.profiling.mem_steps, (
        "Don't profile during probe step"
    )
    assert args.probe_freq != args.profiling.profile_steps, (
        "Don't profile during probe step"
    )

    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name

    if args.probe_freq is not None:
        assert args.distributed.tp_size == 1, (
            "Probing not supported with tensor parallelism"
        )
        assert args.distributed.selective_activation_checkpointing is False, (
            "Probing not supported with selective activation checkpointing"
        )


preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True


def every_n_steps(train_state, freq, acc_step=None, acc_freq=None):
    test = train_state.step % freq == 0
    if acc_step is not None:
        test = test and (train_state.acc_step == acc_step)
    elif acc_freq is not None:
        test = test and ((train_state.acc_step % acc_freq) == 0)
    return test


def train(args: TrainArgs):
    with ExitStack() as context_stack:
        tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        validate_train_args(
            args,
            tokenizer.n_words,
        )
        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")
        init_logger(Path(args.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)  # For handling preemption signals.
        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)
        # store boolean is_master:
        is_master = get_is_master()
        # suppress logs from non-master:
        if not is_master:
            logging.getLogger().setLevel(logging.WARNING)
        logger.info(f"Starting job: {args.name}")
        saved = False
        # build dataloader
        # need dp world size and rank
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if args.distributed.dp_shard > 1:
            dp_rank = (
                dp_rank * world_mesh["dp_shard"].size()
                + world_mesh["dp_shard"].get_local_rank()
            )
            dp_degree *= world_mesh["dp_shard"].size()

        logger.info(f"Running on dp rank : {dp_rank}")
        logger.info(f"Running on dp size : {dp_degree}")

        torch.manual_seed(args.seed)
        logger.info("Building model")

        # Initializing Model in meta device allows us to initialize models much bigger than 1 gpu's memory
        with torch.device("meta"):
            model = LMTransformer(args.model)
        logger.info("Model is built !")

        model_param_count = get_num_params(model)

        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
            tp_parallelize=tp_parallelize,
            no_recompute_ops=get_no_recompute_ops(),
        )

        # Once we shard the model on different gpus we can actually initialize the model
        # First we create empty tensors of the correct shapes
        model = model.to_empty(device="cuda")
        # Then we init the model. Please make sure this function initializes *ALL* parameters
        # and buffers, otherwise you will have random values in the unitialized tensors
        # which will silently fail (give nan gradients for example)

        if args.checkpoint.init_ckpt_path:
            logger.info(f"Loading initial model from {args.checkpoint.init_ckpt_path}")
            load_from_checkpoint(
                args.checkpoint.init_ckpt_path, model, model_key="model"
            )  # Put model_key="" if its directly the model checkpoint
            model.rope_embeddings.reset_parameters()  # For RoPe initialization since it's a buffer it might not be loaded
        else:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.init_weights()
        check_model_value_range(model, range=10.0, std=1.0)

        # log model size

        logger.info(f"Model size: {model_param_count:,} total parameters")

        logger.info("All layers and corresponding parameter counts:")
        for name, param in model.named_parameters():
            logger.info(f"  {name}: {param.numel():,} parameters")

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # build optimizer after apply parallelisms to the model
        optimizer, scheduler = build_optimizer(model, args.optim, args.steps)
        data_loader_state = init_dataloader_state_from_args(
            args.data, dp_rank, dp_degree
        )

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
            early_stopping_state=None,
        )

        checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)

        # Initialize early stopping tracker
        early_stopping_tracker = EarlyStoppingTracker(args.early_stopping)
        if train_state.early_stopping_state is not None:
            early_stopping_tracker.load_state_dict(train_state.early_stopping_state)
            if early_stopping_tracker.should_stop:
                logger.warning(
                    "Early stopping was already triggered before checkpoint, exiting."
                )
                return

        # Either load from latest checkpoint or start from scratch
        if args.probe_freq is not None:
            if get_is_master():
                os.makedirs(Path(args.dump_dir) / "probe", exist_ok=True)
            torch.distributed.barrier()
            probe = AutoProbeD(
                model,
                (
                    Path(args.dump_dir) / "probe" / f"probe.{dp_rank}.jsonl"
                    if (dp_rank % 128 == 0)
                    else None
                ),
            )

        gc.disable()

        # train loop
        model.train()
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )
        data_loader = context_stack.enter_context(
            build_dataloader_from_args(
                args.data,
                state=train_state.data_loader_state,
            )
        )
        torch_profiler = context_stack.enter_context(
            maybe_run_profiler(args.dump_dir, model, args.profiling)
        )
        logger.info(
            "Starting training loop on dataset with sources: "
            + ", ".join(args.data.sources)
        )
        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()
        while train_state.step < args.steps:
            # We constrain train_state.acc_step to be in range 0 to args.grad_acc_steps - 1
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            # get batch
            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            batch, train_state.data_loader_state = next(data_loader)
            batch = torch.tensor(
                batch,
                dtype=torch.long,
            )

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                # we do garbage collection manually otherwise different processes
                # run the GC at different times so they slow down the whole pipeline
                gc.collect()

            input_ids = batch[:, :, 0].cuda()
            labels = batch[:, :, 1].cuda()
            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += input_ids.numel()

            bsz, seqlen = labels.shape

            # forward
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            # This is an automatic probe that will compute statistics
            # of all linears' inputs, weights and outputs
            # along with attention logits and entropy
            # both in forward and backward pass
            if (args.probe_freq is not None) and every_n_steps(
                train_state, args.probe_freq, acc_step=1 % args.grad_acc_steps
            ):
                # Here we do a fake forward and backward pass on a smaller
                # batch size to avoid OOM
                # This assumes the model has no stateful layers (batch norm..)
                assert next(model.parameters()).grad is None, (
                    "Can't probe model if grads are not reset"
                )

                with probe:
                    probe.metadata = {
                        "it": train_state.step,
                        "global_step": train_state.step,
                        "loop": "lingua",
                    }
                    # Non compiled model uses roughly 2x memory in our exps
                    # So we divide bsz by 2 or seqlen by 2
                    probe_bsz = max(1, bsz // 2)
                    probe_seq = seqlen if (bsz // 2 >= 1) else (seqlen // 2)
                    probe_loss = model(
                        input_ids[:probe_bsz, :probe_seq],
                        labels[:probe_bsz, :probe_seq],
                    )
                    probe_loss.backward()
                    # We zero grads to cancel this fake step
                    optimizer.zero_grad()

                assert next(model.parameters()).grad is None, (
                    "Probe model shouldn't have grads at this point"
                )

            loss = model(input_ids, labels)

            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(train_state.acc_step == 0)

            # We scale loss with grad_acc_steps so the gradient is the same
            # regardless of grad_acc_steps
            loss = loss / args.grad_acc_steps
            # backward on scaled loss to create scaled gradients
            loss.backward()
            # For logging we undo that scaling
            loss = loss.detach() * args.grad_acc_steps

            # optimizer step
            grad_norm = -1.0
            if train_state.acc_step == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.optim.clip, foreach=True
                )

                grad_norm = (
                    grad_norm.full_tensor()
                    if isinstance(grad_norm, DTensor)
                    else grad_norm
                ).item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1

            # updates the scale for next iteration
            # training iteration complete
            end_timer.record()

            torch.cuda.synchronize()

            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            # if profiler is active
            if torch_profiler:
                xformers.profiler.step()

            # log metrics
            if every_n_steps(
                train_state,
                args.logging.freq,
                acc_step=None if args.logging.acc_freq else 0,
                acc_freq=args.logging.acc_freq,
            ):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)

                # gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
                total_acc_steps = (
                    args.grad_acc_steps * train_state.step + train_state.acc_step
                )
                tokens_per_gpu = (
                    total_acc_steps * args.data.batch_size * args.data.seq_len
                )
                total_tokens = dp_degree * tokens_per_gpu
                # This is an estimate and the correct values may change
                # if you change the architecture
                # Use xformer's analyze profile trace to get actual measurement
                FLOPS = (
                    get_num_flop_per_token(
                        model_param_count - args.model.vocab_size * args.model.dim,
                        args.model.n_layers,
                        args.model.dim,
                        args.data.seq_len,
                    )
                    * wps
                )
                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "acc_step": train_state.acc_step,
                        # "total_epochs": total_epochs,
                        "speed": {
                            "wps": wps,
                            "FLOPS": FLOPS,
                            "curr_iter_time": curr_iter_time,
                            "data_load_time": data_load_time,
                            "steps_per_sec": 1.0 / curr_iter_time,
                        },
                        "optim": {
                            "grad_norm": grad_norm,
                            "lr": curr_lr,
                            "total_tokens": total_tokens,
                        },
                        # "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )

                to_sync = {}
                to_sync["loss/out"] = loss.item()
                metrics.update(dist_mean_dict(to_sync))

                if get_is_master():
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()
                total_time_estimate = (
                    train_state.step / max(1, train_state.step - 1)
                ) * time_delta
                logger.info(
                    f"step: {train_state.step}"
                    f"  loss: {round(loss.item(), 4):>7}"
                    # f"  total_epochs: {total_epochs:.2f}"
                    # f"  acc: {train_state.acc_step}"
                    f"  est_total_time: {total_time_estimate:.3f}"
                    # f"  grad: {grad_norm:.2e}"
                    f"  steps/sec: {1.0 / curr_iter_time:.3f}"
                    f"  tokens/sec: {wps:.2f}"
                    # f"  iter: {curr_iter_time:.3f}"
                    # f"  data: {data_load_time:.3f}"
                    # f"  lr: {curr_lr:.2e}"
                    # f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  flops: {FLOPS:.2e}"
                    # f"  pow: {gpu_mem_stats.power_draw / 1000} W"
                )

            saved = False
            if every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ) or every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0):
                saved = checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                )

            if args.eval is not None and (
                every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0)
                or every_n_steps(train_state, args.steps, acc_step=0)
            ):
                from apps.main.eval import (
                    EVAL_FOLDER_NAME,
                    EvalArgs,
                    launch_eval,
                )

                eval_args = dataclass_from_dict(EvalArgs, args.eval)

                eval_args.global_step = train_state.step
                latest_ckpt_is_current = (
                    len(checkpoint.existing_saves) > 0
                    and checkpoint.existing_saves[-1].name == f"{train_state.step:010d}"
                )
                if not latest_ckpt_is_current:
                    logger.info(
                        "Saving checkpoint for eval at step "
                        f"{train_state.step} (latest saved step differs or is missing)."
                    )
                    saved = checkpoint.save(
                        model,
                        optimizer,
                        train_state,
                        args,
                        device_mesh=world_mesh,
                    )
                assert len(checkpoint.existing_saves) > 0, (
                    "No checkpoint available to run eval. "
                    "Expected at least one saved checkpoint."
                )
                eval_args.ckpt_dir = str(checkpoint.existing_saves[-1])
                eval_args.dump_dir = str(
                    os.path.join(
                        args.dump_dir,
                        "evals",
                        EVAL_FOLDER_NAME.format(train_state.step),
                    )
                )
                eval_args.metric_log_dir = args.dump_dir
                if args.async_eval_gpus is None:
                    val_results = launch_eval(eval_args)

                    # Check for early stopping (only for synchronous eval)
                    if val_results is not None and args.early_stopping.enabled:
                        should_stop = early_stopping_tracker.update(
                            val_results, train_state.step
                        )
                        # Update train_state with early stopping state for checkpointing
                        train_state.early_stopping_state = (
                            early_stopping_tracker.state_dict()
                        )

                        if should_stop:
                            logger.info(f"Early stopping at step {train_state.step}")
                            # Save checkpoint before stopping
                            if not saved:
                                checkpoint.save(
                                    model,
                                    optimizer,
                                    train_state,
                                    args,
                                    device_mesh=world_mesh,
                                )
                            break
                elif get_is_master():
                    if wandb.run is not None and args.logging.wandb is not None:
                        eval_args.wandb = deepcopy(args.logging.wandb)
                    assert args.async_eval_gpus > 0
                    logger.info(f"Launching evals on {args.async_eval_gpus} gpus")
                    with clean_env():
                        launch_job(
                            StoolArgs(
                                asdict(eval_args),
                                script="apps.main.eval",
                                copy_code=False,
                                nodes=args.async_eval_gpus // 8,
                                qos="lowest",
                            )
                        )

            if preemption_flag["flag"]:
                if not saved:
                    checkpoint.save(
                        model,
                        optimizer,
                        train_state,
                        args,
                        device_mesh=world_mesh,
                    )
                requeue_slurm_job()
                sys.exit(0)

    if not saved:
        checkpoint.save(
            model,
            optimizer,
            train_state,
            args,
            device_mesh=world_mesh,
        )
    gc.collect()


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate TrainArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call train.py with train.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in TrainArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
