# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

# ruff: noqa:F401,E402
import time
from typing import Callable
import ezpz

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm.Barrier()
python_start_time = time.time()

import os
from rich import print
import torch
import math
from functools import partial
from megatron import get_args

# from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import (
    average_losses_across_data_parallel_group,
    update_rotary_pos_emb,
)
from megatron.arguments import core_transformer_config_from_args

# from megatron.utils import Profile, PerfTrace

import logging

import deepspeed
from deepspeed.runtime.utils import see_memory_usage

# from deepspeed.accelerator.real_accelerator import get_accelerator
import subprocess
import wandb

from torch import nn
import torch.nn.functional as F
dt_imports = time.time() - python_start_time
t0_setup = time.time()

# ---- [SETUP COMMS] ------------------------
# if str(os.environ.get('LAUNCH_CMD', 'mpich')).lower() == 'mpich':
RANK = ezpz.setup_torch(backend="deepspeed")  # , timeout=7200)
dt_setup = time.time() - t0_setup
# else:
#     RANK = ezpz.get_rank()
WORLD_SIZE = ezpz.get_world_size()
LOCAL_RANK = ezpz.get_local_rank()
DEVICE_TYPE = ezpz.get_torch_device_type()
if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)

log = logging.getLogger(__name__)
LOG_LEVEL = str(os.environ.get("LOG_LEVEL", "INFO")).upper()
# set logging level to "INFO" on RANK == 0, "CRITICAL" on all other ranks
log.setLevel(LOG_LEVEL) if RANK == 0 else log.setLevel("CRITICAL")

log.info(f"Import python modules in {dt_imports} seconds")
log.info(f"ezpz.setup_torch time: {dt_setup} seconds")

# ---- [SETUP WANDB FROM RANK 0] --------------
WANDB_MODE = os.environ.get("WANDB_MODE", None)
DISABLE_WANDB = WANDB_MODE is not None and str(WANDB_MODE).lower() == "disabled"
if RANK == 0 and not DISABLE_WANDB:
    project_name = os.environ.get(
        "WB_PROJECT",  # look for WB_PROJECT in env
        os.environ.get("WANDB_PROJECT", "AuroraGPT"),  # look for WANDB_PROJECT in env
    )
    log.info(f"Setting up W&B from: {RANK} with {project_name}")
    _ = ezpz.setup_wandb(project_name=project_name)


@ezpz.timeitlogit(rank=RANK)
def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    log.info("building GPT model ...")
    see_memory_usage("Before Building Model", force=True)
    args = get_args()
    assert args is not None
    config = core_transformer_config_from_args(args)
    # if RANK == 0:
    #     git_ds_info()
    if hasattr(mpu, "get_sequence_data_parallel_group"):
        dpg = mpu.get_sequence_data_parallel_group()
    elif hasattr(mpu, "get_data_parallel_group"):
        dpg = mpu.get_data_parallel_group()
    else:
        dpg = None
    deepspeed_zero_init = deepspeed.zero.Init
    if args.use_mics:
        deepspeed_zero_init = deepspeed.zero.MiCS_Init
    with deepspeed_zero_init(
        data_parallel_group=dpg,
        remote_device=(None if args.remote_device == "none" else args.remote_device),
        config_dict_or_path=args.deepspeed_config_dict,
        enabled=args.zero_stage == 3,
        mpu=mpu,
    ):
        if args.deepspeed and not args.no_pipeline_parallel:
            model = GPTModelPipe(config=config, num_tokentypes=0, parallel_output=True)
            # This is a hack to give us a reference to
            # get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe
            # Precompute the attention mask and store it in args.
            # This avoids having to pipeline it
            # as an activation during training.
            # The mask is constant, and thus we can reuse it.
            attention_mask = torch.tril(
                torch.ones(
                    (1, args.seq_length, args.seq_length),
                    device=DEVICE_TYPE,
                )
            ).view(1, 1, args.seq_length, args.seq_length)
            # Convert attention mask to binary:
            attention_mask = attention_mask < 0.5
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()
            # Attention mask must be bool.
            args.attn_mask = attention_mask.to(torch.bool)
            # For pretraining, since sequence length is fixed,
            # cache rotary embedding in args, to avoid communicating around
            if args.use_rotary_position_embeddings:
                update_rotary_pos_emb(args.seq_length)
        else:
            model = GPTModel(
                config=config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
            )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(80 * "-")
    log.info(f"Number of parameters in model: {num_params}")
    log.info(80 * "-")
    see_memory_usage("After Building Model", force=True)
    if wandb is not None and getattr(wandb, "run", None) is not None:
        assert wandb.run is not None
        tbdir = args.tensorboard_dir
        # tbdir = args.getattr('tensorboard_dir', None)
        if tbdir is not None:
            try:
                log.info(f"Patching tensorboard from {tbdir}")
                wandb.tensorboard.patch(root_logdir=tbdir)
            except ValueError as exc:
                log.exception(exc)
                log.warning("Continuing without patching tensorboard!")
        wandb.run.config.update({"num_params": num_params})
        if "args" not in wandb.run.config:
            log.info(
                f"Updating WandB run.config: [{wandb.run.name}]({wandb.run.get_url()})"
            )
            try:
                wandb.run.config.update({"args": dict(sorted(vars(args).items()))})
            except Exception:
                log.error('Unable to `wandb.run.config.update({"args": vars(args)})`')
        # try:
        #     wandb.run.watch(
        #         model,
        #         log='all',
        #         log_graph=True,
        #     )
        # except Exception:
        #     pass
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()
    assert args is not None and tokenizer is not None
    # Items and their type.
    keys = ["text"]
    datatype = torch.int64
    data = next(data_iterator) if data_iterator is not None else None

    if (
        args.iteration < 10
        and RANK == 0
        and os.environ.get("DUMP_TOKENS", None)
        and data is not None
    ):
        log.info(f"{args.iteration=}: {data['text'][:10]=}")
    # # Broadcast data.
    # if data_iterator is not None:
    #     data = next(data_iterator)
    # else:
    #     data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens_ = data_b["text"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    # Get the masks and postition ids.
    skip_mask = args.use_flash_attn or args.use_flash_attn_triton
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        skip_mask,
    )
    # For DS's sequence parallel
    seq_parallel_world_size = mpu.get_sequence_parallel_world_size()
    seq_parallel_world_rank = mpu.get_sequence_parallel_rank()
    # For Megatron's sequence parallel
    if args.sequence_parallel:
        seq_parallel_world_size = mpu.get_tensor_model_parallel_world_size()
        seq_parallel_world_rank = mpu.get_tensor_model_parallel_rank()
    seq_length = tokens.size(1)
    assert seq_length % seq_parallel_world_size == 0
    sub_seq_length = seq_length // seq_parallel_world_size
    sub_seq_start = seq_parallel_world_rank * sub_seq_length
    sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_length
    tokens = tokens[:, sub_seq_start:sub_seq_end]
    position_ids = position_ids[:, sub_seq_start:sub_seq_end]
    # For DS's sequence parallel
    if mpu.get_sequence_parallel_world_size() > 1:
        labels = labels[:, sub_seq_start:sub_seq_end]
    return tokens, labels, loss_mask, attention_mask, position_ids


def data_post_process(data, data_sampler_state_dict):
    args = get_args()
    assert args is not None
    if args.data_efficiency_curriculum_learning:
        if "seqlen_truncate" in data_sampler_state_dict["current_difficulties"]:
            args.data_efficiency_curriculum_learning_seqlen_type = "seqlen_truncate"
            current_seqlen = data_sampler_state_dict["current_difficulties"][
                "seqlen_truncate"
            ]
            if current_seqlen < args.seq_length:
                data["text"] = data["text"][:, : (current_seqlen + 1)].contiguous()
        elif "seqlen_reshape" in data_sampler_state_dict["current_difficulties"]:
            args.data_efficiency_curriculum_learning_seqlen_type = "seqlen_reshape"
            current_seqlen = data_sampler_state_dict["current_difficulties"][
                "seqlen_reshape"
            ]
            if current_seqlen < args.seq_length:
                orig_num_token = torch.numel(data["text"])
                reshape_len = (data["text"].size()[1] // (current_seqlen + 1)) * (
                    current_seqlen + 1
                )
                data["text"] = torch.cat(
                    (
                        data["text"][:, :reshape_len]
                        .contiguous()
                        .view(-1, current_seqlen + 1),
                        data["text"][:, -(current_seqlen + 1) :],
                    ),
                    0,
                ).contiguous()
                num_row = math.ceil(orig_num_token / (current_seqlen + 1))
                num_row = min(num_row, data["text"].size()[0])
                if num_row > 1 and num_row % 2 != 0:
                    num_row -= 1
                data["text"] = data["text"][:num_row, :].contiguous()
        else:
            args.data_efficiency_curriculum_learning_seqlen_type = None
    return data


def get_batch_pipe(data):
    """
    Modification of `get_batch` to work on `next(data_iterator)`
    instead of `data_iterator`
    """
    args = get_args()
    tokenizer = get_tokenizer()
    assert args is not None
    # Items and their type.
    keys = ["text"]
    datatype = torch.int64
    # Broadcast data.
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens_ = data_b["text"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )
    if args.curriculum_learning_legacy and args.curriculum_seqlen < tokens.size()[1]:
        # seqlen-based curriculum learning
        # tokens, position_ids, labels, loss_mask
        # have size [batch size, seqlen]
        tokens = tokens[:, : args.curriculum_seqlen].contiguous()
        position_ids = position_ids[:, : args.curriculum_seqlen].contiguous()
        if labels is not None:
            labels = labels[:, : args.curriculum_seqlen].contiguous()
        loss_mask = loss_mask[:, : args.curriculum_seqlen].contiguous()
    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def loss_func(loss_mask, moe_loss, mos_loss, output_tensor):
    args = get_args()
    assert args is not None
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    if args.mos or args.kd:
        # assert max(args.num_experts) >= 1
        loss = loss + moe_loss + mos_loss
        if args.mos:
            return loss, {
                "total loss": loss,
                "lm loss": averaged_loss[0],
                "moe loss": moe_loss,
                "mos loss": mos_loss,
            }
        elif args.kd:
            return loss, {
                "total loss": loss,
                "lm loss": averaged_loss[0],
                "moe loss": moe_loss,
                "kd loss": mos_loss,
            }
        log.info(
            f">>> total loss: {loss}, "
            f"lm loss {averaged_loss[0]}, "
            f"kd loss {mos_loss}"
        )
    else:
        if max(args.num_experts) <= 1:
            return loss, {"lm loss": averaged_loss[0]}
        loss = loss + moe_loss
        return loss, {"lm loss": averaged_loss[0], "moe loss": moe_loss}


def calculate_mos_loss(
    args, stu_output, teacher_model, tokens, position_ids, attention_mask
):
    mos_loss = 0
    alpha = args.kd_alpha_ce
    beta = args.kd_beta_ce
    kd_temp = args.kd_temp
    if teacher_model:
        with torch.no_grad():
            if (
                args.curriculum_learning_legacy
                and args.curriculum_seqlen < args.seq_length
            ):
                assert args.curriculum_seqlen is not None
                curriculum_seqlen = args.curriculum_seqlen
                tokens = tokens[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                csl = curriculum_seqlen
                attention_mask = attention_mask[:, :, :csl, :csl].contiguous()
                # No need to truncate labels
                # as we do not need it for the teacher logits
            tea_output, tea_other_losses = teacher_model(
                tokens, position_ids, attention_mask
            )
            assert stu_output.size() == tea_output.size(), (
                "teacher and student output should match in size. "
                f"Student: {stu_output.size()}, "
                f"Teacher: {tea_output.size()}, "
                f"CL seq length {args.curriculum_seqlen}"
            )
        student_logits = F.log_softmax(stu_output / kd_temp, dim=2)
        # The target logits is expected to be probabilities.
        # If we use log_softmax,
        # then we need to set target_log to true
        # when initializing the KLDivLoss.
        tea_logits = F.softmax(tea_output / kd_temp, dim=2)
        mos_loss = (
            kd_temp
            * kd_temp
            * nn.KLDivLoss(reduction="batchmean")(student_logits, tea_logits)
        )
        mos_loss = mos_loss.div(args.seq_length) * beta
    return mos_loss


def forward_step(data_iterator, model) -> tuple[torch.Tensor | None, Callable]:
    """Forward step."""
    args = get_args()
    timers = get_timers()
    assert args is not None
    assert timers is not None
    # Get the batch.
    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()
    if args.data_efficiency_curriculum_learning:
        args.curriculum_seqlen = tokens.size()[1]
        if hasattr(args, "data_efficiency_curriculum_learning_seqlen_type") and (
            args.data_efficiency_curriculum_learning_seqlen_type == "seqlen_reshape"
        ):
            args.data_efficiency_curriculum_learning_numel = torch.numel(tokens)
    stu_output = None
    if args.mos or args.kd:
        # The forward func can return either the loss or the logits,
        # depending on whether passing in the labels or not.
        stu_output, other_losses = model(tokens, position_ids, attention_mask)
        if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
            assert args.curriculum_seqlen is not None
            labels = labels[:, : args.curriculum_seqlen].contiguous()
        output_tensor = tensor_parallel.vocab_parallel_cross_entropy(
            stu_output.contiguous().float(), labels
        )
    else:
        output_tensor, other_losses = model(
            tokens, position_ids, attention_mask, labels=labels
        )
    if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
        loss_mask = loss_mask[:, : args.curriculum_seqlen].contiguous()

    moe_losses = []
    for moe_loss in other_losses:
        if moe_loss is not None:
            moe_losses.append(moe_loss)
    moe_loss = sum(moe_losses) * args.moe_loss_coeff

    mos_loss = 0
    if args.mos or args.kd:
        assert model.training
        if args.teacher_forward and args.teacher_model is not None:
            mos_loss = calculate_mos_loss(
                args,
                stu_output,
                args.teacher_model[0],
                tokens,
                position_ids,
                attention_mask,
            )

    # Output_tensor stores the standard loss,
    # loss_func calculates the total loss.
    return output_tensor, partial(loss_func, loss_mask, moe_loss, mos_loss)


@ezpz.timeitlogit(rank=RANK)
def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    t0 = time.perf_counter()
    args = get_args()
    assert args is not None
    # from ezpz.profile import get_context_manager
    # cm = get_context_manager(rank=RANK, outdir=args.save)
    # with cm:
    log.info("> building train, validation, and test datasets for GPT ...")
    files = []
    if args.data_file_list is not None:
        log.info(f"Reading datasets from {args.data_file_list}")
        # [!NOTE]:
        # - We expect each line of args.data_file_list to be of the form:
        #       `weight /path/tp/data_text_document corpus`
        #   where:
        #     - `weight` is the relative weight of that document
        #        across all documents (i.e. lines in `args.data_file_list`)
        #     - `/path/to/data_text_document` is the path to the text document
        #     - `corpus` is the corpus (~ source, can be made up) where that
        #        document came from (i.e. `books`, `arxiv`, etc.)
        with open(args.data_file_list, "r") as flist:
            for f in flist.readlines():
                if len(f.strip()) != 0:
                    try:
                        w, fname, c = f.split()
                    except Exception as exc:
                        log.exception(exc)
                        raise Exception(
                            "Please provide the file list as "
                            "'weight, filename, corpus'"
                        )
                    if fname.find(".bin") != -1:
                        fname = fname.split(".bin")[0]
                    files.extend([float(w), fname, c])  # weight  # filename  # corpus
    elif len(args.data_path) == 1 and os.path.isdir(args.data_path[0]):
        path = args.data_path[0] + "/"
        for f in os.listdir(path):
            if os.path.isfile(path + f) and f.find(".bin") != -1:
                files.append(1)
                files.append(path + f.split(".bin")[0])
    else:
        files = args.data_path

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=files,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path,
    )
    dt = time.perf_counter_ns() - t0
    log.info(f"> finished creating GPT datasets. Took: {dt:.5f}s")
    return train_ds, valid_ds, test_ds


def command_exists(cmd):
    result = subprocess.Popen(f"type {cmd}", stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0


def git_ds_info():
    if RANK != 0:
        return
    from deepspeed.env_report import main as ds_report

    ds_report()

    # Write out version/git info
    git_hash_cmd = "git rev-parse --short HEAD"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
    if command_exists("git"):
        try:
            result = subprocess.check_output(git_hash_cmd, shell=True)
            git_hash = result.decode("utf-8").strip()
            result = subprocess.check_output(git_branch_cmd, shell=True)
            git_branch = result.decode("utf-8").strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"
    print(
        f"**** Git info for Megatron: "
        f"git_hash={git_hash} git_branch={git_branch} ****"
    )


def main():
    if os.getenv("TORCH_PROFILER_ENABLE") == "1":
        #  record_function
        from torch.profiler import profile, ProfilerActivity

        try:
            activities = [
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
                ProfilerActivity.XPU,
            ]
        except Exception as exc:
            log.exception(exc)
            log.warning("TORCH PROFILER WARNING: XPU is not supported")
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=activities) as prof:
            model = pretrain(
                train_valid_test_datasets_provider,
                model_provider,
                ModelType.encoder_or_decoder,
                forward_step,
                # args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
                data_post_process=data_post_process,
            )
        args = get_args()
        assert args is not None
        prof.export_chrome_trace(
            f"{args.trace_dir}/torch-trace-{RANK}-of-{WORLD_SIZE}.json"
        )
    else:
        model = pretrain(
            train_valid_test_datasets_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            # args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
            data_post_process=data_post_process,
        )
    # try:
    #     from megatron.text_generation import generate_and_post_process
    #     with torch.autocast(device_type=DEVICE, dtype=args.dtype):
    #         response, _, _, _ = generate_and_post_process(
    #             model,
    #             prompts=[
    #                 "Hello world",
    #                 "Nature is",
    #                 "Turing test comprises",
    #                 "Explain solar eclipse"
    #             ],
    #             tokens_to_generate=32
    #         )
    #     if RANK == 0:
    #         log.info(f'generation completed..\n response:{response}')
    # except ValueError as ve:
    #     log.critical(f'ValueError: {ve}')
    #     pass
    # dist.barrier()
    # model.train()
    return model


if __name__ == "__main__":
    # git_ds_info()
    # pretrain(train_valid_test_datasets_provider,
    #          model_provider,
    #          ModelType.encoder_or_decoder,
    #          forward_step,
    #          args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    #          data_post_process=data_post_process)
    import sys
    import deepspeed.comm as dist

    model = main()
    dist.log_summary()
    if wandb.run is not None:
        print(f"wandb.run.name: {wandb.run.name}")
        print(f"wandb.run.url: {wandb.run.url}")
        wandb.finish()
    sys.exit(0)
