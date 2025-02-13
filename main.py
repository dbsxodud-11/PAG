import os
from typing import Literal

import torch
from tap import Tap

from utils import seed
from gfn_trainer import GFNTrainer

# import huggingface_hub
# huggingface_hub.login()

class Argument(Tap):
    baseline: bool = False
    mode: Literal["train", "eval"] = "train"
    # model_name: str = "gpt2"
    lm_name: str = "gpt2"
    # target_model: str = "CompVis/stable-diffusion-v1-4"
    sd_name: str = "CompVis/stable-diffusion-v1-4"
    clip_model: str = "ViT-L/14"
    aes_model: str = "aesthetic/sac+logos+ava1-l14-linearMSE.pth"
    sentence_encoder: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    reward_metric: str = "aes"
    
    sft_ckpt: str = "save/gpt2-sft-position-final/latest"
    save_dir: str = "./save"

    prompt_file: str = "prompts/initial_prompt.jsonl"
    eval_prompt_file: str = "prompts/eval_prompt_lexica.jsonl"
    few_shot_file: str = "none"

    epochs: int = 1
    lr: float = 1e-5
    lr_flow: float = 1e-4
    max_norm: float = 1.0
    weight_decay: float = 0.1

    num_warmup_steps: int = 100
    train_steps: int = 10000
    batch_size: int = 64
    grad_acc_steps: int = 4

    max_len: int = 75
    min_len: int = 15

    load_buffer: bool = False
    buffer_size: int = 5000
    sim_tolerance: float = 1.0
    prioritization: Literal["c_reward", "reward", "none"] = "reward"
    buffer_ckpt: str = ""
    compare: str = "reward"
    metric: Literal["edit", "cosine"] = "cosine"

    dtype: str = "float32"
    seed: int = 42

    eval_period: int = 100
    eval_batch_size: int = 256

    # reward scaling
    beta: float = 0.05
    lm_sched_end: float = 1.0
    lm_sched_start: float = 1.0
    lm_sched_horizon: int = 2000

    # reward temperature
    reward_sched_start: float = 2.0
    reward_sched_end: float = 1.0
    reward_sched_horizon: int = 500

    # sampling temperature
    temp_low: float = 0.5
    temp_high: float = 2.0
    temp_scheduler: str = "linear"
    temp_sched_horizon: int = 5000

    # victim model
    num_images_per_prompt: int = 1
    do_sample: bool = True

    # wandb
    exp_name: str = "debug"
    wandb_project: str = "prompt-adaptation"
    num_threads: int = 4

    # additional parameters
    prefix: str = "Rephrase"
    threshold: float = 0.28
    exploration: str = "beam"
    top_p: float = 0.9
    top_k: int = 0
    bsz: int = 16
    
    # flow reset
    flow_reset: bool = True
    flow_reset_period: int = 2000
    
    # training objective
    loss: str = "fl-db"
    vargrad: bool = False

if __name__ == "__main__":
    args = Argument(explicit_bool=True).parse_args()
    torch.set_num_threads(args.num_threads)
    seed(args.seed)
    
    if args.mode == "train":
        print("train")
        trainer = GFNTrainer(args)
        trainer.train()
