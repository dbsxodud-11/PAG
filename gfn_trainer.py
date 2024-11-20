import os
import clip
import math
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from torchvision.models import inception_v3
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import LCMScheduler, AutoPipelineForText2Image, AutoencoderTiny
from collections import defaultdict
from typing import Dict, List, Union
import wandb

import numpy as np
from tqdm import tqdm
from csv_logger import CsvLogger
from dataset import get_dataloader
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from sentence_transformers import SentenceTransformer
from accelerate import Accelerator

from utils import (CosineRelayBuffer, InfIterator,
                   ReplayBuffer, formatted_dict, AestheticMlp)

class FlowFunction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_z1 = nn.Linear(dim, 256)
        self.proj_z2 = nn.Linear(256, 1)
        
        self.mlp = nn.Linear(dim, 256 * 2)
        
    def forward(self, x, y):
        scale_shift = self.mlp(y)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        x = self.proj_z1(x)
        x = F.relu(x)
        x = scale * x + shift
        x = self.proj_z2(x)
        return x


def avg_pooling(last_hidden, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(last_hidden.size()).float()
    denom = torch.clamp(input_mask_expanded.sum(1), min=1)
    avg_pool = torch.sum(last_hidden * input_mask_expanded, 1) / denom
    return avg_pool

def generate_and_return_z_logprob(model, prompt_ids, prompt_attention_mask,  eos_token_id, temperature, min_len=15, max_len=30, num_samples=1, vargrad=False, exploration="beam", sparsemax=False):
    active_seqs = torch.ones(prompt_ids.size(0)).bool().to(prompt_ids.device)
    actions = prompt_ids.clone()
    state = prompt_ids.clone()
    sum_logpf = torch.zeros(prompt_ids.size(0)).float().to(prompt_ids.device)
    attention_mask = prompt_attention_mask.clone()
    # print(active_seqs.shape, actions.shape, state.shape, sum_logpf.shape, attention_mask.shape)
    
    # repeat the prompt_ids for num_samples times
    active_seqs = active_seqs.repeat(num_samples)
    actions = actions.repeat(num_samples, 1)
    state = state.repeat(num_samples, 1)
    sum_logpf = sum_logpf.repeat(num_samples)
    attention_mask = attention_mask.repeat(num_samples, 1)
    # print(active_seqs.shape, actions.shape, state.shape, sum_logpf.shape, attention_mask.shape)
    
    outputs = model.module(
        state[:, :-1], attention_mask=attention_mask[:, :-1], output_hidden_states=True)
    hidden_states = outputs["hidden_states"][-1]
    past_key_values = outputs["past_key_values"]
    
    for i in range(max_len):
        if i == 0:
            # If past_key_values is used, attention_mask needs to contain the masking strategy that was used for past_key_values.
            # In other words, the attention_mask always has to have the length: len(past_key_values) + len(input_ids)
            # https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Model.forward.attention_mask
            output = model.module(
                state[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True
            )
            last_hidden = output.hidden_states[-1]
            last_hidden = torch.cat([hidden_states, last_hidden], dim=1)
            avg_pool = avg_pooling(last_hidden, prompt_attention_mask)

            if not vargrad:
                log_z = model.proj_z(avg_pool).squeeze(-1)
        else:
            output = model.module(
                state[:, -1:], attention_mask=attention_mask, past_key_values=past_key_values)
        past_key_values = output["past_key_values"]

        logits = output["logits"][:, -1, :]
        if i == 0:
            logits[..., eos_token_id] = -100.0
        # Generate next token
        with torch.no_grad():
            modified_logits = logits.clone()
            
            prob = F.softmax(modified_logits / temperature, dim=-1)
            token_ids = torch.multinomial(prob, num_samples=1)

        # Compute Log Likelihood
        logprob = F.log_softmax(logits, dim=-1)
        logprob = torch.gather(logprob, -1, token_ids).squeeze(-1)
        logprob = torch.where(active_seqs, logprob, torch.zeros_like(logprob))
        sum_logpf = sum_logpf + logprob

        token_ids = torch.where(
            active_seqs.unsqueeze(-1), token_ids, torch.ones_like(token_ids) * eos_token_id)

        # update action, state, mask
        masks = torch.where(active_seqs.unsqueeze(-1),
                            torch.ones_like(token_ids), torch.zeros_like(token_ids))
        attention_mask = torch.cat([attention_mask.long(), masks], dim=1)
        actions = torch.cat([actions, token_ids], dim=1)
        state = torch.cat([actions, token_ids], dim=1)

        # check if all sequences have generated eos
        active_seqs = active_seqs * (token_ids != eos_token_id).squeeze(-1)
        if torch.all(~active_seqs):
            break
    # add EOS token to penalize incomplete sentences.
    eos_tokens = torch.ones(actions.size(
        0), dtype=torch.long, device=actions.device) * eos_token_id
    actions = torch.cat([actions, eos_tokens.unsqueeze(1)], dim=1)

    if vargrad:
        results = {"actions": actions, "sum_logpf": sum_logpf, "log_z": -sum_logpf}
    else:
        results = {"actions": actions, "sum_logpf": sum_logpf, "log_z": log_z}
    return results

def generate_and_return_flow_logprob(model, prompt_ids, prompt_attention_mask, eos_token_id, temperature, min_len=15, max_len=30, num_samples=1, vargrad=False, exploration="beam", sparsemax=False, proj_z=None):
    active_seqs = torch.ones(prompt_ids.size(0)).bool().to(prompt_ids.device)
    actions = prompt_ids.clone()
    state = prompt_ids.clone()
    attention_mask = prompt_attention_mask.clone()
    
    # repeat the prompt_ids for num_samples times
    active_seqs = active_seqs.repeat(num_samples)
    actions = actions.repeat(num_samples, 1)
    state = state.repeat(num_samples, 1)
    attention_mask = attention_mask.repeat(num_samples, 1)
    
    log_fs = []
    log_pfs = []
    outputs = model.module(
        state[:, :-1], attention_mask=attention_mask[:, :-1], output_hidden_states=True)
    hidden_states = outputs["hidden_states"][-1]
    past_key_values = outputs["past_key_values"]
    
    hidden_states_orig = torch.sum(hidden_states, dim=1).detach()
    for i in range(max_len):
        # If past_key_values is used, attention_mask needs to contain the masking strategy that was used for past_key_values.
        # In other words, the attention_mask always has to have the length: len(past_key_values) + len(input_ids)
        # https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Model.forward.attention_mask
        output = model.module(
            state[:, -1:],
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True
        )
        last_hidden = torch.sum(output.hidden_states[-1], dim=1).detach()
        log_f = proj_z(last_hidden, hidden_states_orig).squeeze(-1)
        log_f = torch.where(active_seqs, log_f, torch.zeros_like(log_f))
        log_fs.append(log_f)
        # else:
        #     output = model(
        #         state[:, -1:], attention_mask=attention_mask, past_key_values=past_key_values)
        past_key_values = output["past_key_values"]

        logits = output["logits"][:, -1, :]
        if i == 0:
            logits[..., eos_token_id] = -100.0
        with torch.no_grad():
            modified_logits = logits.clone()
            
            prob = F.softmax(modified_logits / temperature, dim=-1)
            token_ids = torch.multinomial(prob, num_samples=1)
                
        logprob = F.log_softmax(logits, dim=-1)
        logprob = torch.gather(logprob, -1, token_ids).squeeze(-1)
        logprob = torch.where(active_seqs, logprob, torch.zeros_like(logprob))
        log_pfs.append(logprob)

        token_ids = torch.where(
            active_seqs.unsqueeze(-1), token_ids, torch.ones_like(token_ids) * eos_token_id)

        # update action, state, mask
        masks = torch.where(active_seqs.unsqueeze(-1),
                            torch.ones_like(token_ids), torch.zeros_like(token_ids))
        attention_mask = torch.cat([attention_mask.long(), masks], dim=1)
        actions = torch.cat([actions, token_ids], dim=1)
        state = torch.cat([actions, token_ids], dim=1)

        # check if all sequences have generated eos
        active_seqs = active_seqs * (token_ids != eos_token_id).squeeze(-1)
        if torch.all(~active_seqs):
            break
    # add EOS token to penalize incomplete sentences.
    eos_tokens = torch.ones(actions.size(
        0), dtype=torch.long, device=actions.device) * eos_token_id
    actions = torch.cat([actions, eos_tokens.unsqueeze(1)], dim=1)

    log_fs = torch.stack(log_fs, dim=1)
    log_pfs = torch.stack(log_pfs, dim=1)
    results = {"actions": actions, "log_fs": log_fs, "log_pfs": log_pfs, "attn_mask": attention_mask[:, prompt_ids.size(1):]}
    return results

def get_logpf_and_logz(prompt_batch, response_batch, model, eos_token_id, max_len=30, vargrad=False, sparsemax=False, temp=1.0):
    prompt_ids = prompt_batch["input_ids"]
    prompt_attention_mask = prompt_batch["attention_mask"]
    
    active_seqs = torch.ones(prompt_ids.size(0)).bool().to(prompt_ids.device)
    actions = prompt_ids.clone()
    state = prompt_ids.clone()
    sum_logpf = torch.zeros(prompt_ids.size(0)).float().to(prompt_ids.device)
    attention_mask = prompt_attention_mask.clone()
    outputs = model(
        state[:, :-1], attention_mask=attention_mask[:, :-1], output_hidden_states=True)
    hidden_states = outputs["hidden_states"][-1]
    past_key_values = outputs["past_key_values"]
    for i in range(max_len):
        if i == 0:
            # If past_key_values is used, attention_mask needs to contain the masking strategy that was used for past_key_values.
            # In other words, the attention_mask always has to have the length: len(past_key_values) + len(input_ids)
            # https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Model.forward.attention_mask
            output = model.module(
                state[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True
            )
            last_hidden = output.hidden_states[-1]
            last_hidden = torch.cat([hidden_states, last_hidden], dim=1)
            avg_pool = avg_pooling(last_hidden, prompt_attention_mask)
            
            if not vargrad:
                log_z = model.proj_z(avg_pool).squeeze(-1)
        else:
            output = model.module(
                state[:, -1:], attention_mask=attention_mask, past_key_values=past_key_values)
        past_key_values = output["past_key_values"]
        
        logits = output["logits"][:, -1, :]
        if i == 0:
            logits[..., eos_token_id] = -100.0
        with torch.no_grad():
            token_ids = response_batch["input_ids"][:, i].unsqueeze(1)
            
        logprob = F.log_softmax(logits, dim=-1)
        logprob = torch.gather(logprob, -1, token_ids).squeeze(-1)
        logprob = torch.where(active_seqs, logprob, torch.zeros_like(logprob))
        sum_logpf = sum_logpf + logprob
        
        token_ids = torch.where(
            active_seqs.unsqueeze(-1), token_ids, torch.ones_like(token_ids) * eos_token_id)
        
        # update action, state, mask
        masks = torch.where(active_seqs.unsqueeze(-1),
                            torch.ones_like(token_ids), torch.zeros_like(token_ids))
        attention_mask = torch.cat([attention_mask.long(), masks], dim=1)
        actions = torch.cat([actions, token_ids], dim=1)
        state = torch.cat([actions, token_ids], dim=1)
        
        # check if all sequences have generated eos
        active_seqs = active_seqs * (token_ids != eos_token_id).squeeze(-1)
        if torch.all(~active_seqs):
            break
    # add EOS token to penalize incomplete sentences.
    eos_tokens = torch.ones(actions.size(
        0), dtype=torch.long, device=actions.device) * eos_token_id
    actions = torch.cat([actions, eos_tokens.unsqueeze(1)], dim=1)
    
    if vargrad:
        return sum_logpf, -sum_logpf
    else:
        return sum_logpf, log_z

def get_logpf_and_logf(prompt_batch, response_batch, model, eos_token_id, max_len=30, vargrad=False, sparsemax=False, temp=1.0, proj_z=None):
    prompt_ids = prompt_batch["input_ids"]
    prompt_attention_mask = prompt_batch["attention_mask"]
    
    active_seqs = torch.ones(prompt_ids.size(0)).bool().to(prompt_ids.device)
    actions = prompt_ids.clone()
    state = prompt_ids.clone()
    
    log_fs = []
    log_pfs = []
    attention_mask = prompt_attention_mask.clone()
    outputs = model(
        state[:, :-1], attention_mask=attention_mask[:, :-1], output_hidden_states=True)
    hidden_states = outputs["hidden_states"][-1]
    past_key_values = outputs["past_key_values"]
    
    hidden_states_orig = torch.sum(hidden_states, dim=1).detach()
    max_len = response_batch["input_ids"].size(1) - 1
    for i in range(max_len):
        # If past_key_values is used, attention_mask needs to contain the masking strategy that was used for past_key_values.
        # In other words, the attention_mask always has to have the length: len(past_key_values) + len(input_ids)
        # https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Model.forward.attention_mask
        output = model.module(
            state[:, -1:],
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True
        )
        last_hidden = torch.sum(output.hidden_states[-1], dim=1).detach()
        log_f = proj_z(last_hidden, hidden_states_orig).squeeze(-1)
        log_f = torch.where(active_seqs, log_f, torch.zeros_like(log_f))
        log_fs.append(log_f)
        # else:
        #     output = model(
        #         state[:, -1:], attention_mask=attention_mask, past_key_values=past_key_values)
        past_key_values = output["past_key_values"]
        
        logits = output["logits"][:, -1, :]
        if i == 0:
            logits[..., eos_token_id] = -100.0
        with torch.no_grad():
            token_ids = response_batch["input_ids"][:, i].unsqueeze(1)
            
        logprob = F.log_softmax(logits, dim=-1)
        logprob = torch.gather(logprob, -1, token_ids).squeeze(-1)
        logprob = torch.where(active_seqs, logprob, torch.zeros_like(logprob))
        log_pfs.append(logprob)

        token_ids = torch.where(
            active_seqs.unsqueeze(-1), token_ids, torch.ones_like(token_ids) * eos_token_id)

        # update action, state, mask
        masks = torch.where(active_seqs.unsqueeze(-1),
                            torch.ones_like(token_ids), torch.zeros_like(token_ids))
        attention_mask = torch.cat([attention_mask.long(), masks], dim=1)
        actions = torch.cat([actions, token_ids], dim=1)
        state = torch.cat([actions, token_ids], dim=1)

        # check if all sequences have generated eos
        active_seqs = active_seqs * (token_ids != eos_token_id).squeeze(-1)
        if torch.all(~active_seqs):
            log_fs = log_fs[:-1]
            log_pfs = log_pfs[:-1]
            attention_mask = attention_mask[:, :-1]
            break
    # add EOS token to penalize incomplete sentences.
    eos_tokens = torch.ones(actions.size(
        0), dtype=torch.long, device=actions.device) * eos_token_id
    actions = torch.cat([actions, eos_tokens.unsqueeze(1)], dim=1)
    
    log_fs = torch.stack(log_fs, dim=1)
    log_pfs = torch.stack(log_pfs, dim=1)
    return log_pfs, log_fs, attention_mask[:, prompt_ids.size(1):]


class GFNTrainer(object):
    def __init__(self, args) -> None:
        self.args = args

        self.accelerator = Accelerator(
            split_batches=True,
        )

        if self.accelerator.is_main_process:
            wandb.init(reinit=True,
                    config=args.as_dict(),
                    project=args.wandb_project,
                    name=args.exp_name)
        
        self.device = self.accelerator.device
        config = AutoConfig.from_pretrained(args.model_name)
        config.use_cache = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            args.sft_ckpt,
            config=config,
            device_map=self.device
        )
        
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            args.sft_ckpt,
            config=config,
            device_map=self.device
        )
        
        if self.args.is_lora:
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["c_attn"],
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
        model_config = self.model.config

        if self.args.loss == "db" or self.args.loss == "fl-db":
        # self.model.proj_z = nn.Linear(model_config.n_embd, 1).to(self.device)
            # self.proj_z = nn.Sequential(
            #     nn.Linear(model_config.n_embd, 256),
            #     nn.ReLU(),
            #     nn.Linear(256, 256),
            #     nn.ReLU(),
            #     nn.Linear(256, 1)
            # ).to(self.device)
            self.proj_z = FlowFunction(model_config.n_embd).to(self.device)
            self.optimizer_flow = torch.optim.AdamW(self.proj_z.parameters(), lr=args.lr_flow)

        output_dir = os.path.join(self.args.save_dir, self.args.exp_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.sft_ckpt, 
            padding_side="left"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            args.target_model,
            subfolder="scheduler",
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            args.target_model,
            variant="fp16",
            scheduler=scheduler,
        )
        pipe.safety_checker = None
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self.diffusion_pipe = pipe
            
        self.clip_model, self.clip_preprocess = clip.load(
            args.clip_model,
            device=self.device,
        )
        
        self.aes_model = AestheticMlp(768)
        state_dict = torch.load(args.aes_model)
        self.aes_model.load_state_dict(state_dict)
        self.aes_model.to(device=self.device)
        self.aes_model.eval()
        
        self.sentence_encoder = SentenceTransformer(
            args.sentence_encoder,
            device=self.device
        )
        
        self.num_samples = self.args.batch_size
        dataloader = get_dataloader(
            "prompt-opt", self.tokenizer, prompt_file=args.prompt_file,
            batch_size=torch.cuda.device_count(), shuffle=True, prefix=args.prefix)
        
        self.eval_dataloader = get_dataloader(
            "prompt-opt", self.tokenizer, prompt_file=args.eval_prompt_file,
            batch_size=args.batch_size, shuffle=False, prefix=args.prefix)

        self.dataloader = self.accelerator.prepare(dataloader)
        self.train_iter = InfIterator(self.dataloader)
        self.eval_iter = InfIterator(self.eval_dataloader)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        t_total = args.train_steps * args.grad_acc_steps
        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, args.num_warmup_steps, t_total)
        
        if self.args.loss == "db" or self.args.loss == "fl-db":
            self.model, self.proj_z, self.ref_model, self.optimizer, self.optimizer_flow = self.accelerator.prepare(
                self.model, self.proj_z, self.ref_model, self.optimizer, self.optimizer_flow
            )
        else:
            self.model, self.ref_model, self.optimizer = self.accelerator.prepare(
                self.model, self.ref_model, self.optimizer
            )
        
        # initialize buffer
        if args.metric == "edit":
            print("edit distance for buffer")
            self.rbuffer = ReplayBuffer(
                self.tokenizer.eos_token_id,
                self.args.buffer_size,
                prioritization=self.args.prioritization,
                compare=self.args.compare,
                train_batch_size=self.args.batch_size,
                sim_tolerance=self.args.sim_tolerance)
        elif args.metric == "cosine":
            print("cosine similarity for buffer")
            self.rbuffer = CosineRelayBuffer(
                self.tokenizer.eos_token_id,
                self.args.buffer_size,
                prioritization=self.args.prioritization,
                compare=self.args.compare,
                train_batch_size=self.args.batch_size,
                sim_tolerance=self.args.sim_tolerance)
        
        if self.args.loss == "db" or self.args.loss == "fl-db":
            self.start = self.load(output_dir, self.model,
                                self.optimizer, self.rbuffer, self.proj_z, self.optimizer_flow)
        else:
            self.start = self.load(output_dir, self.model,
                                self.optimizer, self.rbuffer)

        delimiter = ","
        self.csvlogger = CsvLogger(filename=f"logs/{args.exp_name}.csv",
                                   delimiter=delimiter,
                                   level=logging.INFO,
                                   add_level_nums=None,
                                   fmt=f'%(asctime)s{delimiter}%(message)s',
                                   datefmt='%Y/%m/%d %H:%M:%S',
                                   header=["date", "output", "c_log_reward", "lm_log_reward"])

        self.prompt_fn = self.make_prompt
        self.num_images_per_prompt = args.num_images_per_prompt
        
        # loss type
        self.loss = args.loss

    @staticmethod
    def make_prompt(instruction):
        prompt_template = "{instruction} Rephrase:\n"
        return prompt_template.format(instruction=instruction.rstrip())
    
    def save(self, output_dir, rbuffer, step):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.module.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        if self.args.loss == "db" or self.args.loss == "fl-db":
            ckpt = {"global_step": step,
                    "optimizer": self.optimizer.state_dict(),
                    "proj_z": self.proj_z.module.state_dict(),
                    "optimizer_flow": self.optimizer_flow.state_dict()}
        else:
            ckpt = {"global_step": step,
                    "optimizer": self.optimizer.state_dict()}
        ckpt_file = os.path.join(output_dir, "ckpt.pt")
        torch.save(ckpt, ckpt_file)

        rbuffer.save(os.path.join(output_dir, "buffer.pkl"))
    
    def load(self, output_dir, model, optimizer, rbuffer, proj_z=None, optimizer_flow=None):
        # load checkpoint and return starting step
        if not os.path.exists(output_dir):
            return 1
        dirs = sorted(os.listdir(output_dir))
        if len(dirs) == 0:
            return 1
        else:
            dirs = [int(x) for x in dirs if x.isdigit()]
            dirs = sorted(dirs, reverse=True)
            ckpt_dir = os.path.join(output_dir, str(dirs[0]))
            _model = AutoModelForCausalLM.from_pretrained(ckpt_dir)
            # we do not load proj_z here
            model = self.accelerator.unwrap_model(model)
            msg = model.load_state_dict(_model.state_dict(), strict=False)
            print(msg)

            # load optimizer, scheduler, and proj_z
            ckpt = torch.load(os.path.join(ckpt_dir, "ckpt.pt"))
            if self.args.loss == "db" or self.args.loss == "fl-db":
                proj_z = self.accelerator.unwrap_model(proj_z)
                proj_z.load_state_dict(ckpt["proj_z"])
                optimizer_flow.load_state_dict(ckpt["optimizer_flow"])
            optimizer.load_state_dict(ckpt["optimizer"])
            # scheduler.load_state_dict(ckpt["scheduler"])

            # load buffer
            buffer_ckpt = os.path.join(ckpt_dir, "buffer.pkl")
            rbuffer.load(buffer_ckpt)
            return ckpt["global_step"] + 1
            
    def gen_image_batched(self, prompts, bsz=32):
        prompts = self.tokenizer.batch_decode(prompts, skip_special_tokens=True)
        if prompts[0] == prompts[1]:
            num_repeat = len(prompts)
            prompts = prompts[:1]
        else:
            num_repeat = 1
        images = []
        for i in range(0, len(prompts), bsz):
            pmpts = prompts[i: i + bsz]
            g = torch.Generator(device="cuda").manual_seed(42)
            with autocast("cuda"):
                sub_images = self.diffusion_pipe(pmpts, 
                                                    num_images_per_prompt=self.num_images_per_prompt, 
                                                    num_inference_steps=20,
                                                    generator=g).images
                images.extend(sub_images)
        
        if num_repeat > 1:
            images = images * num_repeat
        return images
    
    def get_clip_features(self, pil_image, is_batched=False):
        if not is_batched:
            image = self.clip_preprocess(pil_image).unsqueeze(0)
        else:
            images = [self.clip_preprocess(i) for i in pil_image]
        image = torch.stack(images)
        image = image.to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
        return image_features
    
    def get_clip_score_batched(self, image_features, prompts):
        prompts = self.tokenizer.batch_decode(prompts, skip_special_tokens=True)
        tokens = clip.tokenize(prompts, truncate=True).to(self.device)

        with torch.no_grad():
            if len(image_features) != len(prompts):
                assert len(image_features) % len(prompts) == 0
                tokens = tokens.unsqueeze(1).expand(-1, self.num_images_per_prompt, -1).reshape(-1, tokens.shape[-1])
            
            text_features = self.clip_model.encode_text(tokens)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            # logit_scale = self.clip_model.logit_scale.exp()
            logit = image_features @ text_features.t()
        scores = logit.diag().tolist()
        return scores
    
    def get_aesthetic_score(self, image_features, is_batched=False):
        features = image_features.cpu().detach().numpy()
        order = 2
        axis = -1
        l2 = np.atleast_1d(np.linalg.norm(features, order, axis))
        l2[l2 == 0] = 1
        im_emb_arr = features / np.expand_dims(l2, axis)
        prediction = self.aes_model(torch.from_numpy(im_emb_arr).to(self.device).float())
        if is_batched:
            return prediction[:, 0].tolist()
        else:
            return prediction.item()

    def get_imagereward_score(self, images, prompts):
        prompts = self.tokenizer.batch_decode(prompts, skip_special_tokens=True)
        images = [self.clip_preprocess.module(i) for i in images]
        images = torch.stack(images).to(self.device)
        return self.image_reward(images, prompts)
    
    def get_hpscore(self, images, prompts):
        prompts = self.tokenizer.batch_decode(prompts, skip_special_tokens=True)
        images = [self.clip_preprocess.module(i) for i in images]
        images = torch.stack(images).to(self.device)
        return self.hpscore(images, prompts)

    def get_logreward(self,
                      prompt_inputs: Dict[str, Union[List, torch.LongTensor]],
                      prompts_responses: torch.LongTensor,
                      return_image: bool = False):
        # prompt_inputs: input_ids, attention_mask of prompt
        # prompts_responses: concatenation of prompt and response
        prompt_len = prompt_inputs["input_ids"].size(1)
        only_prompts = prompts_responses[:, :prompt_len]
        only_responses = prompts_responses[:, prompt_len:]
        # the first pad token is EOS
        pad_mask = (only_responses ==
                    self.tokenizer.pad_token_id).cumsum(1) > 1
        attention_mask = torch.cat(
            [prompt_inputs["attention_mask"], (~pad_mask).long()], 1)
        
        # llh from reference model
        with torch.no_grad():
            # lora_to_base(self.model)
            outputs = self.ref_model.module(input_ids=prompts_responses,
                                 attention_mask=attention_mask)
            logits = outputs.logits[:, prompt_len-1:-1]
            log_prob = F.log_softmax(logits, dim=-1)
            labels = prompts_responses[:, prompt_len:]

            lm_logreward = torch.gather(
                log_prob, -1, labels.unsqueeze(2)).squeeze(2)
            lm_logreward = torch.where(pad_mask, 0.0, lm_logreward)[:, :-1]
            lm_logreward_decom = lm_logreward.clone()
            lm_logreward = torch.sum(lm_logreward, 1)
            # base_to_lora(self.model)
            
        # length penalty
        response_lengths = torch.sum((~pad_mask).long(), 1)
        lm_logreward = torch.where(
            response_lengths < self.args.min_len, -500, lm_logreward)
        
        # aesthetic reward
        images = self.gen_image_batched(only_responses, bsz=self.args.bsz)
        image_features = self.get_clip_features(images, is_batched=True)
        if self.args.reward_metric == "aes":
            aes_scores = self.get_aesthetic_score(image_features, is_batched=True)
        elif self.args.reward_metric == "imagereward":
            aes_scores = self.get_imagereward_score(images, only_responses)
        elif self.args.reward_metric == "hpscore":
            aes_scores = self.get_hpscore(images, only_responses)

        images_plain = self.gen_image_batched(only_prompts, bsz=self.args.bsz)
        images_plain_features = self.get_clip_features(images_plain, is_batched=True)
        if self.args.reward_metric == "aes":
            aes_scores_plain = self.get_aesthetic_score(images_plain_features, is_batched=True)
        elif self.args.reward_metric == "imagereward":
            aes_scores_plain = self.get_imagereward_score(images_plain, only_prompts)
        elif self.args.reward_metric == "hpscore":
            aes_scores_plain = self.get_hpscore(images_plain, only_prompts)
        
        aes_scores = torch.FloatTensor(aes_scores).reshape(-1, self.num_images_per_prompt).mean(dim=-1).to(self.device)
        aes_scores_plain = torch.FloatTensor(aes_scores_plain).reshape(-1, self.num_images_per_prompt).mean(dim=-1).to(self.device)
        aes_reward = aes_scores - aes_scores_plain
        
        # clip scores reward
        clip_scores = self.get_clip_score_batched(image_features, only_prompts)
        clip_scores = torch.FloatTensor(clip_scores).to(self.device)
        clip_scores = torch.maximum(clip_scores, torch.zeros_like(clip_scores))
        clip_reward = torch.where(clip_scores>self.args.threshold, 0, 20*clip_scores-20*self.args.threshold).reshape(-1, self.num_images_per_prompt).mean(-1)
        
        decoded_responses = self.tokenizer.batch_decode(only_responses, skip_special_tokens=True)
        if return_image:
            return lm_logreward, lm_logreward_decom, aes_reward, clip_reward, decoded_responses, images, images_plain
        else:
            return lm_logreward, lm_logreward_decom, aes_reward, clip_reward, decoded_responses
    
    @torch.no_grad()
    def get_avg_pairwise_cossim(self, sentences):
        embeddings = self.sentence_encoder.encode(
            sentences, convert_to_tensor=True)
        cos_sim = F.cosine_similarity(embeddings.unsqueeze(
            0), embeddings.unsqueeze(1), -1).cpu()
        off_diag = cos_sim.masked_select(
            ~torch.eye(cos_sim.size(0), dtype=bool)).view(-1)
        avg_sim = torch.mean(off_diag).item()

        return avg_sim
    
    def get_sampling_temp(self, step):
        args = self.args
        diff = args.temp_high - args.temp_low
        if args.temp_scheduler == "linear":
            temp = args.temp_high - diff * \
                min(1, step / args.temp_sched_horizon)
        elif args.temp_scheduler == "loglinear":
            temp = args.temp_high - diff * \
                min(1, math.log(step + 1) / math.log(args.temp_sched_horizon + 1))
        elif args.temp_scheduler == "random":
            temp = random.uniform(args.temp_low, args.temp_high)
        else:
            temp = args.temp_low
        return temp
    
    def get_total_reward_temp(self, step):
        args = self.args
        diff = args.reward_sched_end - args.reward_sched_start
        temp = args.reward_sched_start + diff * \
            min(1, step / args.reward_sched_horizon)
        return temp

    def get_lm_reward_temp(self, step):
        diff = self.args.lm_sched_end - self.args.lm_sched_start
        temp = self.args.lm_sched_start + diff * \
            min(1, step / self.args.lm_sched_horizon)
        return temp
    
    def get_online_samples(self, batch, min_len, max_len, temp=1.0):
        # input_ids is left-side padded
        if self.loss == "db" or self.loss == "fl-db":
            outputs = generate_and_return_flow_logprob(
                model=self.model,
                prompt_ids=batch["input_ids"],
                prompt_attention_mask=batch["attention_mask"],
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=temp,
                min_len=min_len,
                max_len=max_len,
                num_samples=self.num_samples,
                vargrad=self.args.vargrad,
                exploration=self.args.exploration,
                sparsemax=self.args.sparsemax,
                proj_z=self.proj_z,
            )
            prompts_responses = outputs["actions"]
            log_fs = outputs["log_fs"]
            log_pfs = outputs["log_pfs"]
            attn_mask = outputs["attn_mask"]

            batch = {k: v.repeat(self.num_samples, 1) for k, v in batch.items()}
            lm_logreward, lm_logreward_decom, aes_log_reward, rel_log_reward, decoded_responses = self.get_logreward(
                batch, prompts_responses)
            c_log_reward = aes_log_reward * self.args.aes_weight + rel_log_reward * self.args.rel_weight
            
            results = {"lm_log_reward": lm_logreward,
                       "lm_log_reward_decom": lm_logreward_decom,
                       "c_log_reward": c_log_reward,
                       "aes_log_reward": aes_log_reward,
                       "rel_log_reward": rel_log_reward,
                       "log_fs": log_fs,
                       "log_pfs": log_pfs,
                       "attn_mask": attn_mask,
                       "prompts_responses": prompts_responses,
                       "decoded_responses": decoded_responses,
            }
        else:
            outputs = generate_and_return_z_logprob(
                model=self.model,
                prompt_ids=batch["input_ids"],
                prompt_attention_mask=batch["attention_mask"],
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=temp,
                min_len=min_len,
                max_len=max_len,
                num_samples=self.num_samples,
                vargrad=self.args.vargrad,
                exploration=self.args.exploration,
                sparsemax=self.args.sparsemax,
            )
            prompts_responses = outputs["actions"]
            log_z = outputs["log_z"]
            sum_logpf = outputs["sum_logpf"]

            batch = {k: v.repeat(self.num_samples, 1) for k, v in batch.items()}
            lm_log_reward, lm_logreward_decom, aes_log_reward, rel_log_reward, decoded_responses = self.get_logreward(
                batch, prompts_responses)
            c_log_reward = aes_log_reward * self.args.aes_weight + rel_log_reward * self.args.rel_weight

            results = {"lm_log_reward": lm_log_reward,
                       "c_log_reward": c_log_reward,
                       "aes_log_reward": aes_log_reward,
                       "rel_log_reward": rel_log_reward,
                       "log_z": log_z,
                       "sum_logpf": sum_logpf,
                       "prompts_responses": prompts_responses,
                       "decoded_responses": decoded_responses,
            }
        return results
        
    def get_offline_samples(self, prompt_batch, response_batch, reward_batch, temp=1.0):
        prompt_batch = {k: v.to(self.device) for k, v in prompt_batch.items()}
        response_batch = {k: v.to(self.device) for k, v in response_batch.items()}
        reward_batch = {k: v.to(self.device) for k, v in reward_batch.items()}
        
        if self.loss == "tb":
            sum_logpf, log_z = get_logpf_and_logz(
                prompt_batch, response_batch, self.model, self.tokenizer.eos_token_id, self.args.max_len, self.args.vargrad, self.args.sparsemax, temp)
            if self.args.vargrad:
                log_z += reward_batch["log_reward"]

            decoded_responses = self.tokenizer.batch_decode(
                response_batch["input_ids"], skip_special_tokens=True)

            results = {
                "log_z": log_z,
                "sum_logpf": sum_logpf,
                "decoded_responses": decoded_responses
            }
        elif self.args.loss == "db" or self.args.loss == "fl-db":
            log_pfs, log_fs, attn_mask = get_logpf_and_logf(
                prompt_batch, response_batch, self.model, self.tokenizer.eos_token_id, self.args.max_len, self.args.vargrad, self.args.sparsemax, temp, proj_z=self.proj_z)
            lm_logreward, lm_logreward_decom, aes_log_reward, rel_log_reward, decoded_responses = self.get_logreward(
                prompt_batch, torch.cat([prompt_batch["input_ids"], response_batch["input_ids"]], dim=1))
            
            decoded_responses = self.tokenizer.batch_decode(
                response_batch["input_ids"], skip_special_tokens=True)

            results = {
                "log_fs": log_fs,
                "log_pfs": log_pfs,
                "lm_log_reward_decom": lm_logreward_decom,
                "attn_mask": attn_mask,
                "decoded_responses": decoded_responses
            }
        results.update(reward_batch)
        return results
    
    def simulate_experience(self, batch, rbuffer, beta, min_len, max_len, step):
        policy = random.randint(0, 1)  # integer from [0,1]
        temp = self.get_sampling_temp(step)
        if policy == 0:
            results = self.get_online_samples(
                batch, min_len=min_len, max_len=max_len, temp=temp)

            aes_log_reward = results["aes_log_reward"]
            rel_log_reward = results["rel_log_reward"]
            c_log_reward = results["c_log_reward"]
            lm_log_reward = results["lm_log_reward"]
            log_reward = lm_log_reward + (c_log_reward / beta)
            
            if self.args.vargrad:
                results["log_z"] += log_reward
                prompt_ids = batch["input_ids"].repeat(self.num_samples, 1)
            else:
                prompt_ids = batch["input_ids"].repeat(self.num_samples, 1)

            prompts_responses = results["prompts_responses"]
            responses = prompts_responses[:, prompt_ids.size(1):]

            decoded_responses = results["decoded_responses"]

            response_embs = self.sentence_encoder.encode(
                decoded_responses, convert_to_tensor=True)

            rbuffer.add_batch(prompt_ids, responses, decoded_responses, response_embs,
                              aes_log_reward, rel_log_reward,
                              c_log_reward, lm_log_reward, log_reward)
        else:
            bs = batch["input_ids"].size(0)
            # sample from buffer
            prompt_batch, response_batch, reward_batch = rbuffer.sample(bs)
            results = self.get_offline_samples(
                prompt_batch, response_batch, reward_batch, temp=temp)

        return results
    
    def compute_tb_loss(self, log_z, sum_logpf, log_reward):
        num_samples = log_z.size(0)
        if self.args.vargrad:
            if self.args.z_reset and random.random() < 0.05:
                log_z = torch.zeros_like(sum_logpf).to(self.device)
            else:
                log_z = log_z.mean(dim=0, keepdim=True).repeat(num_samples).detach()
        delta = log_z + sum_logpf - log_reward
        losses = delta**2
        return losses
    
    def compute_db_loss(self, log_fs, log_pfs, attn_mask, log_reward, lm_log_reward_decom):
        if log_fs.shape[1] != lm_log_reward_decom.shape[1]:
            # TODO: Should be fixed in the future
            lm_log_reward_decom = lm_log_reward_decom[:, :log_fs.shape[1]]
        
        last_indices = (attn_mask == 1).sum(dim=1) - 1
        delta = torch.zeros_like(log_fs).to(self.device)
        delta += log_fs + log_pfs
        delta -= torch.cat([log_fs[:, 1:], torch.zeros_like(log_reward.unsqueeze(-1)).to(self.device)], dim=1)
        
        #### Add this part make FL-DB #####
        if self.loss == "fl-db":
            log_reward -= lm_log_reward_decom.sum(dim=1)
            delta -= lm_log_reward_decom
        ###################################

        for i, idx in enumerate(last_indices):
            delta[i, idx] -= log_reward[i]
        
        losses = (delta**2).sum(dim=1)
        return losses
    
    def get_batch_metrics(self, batch, step, rbuffer, min_len, max_len, beta, train=True):
        metrics = {}
        train_test = 'train' if train else 'eval'
        
        results = self.simulate_experience(
            batch, rbuffer, beta=beta, min_len=min_len, max_len=max_len, step=step)
        if step % 100 == 0:
            cos_sim = self.get_avg_pairwise_cossim(
                results["decoded_responses"])
            metrics["cos-sim"] = [cos_sim]
            
        c_log_reward = results["c_log_reward"]
        lm_log_reward = results["lm_log_reward"] 
        
        gamma = self.get_lm_reward_temp(step)
        log_reward = (lm_log_reward / gamma) + (c_log_reward / beta)
        
        rew_temp = self.get_total_reward_temp(step)
        tempered_log_reward = log_reward / rew_temp
        
        if self.loss == "tb":
            losses = self.compute_tb_loss(
                results["log_z"], results["sum_logpf"], tempered_log_reward)
            metrics[f"log_z"] = results["log_z"].detach().tolist()
        elif self.loss == "db" or self.loss == "fl-db":
            lm_log_reward_decom = results["lm_log_reward_decom"]
            lm_log_reward_decom /= gamma
            tempered_lm_log_reward_decom = lm_log_reward_decom / rew_temp
            
            losses = self.compute_db_loss(
                results["log_fs"], results["log_pfs"], results["attn_mask"], tempered_log_reward, tempered_lm_log_reward_decom)
        
        metrics[f'aes_log_reward/{train_test}'] = results["aes_log_reward"].tolist()
        metrics[f'rel_log_reward/{train_test}'] = results["rel_log_reward"].tolist()
        metrics[f"c_log_reward/{train_test}"] = c_log_reward.tolist()
        metrics[f"lm_log_reward/{train_test}"] = lm_log_reward.tolist()
        metrics[f"log_reward/{train_test}"] = log_reward.tolist()
        metrics[f"loss/{train_test}"] = losses.detach().tolist()

        return losses.mean(), metrics
    
    def train(self):
        t = tqdm(range(self.start, self.args.train_steps+1),
            desc="training", dynamic_ncols=True, disable = not self.accelerator.is_main_process)
        for global_step in t:
            batch_metrics = defaultdict(list) 
           
            self.model.train()
            for _ in range(self.args.grad_acc_steps):
                batch = next(self.train_iter)
                batch = batch.to(self.device)
                with self.accelerator.autocast():
                    loss, metrics = self.get_batch_metrics(
                        batch, global_step, self.rbuffer,
                        self.args.min_len, self.args.max_len, self.args.beta)

                    for k, v in metrics.items():
                        batch_metrics[k].extend(v)
                    loss = loss / self.args.grad_acc_steps

                self.accelerator.backward(loss)
            self.accelerator.wait_for_everyone()
            
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.args.loss == "db" or self.args.loss == "fl-db":
                self.optimizer_flow.step()
                self.optimizer_flow.zero_grad()

            # logging
            batch_metrics = {k: sum(v) / float(len(v))
                             for k, v in batch_metrics.items()}

            t.set_description(
                f"Step {global_step}: {formatted_dict(batch_metrics)}")

            if self.accelerator.is_main_process and global_step % self.args.eval_period == 0:
                output_dir = os.path.join(
                    self.args.save_dir, f"{self.args.exp_name}/{global_step}")
                self.save(output_dir, self.rbuffer, global_step)
                
                
            if self.accelerator.is_main_process and self.loss in ["db", "fl-db"]:
                if self.args.flow_reset and global_step % self.args.flow_reset_period == 0:
                    self.accelerator.print("Flow Reset ....")
                    for name, param in self.proj_z.module.named_parameters():
                        # Reset last hidden layer
                        if "proj_z2" in name:
                            param.data = torch.randn_like(param.data)
                    self.accelerator.print("Completed")
            
            if self.accelerator.is_main_process:
                wandb.log(batch_metrics, step=global_step)    
            
            self.accelerator.wait_for_everyone()
                
        if self.accelerator.is_main_process:
            output_dir = os.path.join(
                self.args.save_dir, self.args.exp_name, "latest")
            self.save(output_dir, self.rbuffer, global_step)
        
        if self.accelerator.is_main_process:
            wandb.finish()