import gzip
import heapq
import json
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Dict, List

import editdistance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, pipeline
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
    
class AestheticMlp(nn.Module):

  def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
    super().__init__()
    self.input_size = input_size
    self.xcol = xcol
    self.ycol = ycol
    self.layers = nn.Sequential(
        nn.Linear(self.input_size, 1024),
        #nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 128),
        #nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        #nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 16),
        #nn.ReLU(),
        nn.Linear(16, 1)
    )

  def forward(self, x): 
        return self.layers(x)


def batch_cosine_similarity_kernel(embeddings, batch_size=16):
    num_samples = embeddings.size(0)
    avg_sim = 0.0

    for i in tqdm(range(0, num_samples, batch_size)):
        batch_end = min(i + batch_size, num_samples)
        batch = embeddings[i:batch_end, :]
        with torch.no_grad():
            cos_sim_batch = F.linear(F.normalize(
                batch), F.normalize(embeddings))
        avg_sim += cos_sim_batch.sum().item()

    # Adjust for duplicate pairs and remove diagonal components
    diag = 0.0
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch = embeddings[i:batch_end, :]
        diag += F.cosine_similarity(batch, batch, dim=-1).sum().item()
    avg_sim -= diag

    # Compute average similarity
    avg_sim /= (num_samples * (num_samples - 1))

    return avg_sim


def seed(seed=42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_decay_parameter_names(model) -> List[str]:
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [
        name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.2f}" if type(v) == float else v) for k, v in d.items()}


class InfIterator(object):
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.iterable)
            return next(self.iterator)

    def __len__(self):
        return len(self.iterator)


@dataclass(order=True)
class TrajectoryWithReward:
    prompt_ids: list = field(compare=False)
    response_ids: list = field(compare=False)
    aes_log_reward: float = field(compare=False)
    rel_log_reward: float = field(compare=False)
    c_log_reward: float = field(compare=False)
    lm_log_reward: float = field(compare=False)
    log_reward: float = field(compare=False)  # sorting based on this
    decoded_response: str = field(compare=False)
    emb: torch.tensor = field(compare=False)
    ref_reward: float = field(compare=True, init=False)

    def __post_init__(self):
        self.ref_reward = self.log_reward.mean().item()


@dataclass(order=True)
class TrajectoryWithCReward:
    prompt_ids: list = field(compare=False)
    response_ids: list = field(compare=False)
    aes_log_reward: float = field(compare=False)
    rel_log_reward: float = field(compare=False)
    c_log_reward: float = field(compare=True)  # sorting based on this
    lm_log_reward: float = field(compare=False)
    log_reward: float = field(compare=False)
    decoded_response: str = field(compare=False)
    emb: torch.tensor = field(compare=False)
    ref_reward: float = field(compare=False, init=False)

    def __post_init__(self):
        self.ref_reward = self.c_log_reward


class ReplayBuffer(object):
    def __init__(self,  eos_token_id, max_size=1000, sim_tolerance=0.25, prioritization="c_reward", compare="reward", train_batch_size=16):
        self.eos_token_id = eos_token_id
        self.max_size = max_size
        self.sim_tolerance = sim_tolerance
        self.buffer = []
        self.response_pool = set()
        self.prioritization = prioritization
        self.compare = compare
        self.train_batch_size = train_batch_size

        if compare == "c_reward":
            print("comparison with c_reward")
            self.Trajectory = TrajectoryWithCReward
        else:
            print("comparison with total reward")
            self.Trajectory = TrajectoryWithReward

    def size(self):
        return len(self.buffer)

    def add(self, item):
        # check whether the item has been already added before.
        if item.decoded_response in self.response_pool:
            return
        tokens = [x for x in item.response_ids.tolist() if x !=
                  self.eos_token_id]
        # find examples that are similar to the item and replace it with new one if new one has higher reward
        for buffer_item in self.buffer:
            existing_tokens = [
                x for x in buffer_item.response_ids.tolist() if x != self.eos_token_id]
            print(editdistance.eval(tokens, existing_tokens))
            if editdistance.eval(tokens, existing_tokens) < (len(tokens) + len(existing_tokens)) * self.sim_tolerance:
                if buffer_item.ref_reward >= item.ref_reward:
                    return
                else:
                    # remove the old item
                    self.response_pool.discard(buffer_item.decoded_response)
                    self.buffer.remove(buffer_item)
                    heapq.heapify(self.buffer)

                    # add new item
                    self.response_pool.add(item.decoded_response)
                    heapq.heappush(self.buffer, item)

                    if len(self.buffer) != len(self.response_pool):
                        self.response_pool = set(
                            [x.decoded_response for x in self.buffer])
                    return

        self.response_pool.add(item.decoded_response)

        if len(self.buffer) < self.max_size:
            heapq.heappush(self.buffer, item)
        else:
            popped = heapq.heappushpop(self.buffer, item)
            try:
                self.response_pool.remove(popped.decoded_response)
            except KeyError:
                self.response_pool = set(
                    [x.decoded_response for x in self.buffer])

    def add_batch(self, prompts, responses, decoded_responses, res_embs, 
                  aes_log_rewards, rel_log_rewards,
                  c_log_rewards, lm_log_rewards, log_rewards):
        # move tensors to cpu
        prompts = prompts.cpu()
        responses = responses.cpu()
        res_embs = res_embs.cpu()
        aes_log_rewards = aes_log_rewards.cpu()
        rel_log_rewards = rel_log_rewards.cpu()
        c_log_rewards = c_log_rewards.cpu()
        lm_log_rewards = lm_log_rewards.cpu()
        log_rewards = log_rewards.cpu()
        
        self.add(prompts, responses, aes_log_rewards, rel_log_rewards, c_log_rewards, lm_log_rewards, log_rewards, decoded_responses, res_embs) 
        
    def sample(self, num_samples):
        if self.prioritization == "reward":
            priorities = [item.log_reward.mean() for item in self.buffer]
            priorities = np.array(priorities)
            priorities = priorities - np.max(priorities)
            priorities = np.exp(priorities)
            prob = priorities / np.sum(priorities)

        elif self.prioritization == "c_reward":
            priorities = [item.c_log_reward.mean() for item in self.buffer]
            priorities = np.array(priorities)
            priorities = priorities - np.max(priorities)
            priorities = np.exp(priorities)
            prob = priorities / np.sum(priorities)

        else:
            # raise NotImplementedError
            prob = np.ones(len(self.buffer)) / len(self.buffer)
        idx = np.random.choice(
            len(self.buffer), num_samples, p=prob, replace=False)
        
        # right-side padding
        prompt_ids = [self.buffer[i].prompt_ids for i in idx]
        prompt_mask = [torch.ones_like(x) for x in prompt_ids]
        
        prompt_ids = torch.cat(prompt_ids, dim=0)
        prompt_mask = torch.cat(prompt_mask, dim=0)
                
        prompt_ids = pad_sequence(
            prompt_ids, batch_first=True, padding_value=self.eos_token_id)
        prompt_mask = pad_sequence(
            prompt_mask, batch_first=True, padding_value=0)
        
        prompt_batch = {"input_ids": prompt_ids,
                        "attention_mask": prompt_mask}

        # right-side padding
        response_ids = [self.buffer[i].response_ids for i in idx]
        response_mask = [torch.ones_like(x) for x in response_ids]
        
        response_ids = torch.cat(response_ids, dim=0)
        response_mask = torch.cat(response_mask, dim=0)

        response_batch = {"input_ids": response_ids,
                          "attention_mask": response_mask}

        aes_log_rewards = torch.cat([self.buffer[i].aes_log_reward for i in idx], dim=0)
        rel_log_rewards = torch.cat([self.buffer[i].rel_log_reward for i in idx], dim=0)
        c_log_rewards = torch.cat([self.buffer[i].c_log_reward for i in idx], dim=0)
        lm_log_rewards = torch.cat([self.buffer[i].lm_log_reward for i in idx], dim=0)
        log_rewards = torch.cat([self.buffer[i].log_reward for i in idx], dim=0)

        reward_batch = {"aes_log_reward": aes_log_rewards,
                        "rel_log_reward": rel_log_rewards,
                        "c_log_reward": c_log_rewards,
                        "lm_log_reward": lm_log_rewards,
                        "log_reward": log_rewards}

        return prompt_batch, response_batch, reward_batch

    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, path):
        with gzip.open(path, "rb") as f:
            self.buffer = pickle.load(f)
        heapq.heapify(self.buffer)


class CosineRelayBuffer(ReplayBuffer):
    def __init__(self, eos_token_id, max_size=1000, sim_tolerance=0.4, prioritization="c_reward", compare="reward", train_batch_size=16):
        super().__init__(eos_token_id, max_size, sim_tolerance, prioritization, compare)

    def add(self, prompts, responses, aes_log_rewards, rel_log_rewards, c_log_rewards, lm_log_rewards, log_rewards, decoded_responses, res_embs):
        # check whether the item has been already added before.
        # if item.decoded_response in self.response_pool:
        #     return
        save_indices = []
        num_samples = len(prompts)
        for i in range(num_samples):
            if i < (num_samples//2):
                save_indices.append(i)
            else:
                cos_sims = F.cosine_similarity(res_embs[i].unsqueeze(0), res_embs[save_indices], dim=1)
                cos_sim = torch.max(cos_sims).item()
                if cos_sim < self.sim_tolerance:
                    save_indices.append(i)

        save_indices = torch.tensor(save_indices)
        item = self.Trajectory(
            prompts[save_indices],
            responses[save_indices],
            aes_log_rewards[save_indices],
            rel_log_rewards[save_indices],
            c_log_rewards[save_indices],
            lm_log_rewards[save_indices],
            log_rewards[save_indices],
            [decoded_responses[i] for i in save_indices],
            res_embs[save_indices]
        )
        

        if len(self.buffer) < self.max_size:
            heapq.heappush(self.buffer, item)
        else:
            heapq.heappushpop(self.buffer, item)