import os
import random
import argparse
from typing import Dict, List, Union

import clip
import torch
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from dataset import get_dataloader
from pytorch_fid.inception import InceptionV3
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
from diffusers import DPMSolverMultistepScheduler, LCMScheduler
from diffusers import AutoPipelineForText2Image, AutoencoderTiny
from utils import AestheticMlp, batch_cosine_similarity_kernel, seed
# import huggingface_hub
# huggingface_hub.login()

# For ImageReward
import ImageReward as RM
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    
def imagereward(dtype=torch.float32, device="cuda"):
    # aesthetic = RM.load_score("Aesthetic", device=device)
    reward_model = RM.load("ImageReward-v1.0", device=device)

    rm_preprocess = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _fn(images, prompts):
        dic = reward_model.blip.tokenizer(prompts,
                padding='max_length', truncation=True,  return_tensors="pt",
                max_length=75) # max_length=512
        device = images.device
        input_ids, attention_mask = dic.input_ids.to(device), dic.attention_mask.to(device)
        reward = reward_model.score_gard(input_ids, attention_mask, rm_preprocess(images)) # differentiable
        return reward.reshape(images.shape[0]).cpu().detach().numpy() # bf16 -> f32

    return _fn

def gen_image_batched(pipe, device, tokenizer, prompts,
                      num_images_per_prompt=3,
                      num_inference_steps=4,
                      guidance_scale=0.0,
                      bsz=32,
                      seed=42):
    prompts = tokenizer.batch_decode(prompts, skip_special_tokens=True)
    if prompts[0] == prompts[1]:
        num_repeat = len(prompts)
        prompts = prompts[:1]
    else:
        num_repeat = 1
    images = []
    for i in range(0, len(prompts), bsz):
        g = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            sub_images = pipe(prompts[i:i+bsz], 
                              height=512,
                              width=512,
                              num_images_per_prompt=num_images_per_prompt, 
                              num_inference_steps=num_inference_steps, 
                              guidance_scale=guidance_scale,
                              generator=g).images
            images.extend(sub_images)
    if num_repeat > 1:
        images = images * num_repeat
    return images

def get_clip_features(clip_model, clip_preprocess, device, pil_image, is_batched=False):
    if not is_batched:
        image = clip_preprocess(pil_image).unsqueeze(0)
    else:
        images = [clip_preprocess(i) for i in pil_image]
    image = torch.stack(images)
    image = image.to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    return image_features

def get_clip_score_batched(clip_model, device, tokenizer, image_features, prompts,
                           num_images_per_prompt=3):
    prompts = tokenizer.batch_decode(prompts, skip_special_tokens=True)
    tokens = clip.tokenize(prompts, truncate=True).to(device)
    
    with torch.no_grad():
        if len(image_features) != len(prompts):
            assert len(image_features) % len(prompts) == 0
            tokens = tokens.unsqueeze(1).expand(-1, num_images_per_prompt, -1).reshape(-1, tokens.shape[-1])
        
        text_features = clip_model.encode_text(tokens)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # logit_scale = self.clip_model.logit_scale.exp()
        logit = image_features @ text_features.t()
    scores = logit.diag().tolist()
    return scores

def get_aesthetic_score(aes_model, device, image_features, is_batched=False):
    features = image_features.cpu().detach().numpy()
    order = 2
    axis = -1
    l2 = np.atleast_1d(np.linalg.norm(features, order, axis))
    l2[l2 == 0] = 1
    im_emb_arr = features / np.expand_dims(l2, axis)
    prediction = aes_model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
    if is_batched:
        return prediction[:, 0].tolist()
    else:
        return prediction.item()
    
def get_imagereward_score(image_reward, device, tokenizer, images, prompts):
    prompts = prompts.repeat_interleave(3, dim=0)
    prompts = tokenizer.batch_decode(prompts, skip_special_tokens=True)
    images = [clip_preprocess(i) for i in images]
    images = torch.stack(images).to(device)
    return image_reward(images, prompts)

def get_results(pipe,
                clip_model,
                clip_preprocess,
                aes_model,
                device,
                tokenizer, 
                prompt_inputs: Dict[str, Union[List, torch.LongTensor]],
                prompts_responses: torch.LongTensor,
                num_inference_steps=4,
                guidance_scale=0.0,
                bsz=16,
                num_images_per_prompt=3,
                threshold=0.28,
                reward_metric="aes",
                seed=42,
                image_reward=None,
                manual=False):
    
    prompt_inputs = prompt_inputs
    prompts_responses = prompts_responses.to(device)
    
    prompt_len = prompt_inputs["input_ids"].size(1)
    only_prompts = prompts_responses[:, :prompt_len]
    if manual:
        only_responses = only_prompts
        
        # aesthetic reward
        images = gen_image_batched(pipe, device, tokenizer, only_responses, 
                                num_images_per_prompt,
                                num_inference_steps,
                                guidance_scale,
                                bsz,
                                seed)
        image_features = get_clip_features(clip_model, clip_preprocess, device, images, is_batched=True)
        if reward_metric == "aes":
            aes_scores = get_aesthetic_score(aes_model, device, image_features, is_batched=True)
        elif reward_metric == "imagereward":
            aes_scores = get_imagereward_score(image_reward, device, tokenizer, images, only_responses)
            
        aes_scores = torch.FloatTensor(aes_scores).reshape(-1, num_images_per_prompt).mean(dim=-1).to(device)
        aes_reward = aes_scores - aes_scores
        
        # clip scores reward
        clip_scores = get_clip_score_batched(clip_model, device, tokenizer, image_features, only_prompts)
        clip_scores = torch.FloatTensor(clip_scores).to(device)
        clip_scores = torch.maximum(clip_scores, torch.zeros_like(clip_scores))
        clip_reward = torch.where(clip_scores>threshold, 0, 20*clip_scores-20*threshold).reshape(-1, num_images_per_prompt).mean(-1)
        
        decoded_responses = tokenizer.batch_decode(only_responses, skip_special_tokens=True)
        return aes_scores, aes_scores, aes_reward, clip_scores, clip_scores, clip_reward, decoded_responses, images, images    
    else:
        only_responses = prompts_responses[:, prompt_len:]
    
        # aesthetic reward
        images = gen_image_batched(pipe, device, tokenizer, only_responses, 
                                num_images_per_prompt,
                                num_inference_steps,
                                guidance_scale,
                                bsz,
                                seed)
        image_features = get_clip_features(clip_model, clip_preprocess, device, images, is_batched=True)
        if reward_metric == "aes":
            aes_scores = get_aesthetic_score(aes_model, device, image_features, is_batched=True)
        elif reward_metric == "imagereward":
            aes_scores = get_imagereward_score(image_reward, device, tokenizer, images, only_responses)
        
        images_plain = gen_image_batched(pipe, device, tokenizer, only_prompts,
                                        num_images_per_prompt,
                                        num_inference_steps,
                                        guidance_scale,
                                        bsz,
                                        seed=42)
        images_plain_features = get_clip_features(clip_model, clip_preprocess, device, images_plain, is_batched=True)
        if reward_metric == "aes":
            aes_scores_plain = get_aesthetic_score(aes_model, device, images_plain_features, is_batched=True)
        elif reward_metric == "imagereward":
            aes_scores_plain = get_imagereward_score(image_reward, device, tokenizer, images_plain, only_prompts)
        
        aes_scores = torch.FloatTensor(aes_scores).reshape(-1, num_images_per_prompt).mean(dim=-1).to(device)
        aes_scores_plain = torch.FloatTensor(aes_scores_plain).reshape(-1, num_images_per_prompt).mean(dim=-1).to(device)
        aes_reward = aes_scores - aes_scores_plain
        
        # clip scores reward
        clip_scores = get_clip_score_batched(clip_model, device, tokenizer, image_features, only_prompts)
        clip_scores = torch.FloatTensor(clip_scores).to(device)
        clip_scores = torch.maximum(clip_scores, torch.zeros_like(clip_scores))
        clip_reward = torch.where(clip_scores>threshold, 0, 20*clip_scores-20*threshold).reshape(-1, num_images_per_prompt).mean(-1)
        
        clip_scores_plain = get_clip_score_batched(clip_model, device, tokenizer, images_plain_features, only_prompts)
        clip_scores_plain = torch.FloatTensor(clip_scores_plain).to(device)
        clip_scores_plain = torch.maximum(clip_scores_plain, torch.zeros_like(clip_scores_plain))
        
        decoded_responses = tokenizer.batch_decode(only_responses, skip_special_tokens=True)
        return aes_scores, aes_scores_plain, aes_reward, clip_scores, clip_scores_plain, clip_reward, decoded_responses, images, images_plain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="debug")
    parser.add_argument("--lm_name", type=str, default="gpt2")
    parser.add_argument("--eval_file", type=str, default="prompts/eval_prompt_coco.jsonl")
    parser.add_argument("--sampler", type=str, default="dpm_solver")
    parser.add_argument("--eval_sd_name", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--num_images_per_prompt", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.28)
    parser.add_argument("--temperature", type=str)
    parser.add_argument("--maxlen", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward_metric", type=str, default="aes")
    args = parser.parse_args()
    
    seed(args.seed)
    
    device = torch.cuda.current_device()
    
    config = AutoConfig.from_pretrained(args.lm_name)
    prompter_model = AutoModelForCausalLM.from_pretrained(args.lm_name, config=config)
    
    dirs = sorted(os.listdir(args.checkpoint))
    dirs = [int(x) for x in dirs if x.isdigit()]
    dirs = sorted(dirs, reverse=True)
    ckpt_dir = os.path.join(args.checkpoint, str(dirs[0]))
    print(ckpt_dir)
    
    _prompter_model = AutoModelForCausalLM.from_pretrained(ckpt_dir)
    msg = prompter_model.load_state_dict(_prompter_model.state_dict(), strict=False)
    print(msg)
    prompter_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    prompter_model = prompter_model.to(device)
    
    eval_dataloader = get_dataloader(
        "prompt-adaptation", tokenizer, prompt_file=args.eval_file,
        batch_size=1, shuffle=False)
    
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.eval_sd_name,
        torch_dtype=torch.float16,
    )
    
    if args.sampler == "dpm_solver":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        num_inference_steps = 20
        guidance_scale = 7.5
        bsz = 1
    elif args.sampler == "ddpm":
        num_inference_steps = 20
        guidance_scale = 7.5
        bsz = 1
        
    pipe.safety_checker = None
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    
    aes_model = AestheticMlp(768)
    aes_model.load_state_dict(torch.load("aesthetic/sac+logos+ava1-l14-linearMSE.pth"))
    aes_model = aes_model.to(device)
    aes_model.eval()
    
    if args.reward_metric == "imagereward":
        image_reward = imagereward(dtype=torch.float32, device=device)
    
    sentence_encoder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    
    inception_block_idx = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()
    
    aes_scores_list = []
    aes_scores_plain_list = []
    aes_reward_list = []
    clip_scores_list = []
    clip_scores_plain_list = []
    clip_reward_list = []
    decoded_responses_list = []
    images_list = []
    images_plain_list = []
    cos_sims = []
    cos_sim_images = []
    logits_list = []
    
    for batch in tqdm(eval_dataloader):
        batch = batch.to(device)
        prompter_model.eval()

        outputs = prompter_model.generate(
            **batch,
            do_sample=False, 
            min_new_tokens=15,
            max_new_tokens=75, 
            num_beams=args.num_samples, 
            num_beam_groups=args.num_samples,
            num_return_sequences=args.num_samples, 
            eos_token_id=tokenizer.pad_token_id, 
            pad_token_id=tokenizer.pad_token_id, 
            length_penalty=-1.0,
            diversity_penalty=1.0,
            output_logits=True,
            return_dict_in_generate=True,
        )
        prompts_responses = outputs[0]
        
        if args.reward_metric == "aes":
            results = get_results(pipe, clip_model, clip_preprocess, aes_model, device, tokenizer, batch, prompts_responses, 
                                num_inference_steps,
                                guidance_scale,
                                bsz,
                                args.num_images_per_prompt, 
                                args.threshold,
                                args.reward_metric,
                                args.seed,
                                manual=True if args.checkpoint == "manual" else False)
        elif args.reward_metric == "imagereward":
            results = get_results(pipe, clip_model, clip_preprocess, aes_model, device, tokenizer, batch, prompts_responses, 
                                num_inference_steps,
                                guidance_scale,
                                bsz,
                                args.num_images_per_prompt, 
                                args.threshold,
                                args.reward_metric,
                                args.seed,
                                image_reward)
        aes_scores, aes_scores_plain, aes_reward, clip_scores, clip_scores_plain, clip_reward, decoded_responses, images, images_plain = results
                
        # Measure diversity per prompt
        embs = sentence_encoder.encode(decoded_responses)
        embs = torch.from_numpy(embs)
        cos_sim = batch_cosine_similarity_kernel(embs)
        cos_sims.append(cos_sim)
        
        # Mesure diversity of images
        images_torch = torch.stack([torch.from_numpy(np.array(image)).float() for image in images])
        images_torch = rearrange(images_torch, 'b h w c -> b c h w')
        images_torch = images_torch.to(device)

        features = inception_model(images_torch)[0]
        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        cos_sim_image = batch_cosine_similarity_kernel(features)
        cos_sim_images.append(cos_sim_image)
        
        aes_scores_list.append(aes_scores.cpu().detach().numpy())
        aes_scores_plain_list.append(aes_scores_plain.cpu().detach().numpy())
        aes_reward_list.append(aes_reward.cpu().detach().numpy())

        clip_scores_list.append(clip_scores.reshape(-1, args.num_samples).cpu().detach().numpy())
        clip_scores_plain_list.append(clip_scores_plain.reshape(-1, args.num_samples).cpu().detach().numpy())
        clip_reward_list.append(clip_reward.cpu().detach().numpy())
        decoded_responses_list.append(decoded_responses)
        images_list.append([np.array(image) for image in images])
        images_plain_list.append([np.array(image_plain) for image_plain in images_plain])
    
    aes_scores_list = np.stack(aes_scores_list)
    aes_scores_plain_list = np.stack(aes_scores_plain_list)
    aes_reward_list = np.stack(aes_reward_list)
    clip_scores_list = np.stack(clip_scores_list)
    clip_scores_plain_list = np.stack(clip_scores_plain_list)
    clip_reward_list = np.stack(clip_reward_list)
    images_list = np.stack(images_list)
    images_plain_list = np.stack(images_plain_list)
    
    # save results
    folder_name = f"results/{args.checkpoint.replace('/', '-')}/{args.eval_file.replace('prompts/', '').replace('.jsonl', '')}_{args.eval_sd_name.replace('/', '_')}/"
    folder_name = f"{folder_name}_seed{args.seed}_{args.reward_metric}_{args.num_samples}"
    os.makedirs(folder_name, exist_ok=True)
    np.savez_compressed(f"{folder_name}/results.npz",
                        cos_sim=cos_sims,
                        cos_sim_images=cos_sim_images,
                        aes_scores=aes_scores_list,
                        aes_scores_plain=aes_scores_plain_list,
                        aes_reward=aes_reward_list,
                        clip_scores=clip_scores_list,
                        clip_scores_plain=clip_scores_plain_list,
                        clip_reward=clip_reward_list,
                        decoded_responses=decoded_responses_list,
                       )
    
    # save images
    os.makedirs(f"{folder_name}/images", exist_ok=True)
    for i in range(len(images_list)):
        for j in range(args.num_samples):
            for k in range(args.num_images_per_prompt):
                Image.fromarray(images_list[i][j*args.num_images_per_prompt+k]).save(f"{folder_name}/images/images_{i}_{j}_{k}.png")
                if j == 0:
                    Image.fromarray(images_plain_list[i][j*args.num_images_per_prompt+k]).save(f"{folder_name}/images/images_plain_{i}_{j}_{k}.png")
        
        
    