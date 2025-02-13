# PAG

Official Code for Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation

### Environment Setup
```
conda create -n pag python=3.8 -y
conda activate pag

pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

Please download pre-trained model and dataset from the following link: https://github.com/microsoft/LMOps/tree/main/promptist

Place pretrained SFT model into `save/` folder and prompts into `prompts/` folder and execute `convert_format.ipynb` file

### Training
```
python main.py --mode train --exp_name pag \
               --lm_name gpt2 --sd_name CompVis/stable-diffusion-v1-4 --reward_metric aes \
               --flow_reset True --flow_reset_period 2000 \  # Flow Reactivation
               --prioritization reward \ # Reweighted Training
               --loss fl-db \ # Reward Decomposition
               --batch_size 16 --grad_acc_steps 1
```

### Evaluation
```
python eval.py --checkpoint save/pag \
               --lm_name gpt2 --eval_sd_name CompVis/stable-diffusion-v1-4 --reward_metric aes \
               --flow_reset True --flow_reset_period 2000 \  # Flow Reactivation
               --prioritization reward \ # Reweighted Training
               --loss fl-db \ # Reward Decomposition
               --eval_prompt_file prompts/eval_prompt_chatgpt.jsonl
```