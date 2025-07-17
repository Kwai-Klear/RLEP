# [RLEP: Reinforcement Learning with Experience Replay for LLM Reasoning](https://arxiv.org/abs/2507.07451)

<p align="center">
  <img src="image/trip_demo.png" width="50%">
</p>

> RL training is an energy‑intensive journey. Leveraging collected experience, RLEP rapidly converges on promising reasoning paths and achieves stronger final results.


# Summary 
*RLEP*—**R**einforcement **L**earning with **E**xperience re**P**lay—first collects verified successful trajectories and then replays them during subsequent training. At every update step, the policy is optimized on mini‑batches that blend newly generated rollouts with these replayed successes. By replaying high‑quality examples, RLEP steers the model away from fruitless exploration, focuses learning on promising reasoning paths, and delivers both faster convergence and stronger final performance.

<p align="center">
  <img src="image/rlep_method.png" width="85%">
</p>

Experimental results show that RLEP reaches the vanilla‑RL baseline’s best score much sooner —and ultimately pushes that score even higher.
* **Rapid early gains.** On AIME‑2024 RLEP hits the baseline’s peak accuracy by step 135 (the baseline needs 380). On AIME‑2025 it surpasses the baseline’s best score after only 50 steps.

* **Higher final performance.** RLEP ultimately lifts the peak accuracy from 38.2 % → 39.9 % (AIME‑2024), 19.8 % → 22.3 % (AIME‑2025), and 77.0 % → 82.2 % on AMC‑2023 benchmark. 

<p align="center">
  <img src="image/exp_acc.png" width="85%">
</p>

# Training
The model is trained with **VERL**, using **vLLM** as the inference engine.
## Install 
```bash 
git clone https://github.com/Kwai-Klear/RLEP.git
cd rlep
pip3 install -e .[vllm]
```

## DAPO baseline 
### 1.DAPO-nodyn-bs64
Download the datasets and pretrained checkpoint
```bash
wget https://huggingface.co/datasets/Kwai-Klear/RLEP_dataset/resolve/main/dapo_format_aime2024_aime2025_amc2023.parquet # eval set
wget https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet # trianing set
git clone https://huggingface.co/Qwen/Qwen2.5-Math-7B
```
Update paths in `recipe/rlep/dapo_nodyn_bs_64.sh`
```bash
TRAIN_FILE=TheLocalPathTo/RLEP_dataset/dapo-math-17k.parquet
TEST_FILE=TheLocalPathTo/RLEP_dataset/dapo_format_aime2024_aime2025_amc2023.parquet
MODEL_PATH=TheLocalPathTo/Qwen2.5-Math-7B
CKPTS_DIR=TheLocalPathTo/save/${project_name}/${exp_name}
```
Launch Ray workers (multi‑node training).
The default expects 8 nodes—adjust as needed.
See the VERL documentation: https://verl.readthedocs.io/en/latest/start/multinode.html for details.

Start training
```
bash recipe/rlep/dapo_nodyn_bs_64.sh
```

### 2.DAPO official &&  DAPO-nodyn-bs32(optional)
Please refer to `recipe/rlep/official_dapo.sh` & `recipe/rlep/rlep_training.sh`; the configuration pattern is identical.

## Experience Collection
Run inference with the model trained in DAPO‑nodyn‑bs64 (we used the checkpoint at 400 steps) and append the collected trajectories to the training set.
```bash
Inference code and data process script tbd.
```

We also provide the final RLEP training set, `dapo-math-17k-with-experience-pool.parquet`. You can also launch RLEP training directly with this dataset.
```bash
git clone https://huggingface.co/datasets/Kwai-Klear/RLEP_dataset
cd RLEP_dataset
# concatenate the pieces in order
cat dapo-math-17k-with-experience-pool.parquet.part-* \
    > dapo-math-17k-with-experience-pool.parquet
# the trianing dataet is RLEP_dataset/dapo-math-17k-with-experience-pool.parquet 
```

## RL with Experience Replay
Set the paths for the `training set with experience pool`, `evaluation set`, `pretrained checkpoint`, and `output directory`.

```bash
TRAIN_FILE=TheLocalPathTo/RLEP_dataset/dapo-math-17k-with-experience-pool.parquet
TEST_FILE=TheLocalPathTo/RLEP_dataset/dapo_format_aime2024_aime2025_amc2023.parquet
MODEL_PATH=TheLocalPathTo/Qwen2.5-Math-7B
CKPTS_DIR=TheLocalPathTo/save/${project_name}/${exp_name}
```

```bash 
bash recipe/rlep/rlep_training.sh
```

We released our trained CKPT (at 320 steps) at https://huggingface.co/Kwai-Klear/qwen2.5-math-rlep


## Evaluation
```bash
python inference/eval_main.py --model_path  xx
```
We evaluated the converged RLEP model at 320 training steps and the DAPO‑nodyn‑bs64 baseline at 400 steps.

|                   | AIME-2024 | AIME-2025 | AMC-2023 |
|-------------------|-----------|-----------|----------|
| DAPO              | 32.6      | 18.9      | 77.5     |
| DAPO\-nodyn\-bs64 | 37.4      | 19.4      | 77.3     |
| RLEP              | 38.5      | 21.3      | 83.0     |



# Citation
If you find our paper or code helpful, we would appreciate it if you could cite our work:

```
@misc{zhang2025rlepreinforcementlearningexperience,
      title={RLEP: Reinforcement Learning with Experience Replay for LLM Reasoning}, 
      author={Hongzhi Zhang and Jia Fu and Jingyuan Zhang and Kai Fu and Qi Wang and Fuzheng Zhang and Guorui Zhou},
      year={2025},
      eprint={2507.07451},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.07451}, 
}
```

# Acknowledgement 
We conducted our experiments with the [VERL](https://github.com/volcengine/verl) framework and the [Qwen2.5‑7B‑Math](https://huggingface.co/Qwen/Qwen2.5-Math-7B) model, using the dataset and training scripts provided by [DAPO](https://dapo-sia.github.io/). 
Many thanks to the open‑sourced works and the broader community for making these resources available!
