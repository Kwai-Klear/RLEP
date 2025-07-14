# DAPO based Experiment

## * 标准DAPO实验

```bash
bash recipe/dapo/official_dapo_math_7b.sh
```

[wandb log](https://wandb.ai/hongzhi/ForRelease/runs/k9e43qfr)  
模型存储路径：  
`/nlp_group/zhanghongzhi/verl/save/ForRelease/DAPO-Qwen2.5-7b-MATH-0529_real_offcial-3k`
这个结果还可以，能够到34%左右；

---

[另外一个实验,8k context](https://wandb.ai/hongzhi/ForRelease/runs/pczps9vo/workspace?nw=nwuserhongzhi)  
模型存储路径：  
`/nlp_group/zhanghongzhi/verl/save/ForRelease/DAPO-Qwen2.5-7b-MATH-0527_offcial`  
一直在涨，但是只到30（还有增长趋势倒是）  
问题是DAPO要跑到33才能认为自己跑的版本没问题...  
要不跑一个7K Context的实验？

---

修好RM重新在verl_after_sft之后的版本，跑一个训练。4台机器。  
模型存储路径：  
`/nlp_group/zhanghongzhi/verl/save/ForRelease/DAPO-Qwen2.5-7b-MATH-0529_offcial-3k-fix-rm`

```bash
cd /nlp_group/zhanghongzhi/verl_after_sft/
bash recipe/dapo/dapo_7b_baseline_v0530_rm_fix.sh
```

[wandb](https://wandb.ai/hongzhi/ForRelease/runs/i6f0r2ji)

---

## * 标准DAPO实验，关闭3倍Rollout采样(这个实验可以跑多个)

```bash
bash recipe/dapo/official_dapo_math_7b_no_3x_rollout.sh
bash recipe/dapo/official_dapo_math_7b_no_3x_rollout_r2.sh # 跑第二次,对比上面开启了数据shuffle，默认seed为1
bash recipe/dapo/official_dapo_math_7b_no_3x_rollout_r3.sh # 跑第三次，对比上面设置seed为3 8台在走之前把这个跑了，回来可以跑exp组了
```
先跑一个来看跟标准DAPO的差距，我记得应该是好一点还；
