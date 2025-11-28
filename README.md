# Usage

# 1. Training Setups for CARLA Overtaking and Lane Following Tasks      

## 1. Training Teacher on Dense Reward on CARLA Overtaking and Lane Following Tasks 

### 1. Train Teacher Model - Overtaking

   Train the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash train_dm3_teacher.sh 3000 0 --task carla_overtake --dreamerv3.logdir ./logdir/carla_overtake_teacher/ --dreamerv3.run.steps=100000
   ```

### 2. Train Teacher Model - Lane Following

   Train the DreamerV3 teacher model for the lane following task:

   ```bash
   bash train_dm3_teacher.sh 3000 0 --task carla_lane_following --dreamerv3.logdir ./logdir/carla_lane_following_teacher/ --dreamerv3.run.steps=100000
   ```

## 2. Baseline Hindesight Experience Replay (HER) without distillation on CARLA Overtaking and Lane Following Tasks

### 1. Train Model - Lane Following with HER without distillation

   Run the DreamerV3 with HER without distillation for lane following task:

   ```bash
   bash train_dm3_HER.sh 3000 0 --task carla_lane_following_student --dreamerv3.logdir ./logdir/carla_lane_following_HER_without_distillation/ --dreamerv3.run.steps=100000
   ```

### 2. Train Model - Overtaking without HER without distillation

   Run the DreamerV3 with HER without distillation for overtaking task:

   ```bash
   bash train_dm3_HER.sh 3000 0 --task carla_overtake_student --dreamerv3.logdir ./logdir/carla_overtake_HER_without_distillation/ --dreamerv3.run.steps=100000
   ```

## 3. Baseline Hindesight Experience Replay (HER) with distillation on CARLA Overtaking and Lane Following Tasks

### 1. Train Model - Lane Following with HER with distillation

   Run the DreamerV3 with HER with distillation for lane following task:

   ```bash
   bash train_dm3_student_HER.sh 3000 0 --task carla_lane_following_student --dreamerv3.logdir ./logdir/carla_lane_following_HER_with_distillation/ --dreamerv3.run.steps=100000
   ```

### 2. Train Model - Overtaking with HER without distillation

   Run the DreamerV3 with HER with distillation for overtaking task:

   ```bash
   bash train_dm3_student_HER.sh 3000 0 --task carla_overtake_student --dreamerv3.logdir ./logdir/carla_overtake_HER_with_distillation/ --dreamerv3.run.steps=100000
   ```

## 4. Baseline Distillation on CARLA Overtaking and Lane Following Tasks 

### 1. Train Student Model - Lane Following with distillation

   Run the DreamerV3 student for lane following without bisimulation:

   ```bash
   bash train_dm3_student_bisim.sh 3000 0 \
     --task carla_lane_following_student \
     --dreamerv3.logdir ./logdir/carla_lane_following_student/ \
     --dreamerv3.enable_bisim=False \
     --dreamerv3.loss_scales.posterior_deter_kl=1.0 \
     --dreamerv3.loss_scales.posterior_stoch_kl=1.0 \
     --dreamerv3.loss_scales.prior_deter_kl=1.0 \
     --dreamerv3.loss_scales.prior_stoch_kl=1.0 \
     --dreamerv3.run.steps=100000
   ```

### 2. Train Student Model - Overtaking with distillation

   Run the DreamerV3 student for overtaking without bisimulation:

   ```bash
   bash train_dm3_student_bisim.sh 3000 0 \
     --task carla_overtake_student \
     --dreamerv3.logdir ./logdir/carla_overtake_student/ \
     --dreamerv3.enable_bisim=False \
     --dreamerv3.loss_scales.posterior_deter_kl=1.0 \
     --dreamerv3.loss_scales.posterior_stoch_kl=1.0 \
     --dreamerv3.loss_scales.prior_deter_kl=1.0 \
     --dreamerv3.loss_scales.prior_stoch_kl=1.0 \
     --dreamerv3.run.steps=100000
   ```

### 3. Run without bisim with loss scaling - Lane Following

   Run the DreamerV3 student without bisimulation but with loss scaling:

   ```bash
   bash train_dm3_student_bisim.sh 3000 0 \
    --task carla_lane_following_student \
    --dreamerv3.logdir ./logdir/carla_lane_following_student/ \
    --dreamerv3.enable_bisim=False \
    --dreamerv3.loss_scales.posterior_deter_kl=5.0 \
    --dreamerv3.loss_scales.posterior_stoch_kl=5.0 \
    --dreamerv3.loss_scales.prior_deter_kl=5.0 \
    --dreamerv3.loss_scales.prior_stoch_kl=5.0
   ```

## 5. Train Student on Sparse Rewards on CARLA Overtaking and Lane Following Tasks 

### 1. Train  Student Model on Sparse Rewards - Overtaking

   Train the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash train_dm3_teacher.sh 3000 0 \
     --task carla_overtake_student \
     --dreamerv3.logdir ./logdir/carla_overtaking_student_sparse/ \
     --dreamerv3.run.steps=100000
   ```

### 2. Train  Student Model on Sparse Rewards - Lane Following

   Train the DreamerV3 teacher model for the lane following task:

   ```bash
   bash train_dm3_teacher.sh 3000 0 \
     --task carla_lane_following_student \
     --dreamerv3.logdir ./logdir/carla_lane_following_student_sparse/\
     --dreamerv3.run.steps=100000
   ```

# 2. Evaluation Setups for CARLA Overtaking and Lane Following Tasks

## 1. Evaluation on CARLA Overtaking 

### Evaluate Teacher - Overtaking Seen

   Evaluate the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_overtake_teacher/checkpoint.ckpt --task carla_overtake --dreamerv3.logdir ./eval_logdir/eval_overtake_teacher_seen
   ```

### Evaluate Teacher - Overtaking Unseen

   Evaluate the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_overtake_teacher/checkpoint.ckpt --task carla_overtake_test --dreamerv3.logdir ./eval_logdir/eval_overtake_teacher_unseen
   ```

### Evaluate Student - Overtaking Seen

   Evaluate the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_overtake_student/checkpoint.ckpt --task carla_overtake_student --dreamerv3.logdir ./eval_logdir/eval_overtake_student_seen
   ```

### Evaluate Student - Overtaking Unseen

   Evaluate the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_overtake_student/checkpoint.ckpt --task carla_overtake_student_test --dreamerv3.logdir ./eval_logdir/eval_overtake_student_unseen
   ```

### Evaluate Student on Sparse Reward - Overtaking Seen

   Evaluate the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_overtaking_student_sparse/checkpoint.ckpt --task carla_overtake_student --dreamerv3.logdir ./eval_logdir/eval_overtake_student_sparse_seen
   ```

### Evaluate Student on Sparse Reward - Overtaking Unseen

   Evaluate the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_overtaking_student_sparse/checkpoint.ckpt --task carla_overtake_student_test --dreamerv3.logdir ./eval_logdir/eval_overtake_student_sparse_unseen
   ```

### Evaluate Student on Sparse Reward and HER without Distillation - Overtaking Seen

   Evaluate the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_overtake_HER_without_distillation/checkpoint.ckpt --task carla_overtake_student --dreamerv3.logdir ./eval_logdir/eval_overtake_student_HER_without_distillation_seen
   ```

### Evaluate Student on Sparse Reward and HER without Distillation - Overtaking Unseen

   Evaluate the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_overtake_HER_without_distillation/checkpoint.ckpt --task carla_overtake_student_test --dreamerv3.logdir ./eval_logdir/eval_overtake_student_HER_without_distillation_unseen
   ```

### Evaluate Student on Sparse Reward and HER with Distillation - Overtaking Seen

   Evaluate the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_overtake_HER_with_distillation/checkpoint.ckpt --task carla_overtake_student --dreamerv3.logdir ./eval_logdir/eval_overtake_student_HER_with_distillation_seen
   ```

### Evaluate Student on Sparse Reward and HER with Distillation - Overtaking Unseen

   Evaluate the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_overtake_HER_with_distillation/checkpoint.ckpt --task carla_overtake_student_test --dreamerv3.logdir ./eval_logdir/eval_overtake_student_HER_with_distillation_unseen
   ```

## 2. Evaluation Teacher on CARLA Lane Following Tasks  

### Evaluate Teacher - Lane Following Seen

   Evaluate the DreamerV3 teacher model for the lane following task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_lane_following_teacher/checkpoint.ckpt --task carla_lane_following --dreamerv3.logdir ./eval_logdir/eval_lane_following_teacher_seen
   ```

### Evaluate Teacher - Lane Following Unseen

   Evaluate the DreamerV3 teacher model for the lane follwoing task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_lane_following_teacher/checkpoint.ckpt --task carla_lane_following_test --dreamerv3.logdir ./eval_logdir/eval_lane_following_teacher_unseen
   ```

### Evaluate Student - Lane Following Seen

   Evaluate the DreamerV3 student model for the lane following task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_lane_following_student/checkpoint.ckpt --task carla_lane_following_student --dreamerv3.logdir ./eval_logdir/eval_lane_following_student_seen
   ```

### Evaluate Student - Lane Following Unseen

   Evaluate the DreamerV3 teacher model for the lane following task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_lane_following_student/checkpoint.ckpt --task carla_lane_following_student_test --dreamerv3.logdir ./eval_logdir/eval_lane_following_student_unseen
   ```

### Evaluate Student on Sparse Reward - Lane Following Seen

   Evaluate the DreamerV3 teacher model for the lane following task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_lane_following_student_sparse/checkpoint.ckpt --task carla_lane_following_student --dreamerv3.logdir ./eval_logdir/eval_lane_following_student_sparse_seen
   ```

### Evaluate Student on Sparse Reward - Lane Following Unseen

   Evaluate the DreamerV3 teacher model for the lane following task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_lane_following_student_sparse/checkpoint.ckpt --task carla_lane_following_student_test --dreamerv3.logdir ./eval_logdir/eval_lane_following_student_sparse_unseen
   ```

### Evaluate Student on Sparse Reward and HER without Distillation - Lane Following Seen

   Evaluate the DreamerV3 teacher model for the lane following task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_lane_following_HER_without_distillation/checkpoint.ckpt --task carla_lane_following_student --dreamerv3.logdir ./eval_logdir/eval_lane_following_student_HER_without_distillation_seen
   ```

### Evaluate Student on Sparse Reward and HER without Distillation - Lane Following Unseen

   Evaluate the DreamerV3 teacher model for the lane following task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_lane_following_HER_without_distillation/checkpoint.ckpt --task carla_lane_following_student_test --dreamerv3.logdir ./eval_logdir/eval_lane_following_student_HER_without_distillation_unseen
   ```

### Evaluate Student on Sparse Reward and HER with Distillation - Lane Following Seen

   Evaluate the DreamerV3 teacher model for the lane following task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_lane_following_HER_with_distillation/checkpoint.ckpt --task carla_lane_following_student --dreamerv3.logdir ./eval_logdir/eval_lane_following_student_HER_with_distillation_seen
   ```

### Evaluate Student on Sparse Reward and HER with Distillation - Lane Following Unseen

   Evaluate the DreamerV3 teacher model for the lane following task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_lane_following_HER_with_distillation/checkpoint.ckpt --task carla_lane_following_student_test --dreamerv3.logdir ./eval_logdir/eval_lane_following_student_HER_with_distillation_unseen
   ```

