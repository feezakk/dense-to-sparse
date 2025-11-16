# Usage

## Training Teacher on Dense Reward on CARLA Overtaking and Lane Following Tasks 

### 1. Train Teacher Model - Overtaking

   Train the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash train_dm3_teacher.sh 3000 0 \
     --task carla_overtake \
     --dreamerv3.logdir ./logdir/carla_overtake_teacher/
   ```

### 2. Train Teacher Model - Lane Following

   Train the DreamerV3 teacher model for the lane following task:

   ```bash
   bash train_dm3_teacher.sh 3000 0 \
     --task carla_lane_following \
     --dreamerv3.logdir ./logdir/carla_lane_following_teacher/
   ```

## Distillation on CARLA Overtaking and Lane Following Tasks 

### 1. Train Student Model - Lane Following with distillation

   Run the DreamerV3 student without bisimulation:

   ```bash
   bash train_dm3_student_bisim.sh 3000 0 \
     --task carla_lane_following_student \
     --dreamerv3.logdir ./logdir/carla_lane_following_student/ \
     --dreamerv3.enable_bisim=False
   ```

### 2. Train Student Model - Overtaking with distillation

   Run the DreamerV3 student without bisimulation:

   ```bash
   bash train_dm3_student_bisim.sh 3000 0 \
     --task carla_overtake_student \
     --dreamerv3.logdir ./logdir/carla_overtake_student/ \
     --dreamerv3.enable_bisim=False
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

## Train Student on Sparse Rewards on CARLA Overtaking and Lane Following Tasks 

### 1. Train  Student Model on Sparse Rewards - Overtaking

   Train the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash train_dm3_teacher.sh 3000 0 \
     --task carla_overtake_student \
     --dreamerv3.logdir ./logdir/carla_overtaking_teacher/
   ```

### 2. Train  Student Model on Sparse Rewards - Lane Following

   Train the DreamerV3 teacher model for the lane following task:

   ```bash
   bash train_dm3_teacher.sh 3000 0 \
     --task carla_lane_following_student \
     --dreamerv3.logdir ./logdir/carla_lane_following_teacher/
   ```

 

### Evaluate Teacher - Overtaking Seen

   Evaluate the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_overtake/checkpoint.ckpt --task carla_overtake --dreamerv3.logdir ./logdir/eval_overtake_teacher_seen
   ```

### Evaluate Teacher - Overtaking Unseen

   Evaluate the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_overtake/checkpoint.ckpt --task carla_overtake_test --dreamerv3.logdir ./logdir/eval_overtake_teacher_unseen
   ```

### Evaluate Student - Overtaking Seen

   Evaluate the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_overtake_student/checkpoint.ckpt --task carla_overtake_student --dreamerv3.logdir ./logdir/eval_overtake_student_seen
   ```

### Evaluate Student - Overtaking Unseen

   Evaluate the DreamerV3 teacher model for the overtaking task:

   ```bash
   bash eval_dm3_teacher.sh 3000 0 ./logdir/carla_overtake_student/checkpoint.ckpt --task carla_overtake_student_test --dreamerv3.logdir ./logdir/eval_overtake_student_unseen
   ```