## Usage

1. **Run without bisim**

   Run the DreamerV3 student without bisimulation:

   ```bash
   bash train_dm3_student_bisim.sh 3000 0 \
     --task carla_lane_following_student \
     --dreamerv3.logdir ./logdir/carla_lane_following_student/ \
     --dreamerv3.enable_bisim=False

2. **Run without bisim with loss scaling**

   Run the DreamerV3 student without bisimulation but with loss scaling:

   ```bash
   bash train_dm3_student_bisim.sh 3000 0 
    --task carla_lane_following_student 
    --dreamerv3.logdir ./logdir/carla_lane_following_student/ 
    --dreamerv3.enable_bisim=False 
    --dreamerv3.loss_scales.posterior_deter_kl=5.0
    --dreamerv3.loss_scales.posterior_stoch_kl=5.0
    --dreamerv3.loss_scales.prior_deter_kl=5.0
    --dreamerv3.loss_scales.prior_stoch_kl=5.0
   ```
