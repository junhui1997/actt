#conda activate surrol_aloha
cd ..
for model in ACT
do
  for task_name in PegTransfer-v0
do
python -u imitate_episodes.py \
--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-rope_s42/ --policy_class $model --kl_weight 200 --chunk_size 20 --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --state_dim 7 --action_dim 7 --num_steps 120000 --lr 75e-7 --seed 42 --is_surgical  --eval_every 2500 --save_every 5000 --temporal_agg --is_joint
done
done

# single arm
#for model in ACT
#do
#  for task_name in  PegTransfer-v0
#do
#python -u imitate_episodes.py \
#--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-rope/ --policy_class $model --kl_weight 200 --chunk_size 20 --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --state_dim 7 --action_dim 7 --num_steps 100000 --lr 75e-7 --seed 0 --is_surgical  --eval_every 2500 --save_every 5000 --temporal_agg --is_joint
#done
#done

#for model in ACT
#do
#  for task_name in  NeedleReach-v0
#do
#python -u imitate_episodes.py \
#--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-rope/ --policy_class $model --kl_weight 200 --chunk_size 20 --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --state_dim 7 --action_dim 5 --num_steps 20000 --lr 75e-7 --seed 0 --is_surgical  --eval_every 2500 --save_every 5000 --temporal_agg --is_joint
#done
#done


#for model in ACT
#do
#  for task_name in  BiPegTransfer-v0
#do
#  for chunk_size in  40
#do
#python -u imitate_episodes.py \
#--task_name $task_name --ckpt_dir checkpoint_m/$model/$task_name-rope30/ --policy_class $model --kl_weight 400 --chunk_size $chunk_size --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --state_dim 14 --action_dim 14 --num_steps 100000 --lr 1e-5 --seed 0 --is_surgical  --eval_every 2500 --save_every 5000 --temporal_agg --is_joint --resume_ckpt_path checkpoint_m/$model/$task_name-rope30/policy_last.ckpt
#done
#done
#done


