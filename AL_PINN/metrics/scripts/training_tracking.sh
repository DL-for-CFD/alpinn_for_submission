tmux
zsh
conda activate pinn

cd AL_PINN/autoencoder
export CUDA_VISIBLE_DEVICES=


tmux attach -t 
tmux kill-session


/project/t3_zxiaoal/Validation_Dataset/dt4_rlpinn_normalized
/project/t3_zxiaoal/Validation_Dataset/dt4_baseline_normalized


for task in airfoil complex cylinder
do
i=0
while [ $i -le 149 ]; do
  python AL_PINN/metrics/run.py --gt_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_overfitted/$task\_normalized --ds_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_rlpinn_normalized/$task$i --feature=tracking --store_dir=AL_PINN/metrics/res/tracking/$task$i > AL_PINN/metrics/log/tracking_pinn_$task$i.log
  if [ $i -eq 0 ]; then
    i=$((i+9))
  else
    i=$((i+10))
  fi
done
done


# !baseline
for task in cylinder
do
i=0
while [ $i -le 199 ]; do
  python AL_PINN/metrics/run.py --gt_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_overfitted/$task\_normalized --ds_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_baseline_normalized/$task$i --feature=tracking --store_dir=AL_PINN/metrics/res/tracking/baseline_$task$i > AL_PINN/metrics/log/tracking_baseline_$task$i.log
  if [ $i -eq 0 ]; then
    i=$((i+9))
  else
    i=$((i+10))
  fi
done
done

complex cylinder
for task in airfoil complex cylinder
do
i=0
while [ $i -le 199 ]; do
  python AL_PINN/metrics/run.py --gt_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_overfitted/$task\_normalized --ds_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_baseline_normalized/$task$i --feature=tracking --store_dir=AL_PINN/metrics/res/tracking/baseline_$task$i > AL_PINN/metrics/log/tracking_baseline_$task$i.log
  if [ $i -eq 0 ]; then
    i=$((i+9))
  else
    i=$((i+10))
  fi
done
done


 cylinder
for task in cylinder
do
i=0
while [ $i -le 199 ]; do
  python AL_PINN/metrics/run.py --gt_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_overfitted_v/$task\_normalized --ds_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_baseline_normalized_v/$task$i --feature=tracking --store_dir=AL_PINN/metrics/res/tracking/baseline_$task$i --use_v > AL_PINN/metrics/log/tracking_baseline_$task$i_v.log
  if [ $i -eq 0 ]; then
    i=$((i+9))
  else
    i=$((i+10))
  fi
done
done


# supplementary
 complex cylinder

#  !pinn
for task in complex complex cylinder
do
i=0
while [ $i -le 199 ]; do
  python AL_PINN/metrics/run.py --gt_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_overfitted/$task\_normalized --ds_dir=/csproject/t3_lzengaf/lzengaf/fyp/zxiaoal/dt4_rlpinn_normalized/$task$i --feature=tracking --store_dir=AL_PINN/metrics/res/tracking/tracking_rlpinn_normalized_$task$i > AL_PINN/metrics/log/tracking_rlpinn_normalized_$task$i.log
  if [ $i -eq 0 ]; then
    i=$((i+9))
  else
    i=$((i+10))
  fi
done
done



# use v
for task in airfoil complex cylinder
do
i=0
while [ $i -le 159 ]; do
  python AL_PINN/metrics/run.py --gt_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_overfitted_v/$task\_normalized --ds_dir=/csproject/t3_lzengaf/lzengaf/fyp/zxiaoal/dt4_rlpinn_normalized_v/$task$i --feature=tracking --store_dir=AL_PINN/metrics/res/tracking/tracking_rlpinn_normalized_$task$i --use_v   > AL_PINN/metrics/log/tracking_rlpinn_normalized_$task$i\_v.log
  if [ $i -eq 0 ]; then
    i=$((i+9))
  else
    i=$((i+10))
  fi
done
done



for task in airfoil complex cylinder
do
i=149
while [ $i -le 199 ]; do
  python AL_PINN/metrics/run.py --gt_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_overfitted_v/$task\_normalized --ds_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_rlpinn_normalized_v/$task$i --feature=tracking --store_dir=AL_PINN/metrics/res/tracking/tracking_rlpinn_normalized_$task$i --use_v   > AL_PINN/metrics/log/tracking_rlpinn_normalized_$task$i\_v.log
  if [ $i -eq 0 ]; then
    i=$((i+9))
  else
    i=$((i+10))
  fi
done
done


for task in airfoil complex cylinder
do
i=0
while [ $i -le 199 ]; do
  echo $i
  python AL_PINN/metrics/run.py --gt_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_overfitted_v/$task\_normalized --ds_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_rlpinn_20g_v/$task$i --feature=tracking --use_v --store_dir=AL_PINN/metrics/res/tracking/tracking_20g_$task$i > AL_PINN/metrics/log/tracking_rlpinn_20g_$task$i\_v.log
  if [ $i -eq 0 ]; then
    i=$((i+9))
  else
    i=$((i+10))
  fi
done
done


for task in airfoil complex cylinder
do
i=0
while [ $i -le 199 ]; do
  python AL_PINN/metrics/run.py --gt_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_overfitted/$task\_normalized --ds_dir=/csproject/t3_lzengaf/lzengaf/fyp/zxiaoal/dt4_rlpinn_20g/$task$i --feature=tracking --store_dir=AL_PINN/metrics/res/tracking/tracking_rlpinn_20g_$task$i > AL_PINN/metrics/log/tracking_rlpinn_20g_$task$i.log
  if [ $i -eq 0 ]; then
    i=$((i+9))
  else
    i=$((i+10))
  fi
done
done
