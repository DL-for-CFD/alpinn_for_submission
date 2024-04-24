tmux
zsh
conda activate pinn

cd /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder
export CUDA_VISIBLE_DEVICES=


tmux attach -t 
tmux kill-session


test_0  test_1  test_2  test_3
airfoil  complex  cylinder
/project/t3_zxiaoal/Validation_Dataset/ablation_result
/project/t3_zxiaoal/Validation_Dataset/ablation_result_v
--use_v

for t in 0 1 2 3
do
for d in airfoil complex cylinder
do
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/run.py --gt_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_overfitted/$d\_normalized --ds_dir=/project/t3_zxiaoal/Validation_Dataset/ablation_result/test_$t/$d --feature=ablation --store_dir=/csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/res/ablation/test_$t\_$d > /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/log/ablation_test_$t\_$d.log
done
done

for t in 0 1 2 3
do
for d in airfoil complex cylinder
do
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/run.py --gt_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_overfitted_v/$d\_normalized --ds_dir=/project/t3_zxiaoal/Validation_Dataset/ablation_result_v/test_$t/$d --feature=ablation --store_dir=/csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/res/ablation/test_$t\_$d --use_v > /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/log/ablation_test_$t\_$d\_v.log
done
done


# supplementary experiments

for d in airfoil complex cylinder
do
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/run.py --gt_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_overfitted/$d\_normalized --ds_dir=/project/t3_zxiaoal/Validation_Dataset/rlpinn_10q_30ep/$d --feature=ablation_sup --store_dir=/csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/res/ablation/sup_test_$d > /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/log/ablation_test_sup_$d.log
done


for d in airfoil complex cylinder
do
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/run.py --gt_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_overfitted_v/$d\_normalized --ds_dir=/project/t3_zxiaoal/Validation_Dataset/rlpinn_10q_30ep_v/$d --feature=ablation --store_dir=/csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/res/ablation/sup_test_$d --use_v > /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/log/ablation_test_sup_$d\_v.log
done



# 100
/project/t3_zxiaoal/Validation_Dataset/ablation_result_100

for t in 0 1 2 3
do
for d in airfoil complex cylinder
do
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/run.py --gt_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_overfitted/$d\_normalized --ds_dir=/project/t3_zxiaoal/Validation_Dataset/ablation_result_100/test_$t/$d --feature=ablation_100 --store_dir=/csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/res/ablation_100/test_$t\_$d > /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/log/ablation_100_test_$t\_$d.log
done
done

for t in 0 1 2 3
do
for d in airfoil complex cylinder
do
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/run.py --gt_dir=/project/t3_zxiaoal/Validation_Dataset/dt4_overfitted_v/$d\_normalized --ds_dir=/project/t3_zxiaoal/Validation_Dataset/ablation_result_100_v/test_$t/$d --feature=ablation_100 --store_dir=/csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/res/ablation_100/test_$t\_$d --use_v > /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/log/ablation_100_test_$t\_$d\_v.log
done
done

