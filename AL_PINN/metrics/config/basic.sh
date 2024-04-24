tmux
zsh
conda activate pinn

cd /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder
export CUDA_VISIBLE_DEVICES=


tmux attach -t 
tmux kill-session


# metric
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/run.py --use_v --calc_l1 --suffix=baseline_retrained --task=cylinder > /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/log/baseline_retrained_cylinder_all_v.log

python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/run.py --use_v --calc_l1 --suffix=baseline_retrained --task=complex > /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/log/baseline_retrained_complex_all_v.log

python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/run.py --calc_l1 --suffix=baseline_retrained --task=complex > /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/log/baseline_retrained_complex_all.log

python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/run.py --calc_l1 --suffix=baseline_retrained --task=cylinder > /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/log/baseline_retrained_cylinder_all.log


# cylinder normalized
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/run.py --calc_l1 --suffix=rlpinn_normalized --task=cylinder > /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/log/rlpinn_normalized_all.log

python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/run.py --calc_l1 --use_v --suffix=rlpinn_normalized --task=cylinder_normalized > /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/metrics/log/rlpinn_normalized_all_v.log