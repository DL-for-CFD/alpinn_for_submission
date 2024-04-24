# metric
python AL_PINN/metrics/run.py --use_v --calc_l1 --suffix=baseline_retrained --task=cylinder > AL_PINN/metrics/log/baseline_retrained_cylinder_all_v.log

python AL_PINN/metrics/run.py --use_v --calc_l1 --suffix=baseline_retrained --task=complex > AL_PINN/metrics/log/baseline_retrained_complex_all_v.log

python AL_PINN/metrics/run.py --calc_l1 --suffix=baseline_retrained --task=complex > AL_PINN/metrics/log/baseline_retrained_complex_all.log

python AL_PINN/metrics/run.py --calc_l1 --suffix=baseline_retrained --task=cylinder > AL_PINN/metrics/log/baseline_retrained_cylinder_all.log


# cylinder normalized
python AL_PINN/metrics/run.py --calc_l1 --suffix=rlpinn_normalized --task=cylinder > AL_PINN/metrics/log/rlpinn_normalized_all.log

python AL_PINN/metrics/run.py --calc_l1 --use_v --suffix=rlpinn_normalized --task=cylinder_normalized > AL_PINN/metrics/log/rlpinn_normalized_all_v.log