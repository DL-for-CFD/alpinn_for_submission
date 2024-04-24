# Adversarial or reinforcement learning-based closed-loop training strategy for PINN-based fluid simulators that generalize
LIU, Ran; XIAO, Ziruo; ZENG, Lingqi; ZHANG, Rushan (ordered alphabetically by last name)

## Install
```
cd AL_PINN
```

```
conda env create -f environment.yml
```

## Train
```
python train_ours.py
```

## Interactive interface
```
python interface.py --explicit_weights ./pretrain_weights/ours.pth
```
