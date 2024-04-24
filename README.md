# Adversarial-learning-based closed-loop training strategy for PINN-based fluid simulators that generalize
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
Usage:
    Use mouse to drag the objects
    Press 'x' to increase flow velocity
    Press 'y' to decrease flow velocity
**Note**: The interface is only tested on Windows 11

## Evaluation tools

*evaluation/inference.py* : given an input dataset and a PINN solver weight, obtain the inference results
*evaluation/visualization.py* : given inference results in .npy, visualize the results and save as .gif and .png.
    
## Evaluation metrics
`run.py` : runs the evaluation for the overfitted results and the inference results. The inputs are the folder of those dataset. \
`metrics/scripts/basic.sh` : basic commands for the evaluation \
`metrics/scripts/ablation.sh` : commands for ablation study \
`metrics/scripts/training_tracking.sh` : commands for tracking the training performance