# Learning to Generate Noise for Multi Attack robustness
This is the *Pytorch Implementation* for the paper Learning to Generate Noise for Multi Attack Robustness

## Prerequisites
```
$ pip install -r requirements.txt

```
## Run
Bash script for running MNG-AC and baselines.
1. __CIFAR-10__ experiment
```

# MNG-AC
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python train_MNG.py --fname MNG_cifar10 --dataset cifar10 --batchsize 256 --model WideResNet 

# Evaluation
## PGD attacks
$ CUDA_VISIBLE_DEVICES=0 python evaluate.py --norm linf --fname MNG_cifar10 --dataset cifar10 --model WideResNet 

## Foolbox attacks
$ CUDA_VISIBLE_DEVICES=0 python evaluate.py --norm linf --fname MNG_cifar10 --dataset cifar10 --attack_lib foolbox --model WideResNet 
```

2. __SVHN__ experiment

```

# MNG-AC
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python train_MNG.py --fname MNG_svhn --dataset svhn  --batchsize 256 --model WideResNet 

# Evaluation
## PGD attacks
$ CUDA_VISIBLE_DEVICES=0 python evaluate.py --norm linf --fname MNG_svhn --dataset svhn --model WideResNet 

## Foolbox attacks
$ CUDA_VISIBLE_DEVICES=0 python evaluate.py --norm linf --fname MNG_svhn --dataset svhn --attack_lib foolbox --attack_lib foolbox --model WideResNet 
```

2. __Tiny-Imagenet__ experiment

```

# MNG-AC
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python train_MNG.py --fname MNG_tinyimagenet --dataset tinyimagenet --batchsize 256 --model resnet50

# Evaluation
## PGD attacks
$ CUDA_VISIBLE_DEVICES=0 python evaluate.py --norm linf --fname MNG_tinyimagenet --dataset tinyimagenet --model resnet50

## Foolbox attacks
$ CUDA_VISIBLE_DEVICES=0 python evaluate.py --norm linf --fname MNG_tinyimagenet --dataset tinyimagenet --attack_lib foolbox --model resnet50
`
