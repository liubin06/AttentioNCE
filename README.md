# AttentioNCE: Contrastive Learning with Instance Attention
AttentioNCE integrates the attention mechanism into contrastive learning to guide the model's attention towards high-quality samples while disregarding noisy ones. Consequently, AttentioNCE constructs a variational lower bound for an ideal contrastive loss, offering a worst-case guarantee for maximum likelihood estimation under noisy conditions.

## AttentioNCE Framework:
<p align='center'>
<img src='https://github.com/liubin06/AttentioNCE/blob/main/pic/framework.png' width='350'/>
</p>

## Flags:
`--d_pos`: scaling factor for positive samples.

`--d_neg`: scaling factor for negative samples.


## Model Pretraining
For instance, run the following command to train an embedding on different datasets.
```
python main.py --dataset_name 'cifar10'  --d_pos 1 --d_neg 10
```
```
python main.py --dataset_name 'stl10'  --d_pos 2 --d_neg 0.5
```
```
python main.py --dataset_name 'cifar100'  --d_pos 1 --d_neg 10
```
```
python main.py --dataset_name 'tinyImageNet'  --d_pos 4 --d_neg 1
```


## Linear evaluation
The model is evaluated by training a linear classifier after fixing the learned embedding.

path flags:
`--model_path`: choose the model for evaluation.
```
python linear.py --dataset_name 'stl10' --model_path '../results/stl10/stl10_SimCLR_4model_256_400_2.0_0.5.pth'
```

## Pretrained Models on STL10 Dataset
| Method  | $d_\text{pos}$ | $d_\text{neg}$ | Arch | Epoch | Batch Size  | Accuracy(%) | Download | 
|---------|:--------------:|:--------------:|:----:|:-----:|:---:|:-----------:|:---:|
| SimCLR  |     -        |       -        | ResNet50 |  400  | 256  |    80.15    |  [model](https://drive.google.com/file/d/1qQE03ztnQCK4dtG-GPwCvF66nq_Mk_mo/view?usp=sharing)|
| AttentioNCE |       2        |      0.5       | ResNet50 |  200  | 256  |    87.12    |  [model](https://drive.google.com/file/d/1f3d8LYeX_8VLtqK1oai6SywmHAV0TwK9/view?usp=sharing)| 
| AttentioNCE        |       2        |      0.5       | ResNet50 |  400  | 256  |    89.45    |  [model](https://drive.google.com/file/d/1cQquMQA74GlQD6MQeP1l2zEhkJfdrKCA/view?usp=sharing)| 


## Pretrained Models on CIFAR10 Dataset
|Method | $d_\text{pos}$ | $d_\text{neg}$ | Arch | Epoch | Batch Size  | Accuracy(%) | Download | 
|--|:--------------:|:--------------:|:----:|:-----:|:---:|:-----------:|:---:|
| SimCLR |        -        |       -       | ResNet50 |  400  | 256  |    91.12    |  [model](https://drive.google.com/file/d/1AgKdRXnqBmhTPMAuzwsk1kE-X3OwVGpH/view?usp=drive_link)| 
| AttentioNCE |       1        |       10       | ResNet50 |  200  | 256  |    92.42    |  [model](https://drive.google.com/file/d/1Pq8bMZzqdN9-7c-HmyeKSxTlKhufaYf4/view?usp=sharing)| 
| AttentioNCE |       1        |       10       | ResNet50 |  400  | 256  |    93.08    |  [model](https://drive.google.com/file/d/1pKRs_QT4goC-l62tT48FxrDSRLxQ1Hfd/view?usp=sharing)|

## Pretrained Models on CIFAR100 Dataset
|Method  | $d_\text{pos}$ | $d_\text{neg}$ | Arch | Epoch | Batch Size  | Accuracy(%) | Download | 
|---|:--------------:|:--------------:|:----:|:-----:|:---:|:-----------:|:---:|
| SimCLR |      -        |       -        | ResNet50 |  400  | 256  |    66.55    |  [model]()| 
| AttentioNCE |       1        |       10       | ResNet50 |  200  | 256  |    69.78    |  [model](https://drive.google.com/file/d/1hnQAvAgsNa3rOY1cRnZgUsWZLh6KOmUM/view?usp=sharing)| 
| AttentioNCE|       1        |       10       | ResNet50 |  400  | 256  |    70.23    |  [model](https://drive.google.com/file/d/1zGqa28oNiogYjriWJjp-MqkzwDjTskIO/view?usp=sharing)|

## Pretrained Models on TinyImageNet
|Method  | $d_\text{pos}$ | $d_\text{neg}$ | Arch | Latent Dim | Batch Size  | Accuracy(%) | Download | 
|---|:--------------:|:--------------:|:----:|:----------:|:---:|:-----------:|:---:|
| SimCLR |       -        |       -        | ResNet50 |    400     | 256  |    53.40    |  [model]()| 
| BCL |       4        |       1        | ResNet50 |    200     | 256  |    56.58    |  [model](https://drive.google.com/file/d/1oQaHY5fW2_3trpBe4xMP4K_x3cRW8UjL/view?usp=sharing)| 
| BCL |       4        |       1        | ResNet50 |    400     | 256  |    58.61    |  [model](https://drive.google.com/file/d/1lVRxeZBRP18uQsw4FVOtaoBQnK7hU0yG/view?usp=sharing)| 
## Acknowledgements
Part of this code is credited to [SimCLR(ICML 2020)](https://github.com/leftthomas/SimCLR), [DCL(NeurIPS 2020)](https://github.com/chingyaoc/DCL) and [HCL(ICLR 2021)](https://github.com/joshr17/HCL).
