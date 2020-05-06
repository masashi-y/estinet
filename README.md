
# Reimplementation of [Estinet](https://arxiv.org/abs/1901.03995)

This repo tries to reproduce the results of the Image Addition experiment of ICLR2019 paper [Neural network gradient-based learning of black-box function interfaces](https://arxiv.org/abs/1901.03995) by Jacovi et al.

Their method enables the end-to-end training of a combination of deep neural networks and a black-box function, which is a standard programmatic function, such as look-up and arithmetic ops and cannot be differentiated through (i.e. black-box to neural network).

The key to their method is the "Estimate and Replace" method, in which a _black-box estimator_, mimicking the black-box function's interface, is firstly pretrained, and plugged into the system as a differentiable replacement of the function, enabling the end-to-end training.
At test time, the true black-box function is used.
Please refer to the paper for the detail.

## Image addition task

In this task, a system is given a list of MNIST images and asked to answer the sum of the represented numbers.
An ideal system would be an MNIST classifier plus just a call to `sum` function to the outputs, which is not end-to-end trainable.

Moveover, the difficulty of the task is that the MNIST dataset is not available in this setting, and a system must learn to solve the task by just pairs of a list of images and its sum instead.

The base system is an (identical) Conv classifier on each images, followed by a [NALU](https://arxiv.org/abs/1808.00508)-activation regression on the final output by an LSTM scanning the classification outputs.
The Estinet system uses the NALU regression as the black-box estimator, and replaces it with a call to `sum` at test time.

When trained on a dataset consisting of lists of images of length `k=10`, the Estinet system not only outperforms the NALU, but it generalizes to samples of longer sequences (`k=100`), showing that it successfully trains the image classifier module, though the labels are not explicitly given.

(from their paper)

|Model| k = 10 | k = 100 |
|:--:|:--:|:--:|:--:|
|NALU| 1.42 | 7.88 |
|Estinet| 0. 42 | 3.3 |

(reproduced result by me)

|Model| k = 10 | k = 100 |
|:--:|:--:|:--:|:--:|
|NALU| - | - |
|Estinet| - | - |

## Running the code

Please refer to `requirements.txt` for the versions of libraries used in the reproduction.

```sh
pip install torch torchvision hydra
```

```sh
python train_addition.py
```

Please check available command line options by `--help`.


## Citation

```
@inproceedings{jacovi2019blackbox,
  author = {A. Jacovi and G. Hadash and E. Kermany and B. Carmeli and O. Lavi and G. Kour and J. Berant},
  booktitle = {International Conference on Learning Representations (ICLR)},
  title = {Neural network gradient-based learning of black-box function interfaces},
  year = {2019},
}
```