# Lightning Fast Video Anomaly Detection via Multi-Scale Adversarial Distillation 

## Computer Vision and Image Understanding (Official Repository)

## Overview

Fast-AED is a high-performance frame-level anomaly detection model for videos. It offers exceptional speed while maintaining competitive accuracy, making it ideal for real-time applications. The model achieves this by distilling knowledge from multiple precise object-level teacher models and employing a unique combination of standard and adversarial distillation techniques.

For more details please read our full [paper](https://arxiv.org/pdf/2211.15597.pdf).

## Features

- **Knowledge Distillation:** Transfer of information from accurate object-level teacher models to a fast student model.
  
- **Adversarial Distillation:** Joint application of standard and adversarial distillation to enhance the fidelity of anomaly maps.

- **Adversarial Discrimination:** Introduction of adversarial discriminators for each teacher model to distinguish between target and generated anomaly maps.

- **Benchmark Performance:** Extensive experiments on Avenue, ShanghaiTech, and UCSD Ped2 datasets showcase the model's superiority, outperforming competitors by over 7 times in speed and 28 to 62 times faster than object-centric models.

- **Unprecedented Speed:** Operating at an outstanding 1480 frames per second (FPS), Fast-AED achieves the best trade-off between speed and accuracy.

## License

The source code and models are released under the Creative Common Attribution-NonCommercial-ShareAlike 4.0 International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)) license.

## Citation 
Please cite our work if you use any material released in this repository.
```
@article{Croitoru-CVIU-2024,
  author    = {Croitoru, Florinel-Alin and Ristea, Nicolae-Catalin and Dascalescu, Dana and Ionescu, Radu Tudor and Khan, Fahad Shahbaz and Shah, Mubarak},
  title     = "{Lightning Fast Video Anomaly Detection via Multi-Scale Adversarial Distillation}",
  journal   = {Computer Vision and Image Understanding},
  year      = {2024},
  }
```

## Getting Started

### Prerequisites

- Python 3.6+
- PyTorch 1.2+
- CUDA Toolkit (for GPU support)

### Installation

Clone the repository:

```bash
git clone https://github.com/ristea/fast-aed.git
cd fast-aed
```
