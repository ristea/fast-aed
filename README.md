# Fast-AED: Lightning Fast Video Anomaly Detection via Multi-Scale Adversarial Distillation

## Overview

Fast-AED is a high-performance frame-level anomaly detection model for videos. It offers exceptional speed while maintaining competitive accuracy, making it ideal for real-time applications. The model achieves this by distilling knowledge from multiple precise object-level teacher models and employing a unique combination of standard and adversarial distillation techniques.

For more details please read our full [paper](https://arxiv.org/pdf/2211.15597.pdf).

## Features

- **Knowledge Distillation:** Transfer of information from accurate object-level teacher models to a fast student model.
  
- **Adversarial Distillation:** Joint application of standard and adversarial distillation to enhance the fidelity of anomaly maps.

- **Adversarial Discrimination:** Introduction of adversarial discriminators for each teacher model to distinguish between target and generated anomaly maps.

- **Benchmark Performance:** Extensive experiments on Avenue, ShanghaiTech, and UCSD Ped2 datasets showcase the model's superiority, outperforming competitors by over 7 times in speed and 28 to 62 times faster than object-centric models.

- **Unprecedented Speed:** Operating at an outstanding 1480 frames per second (FPS), Fast-AED achieves the best trade-off between speed and accuracy.

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

## License
This project is licensed under the MIT License - see the LICENSE file for details.
