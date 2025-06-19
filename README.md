# EEPNet-V2
EEPNet-V2 is an enhanced version of the EEPNet architecture, built with PyTorch Lightning for improved modularity, scalability, and performance. This project focuses on cross-modal data registration, leveraging state-of-the-art neural networks to align multi-modal data, such as images and LiDAR point clouds.

![Video Demo](./videos/output_video.gif)

**⚠️ Important Notice:**
Due to the fact that the related paper for this project has not yet been published, we do not provide pre-trained models at this time. Additionally, to protect the core innovations of this project from being replicated, some critical parts of the code have been removed. The complete code will be released once the paper is accepted. Thank you for your understanding and support!

## Table of Contents
- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Configuration](#configuration)

## Environment Setup

To run the code, you need to have the following packages installed. You can create a virtual environment and install the required packages using `pip`. Here's a list of packages you need to install:

```bash
pip install pandas tqdm tensorboardX numpy torch pytorch_lightning scikit-learn joblib
```
If need GPU for training and testing, install the appropriate PyTorch version for your GPU drivers:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Training

To train the model, modify the hyperparameters in the `CFG` class within the `train.py` file as needed. Once you have set the desired hyperparameters, you can run the training script:

```bash
python train.py
```

### Testing

Similarly, for testing the trained model, make sure to adjust any parameters in the `CFG` class within the `test.py` file. After making the necessary changes, you can run the testing script:

```bash
python test.py
```

## Configuration

The `CFG` class within both `train.py` and `test.py` contains various hyperparameters that you can modify to suit your needs. Make sure to review and adjust these parameters before executing the scripts.

Feel free to explore the code and contribute to the project. If you have any questions, please open an issue or reach out!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.


