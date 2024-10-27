# Integrating-Vision-Transformers-with-DenseNet-for-Marine-Species-Classification.

This project combines two branches: a Vision Transformer (ViT) and a Convolutional Neural Network (DenseNet) to leverage both global and local features from the input images. Training and evaluation processes include various configurations to handle multi-GPU distributed training with support for mixed-precision via Apex.

Features

Dual Branch Architecture: Uses Vision Transformer (ViT) and ResNet/CNN to extract complementary features.
Multi-Scale Attention: Implements multi-scale sliding window self-attention across image patches.
Distributed Training: Supports distributed data parallelism with mixed-precision training using NVIDIA's Apex.
Custom Schedulers and Optimizers: Utilizes warmup linear and cosine learning rate schedules for stable training.
Extensive Logging: Uses TensorBoard for monitoring accuracy, loss, and other training metrics.
Getting Started
Prerequisites
To run this project, install the required dependencies:

Python 3.8+
PyTorch (with CUDA support for GPU training)
NVIDIA Apex (for mixed precision)
Other libraries like torchvision, sklearn, numpy, and tqdm.

Dataset
Ensure the dataset is organized and accessible for training:

Modify the dataset paths in utils/origidata_utils.py or directly in get_loader() method if needed.
Training
Run the training script with default or customized arguments:

bash
Copy code
python main.py --model_type ViT-B_16 --img_size 224 --train_batch_size 8 --eval_batch_size 8 --num_steps 100000 --learning_rate 3e-2 --fp16
model_type: Model variant to be used (e.g., ViT-B_16).
img_size: Input image resolution.
train_batch_size and eval_batch_size: Batch size for training and validation.
num_steps: Total training steps.
learning_rate: Initial learning rate.
fp16: Enables mixed-precision training.


Evaluation
To evaluate the model, use:

bash
Copy code
python main.py --do_eval --eval_batch_size 8
This will compute accuracy and log results to TensorBoard.

Logging
Logs can be found in the logs folder. To monitor training progress:

bash
Copy code
tensorboard --logdir logs
Model Checkpoints
The best model checkpoints are saved automatically during training in the specified output_dir. To load a checkpoint:

python
Copy code
checkpoint = torch.load('path_to_checkpoint')
model.load_state_dict(checkpoint['model'])

Troubleshooting
Memory Issues: Try reducing batch size or enabling mixed-precision with --fp16.
Multi-GPU: Ensure the torch.distributed setup is correctly configured for multi-GPU training


DataSets 
Shark =https://www.kaggle.com/larusso94/shark-species
Aslo=http://ouc.ai/dataset/ASLO-Plankton.zip
Fish=https://github.com/PeiqinZhuang/WildFish
