# This code is the source code implementation for the paper "AdaDP-CFL: Cluster Federated Learning with Adaptive Clipping Threshold Differential Privacy."



## Abstract
![输入图片说明](https://github.com/csmaxuebin/AdaDP-CFL/blob/main/pic/1.png)
Federated learning is a distributed machine learning approach that enables multiple clients to train models collaboratively. As its data remains stored locally on each client, this approach significantly enhances the protection of private information. However, federated learning still faces privacy leakage risks in environments with data heterogeneity. Differential privacy mechanisms are widely utilized in federated learning to ensure privacy for clients, and the magnitude of the clipping threshold directly impacts model utility. The current research does not adequately address the impact of model accuracy and training loss on clipping thresholds and is challenged by excessive hyperparameter adjustments. In response to these challenges, we propose an adaptive clipping-based differential privacy federated learning algorithm named AdaDP-CFL. It achieves model personalization and facilitates knowledge sharing among different groups through clustering and regularization techniques. Subsequently, the algorithm addresses the issue of adaptive clipping for various clients, formulated as a Markov decision process, by utilizing a deep deterministic policy gradient model based on gradient differences across client groups. Experimental results demonstrate that our algorithm outperforms current algorithms in accuracy, effectively balancing privacy protection and model utility


# Experimental Environment

```
- breaching==0.1.2
- calmsize==0.1.3
- h5py==3.8.0
— opacus==1.4.0
- Pillow==9.2.0
- scikit-learn==1.2.2
- sklearn==0.0.post1
- torch==2.0.0
- torchvision~=0.15.1+cu117
- ujson==5.7.0
- numpy==1.23.2
- scipy==1.8.1
- matplotlib==3.5.2
```

## Datasets

`CIFAR10, FMNIST, SVHN, and CIFAR100`


## Experimental Setup

### Model Configurations

1.  **CNN Architecture**:
    
    -   For the CIFAR10, FMNIST, and SVHN datasets, a Convolutional Neural Network (CNN) is used that comprises two convolutional layers and three fully connected (FC) layers.
    -   The convolutional layers have 6 and 16 output channels, respectively, with 5x5 kernels.
    -   The fully connected layers have outputs of 120, 84, and 10, respectively.
2.  **ResNet9 Architecture**:
    
    -   For the CIFAR100 dataset, the ResNet9 network is utilized.
    -   This network includes one initial convolution layer and three groups of layers, each consisting of a head layer and a residual block.

### Comparison Algorithms

-   **FedAvg**: The standard Federated Averaging algorithm.
-   **SCAFFOLD**: Utilizes control variates to correct client update biases.
-   **IFCA**: Chooses different models for different clusters to adapt to non-IID data distributions.
-   **CFL**: Clustered Federated Learning, clusters clients based on data characteristics.
-   **DP-FedAvg**: Integrates Differential Privacy with the FedAvg algorithm.
-   **DP-SCAFFOLD**: Integrates Differential Privacy with the SCAFFOLD algorithm.
-   **DP-AGR**: Employs a differentially private aggregation method.

### Hyperparameter Settings

-   **Global Rounds**: Set to 100 rounds.
-   **Local Iteration Rounds**: Each client performs 5 rounds of training locally.
-   **Batch Size**: Set at 20 samples per batch.
-   **Learning Rate**: Set to 0.01.
-   **Client Count**: A total of 100 clients participate in the computation.
-   **Sampling Rates**: Selected from {0.1, 0.5}, determining the proportion of clients chosen per round.
-   **Momentum Magnitude**: Set at 0.5.
-   **Regularization Coefficients**: Options include {1, 0.1, 0.001}, used to adjust the regularization strength during model training.
## Python Files
	
-   **Client_ClusterFL.py**:    
    The code defines a class `Client_ClusterFL` for a federated learning client with differential privacy features. It includes methods for training with gradient clipping and noise addition for privacy, adjusting learning parameters, and evaluating model performance. This setup allows clients to improve their local models in a privacy-preserving manner during federated learning rounds.
-   **cluster_fl.py**:    
    The code facilitates the evaluation and clustering of client models in a federated learning setting based on their predictive performance on shared data, aiming to improve model collaboration and privacy through effective grouping and performance analysis.
-   **fedavg.py**:    
Used to aggregate parameters.
-   **Distance.py**:    
The provided Python function, `Distance`, calculates the relationship or similarity between different clusters of clients in a federated learning environment based on their model parameters.
-   **prox.py**:
This code defines a function called L2 that calculates and returns the loss value in a particular optimization environment. This function is mainly used in federated learning or model aggregation scenarios, where the L2 norm difference of the model parameters and the weighted difference between related parameters are considered.
-   **utils.py**:    
   Used to load data from different datasets, set model weights, and initialize model parameters.

##  Experimental Results
	Figure 4: Test Accuracy and Loss of ResNet9 on CIFAR100
	Tables III & IV: Comparison of Algorithm Accuracy with 20% and 30% Label Skew
	Figure 5: Test Accuracy for Different Privacy Budgets
	Figure 6: Test Accuracy Using Adaptive Clipping Threshold
![输入图片说明](https://github.com/csmaxuebin/AdaDP-CFL/blob/main/pic/2.png)
![输入图片说明](https://github.com/csmaxuebin/AdaDP-CFL/blob/main/pic/3.png)
![输入图片说明](https://github.com/csmaxuebin/AdaDP-CFL/blob/main/pic/4.png)



## Update log

```
- {24.06.13} Uploaded overall framework code and readme file
```

