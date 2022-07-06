# Uncertainty-Matching Graph Neural Networks to Defend Against Poisoning Attacks

## Introduction
This is a preliminary version paper, which was published in September 2020. It aims to defend the model from poisoning attacks using uncertainty-matching graph neural networks (UM-GNN). Our goal is to implement the UM-GNN defined in the paper.

## Summaries
### 1.1
Graph neural networks are known for their usage in challenging tasks with the graph structure. On the other hand, they are vulnerable to adversarial attacks as regular neural network models. In this paper, they have contributed a new approach and labeled it as uncertainty-matching graph neural networks (UM-GNN). This network is a combination of a GNN and FCN with specific regulations. The regulations are basically taking the output from both models after the training process while contributing a new loss, which consists of the GNN model uncertainty value additionally to enhance the model from poisoning attacks.

### 2.1
The method behind this approach is constructing a new loss function. The loss function is a combination of Cross-Entropy Loss of the GCN model output, aligning predictions over FCN and GCN, consisting of the uncertainty of the GCN model and KL divergence of the FCN model output. The intuition behind this new loss function is to avoid wrong learning by updating the gradients when given the poisoned dataset.

### 2.2
We have implemented a GCN model, FCN model, and dataset builder, where the datasets are the same as mentioned in the paper. The critical point is also implemented, which is the loss function. In the source code, it can be visible that uncertainty matching functions are applied by using the Monte Carlo method. The paper did not specifically explain the key points of the UM-GNN. With deep research and reading, we have reached the main points of the essence of it.

### 3.1
The problem setup of our implementation is mostly the same as the given approach in the paper. The implementation of the GCN and FCN models may differ from others due to the creativity among humans. But the datasets are the same: Cora, Citeseer, and Pubmed.

### 3.2

### 3.3

### 4.

### 5.

