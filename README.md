# Pytorch Metric Learning

These repositories contain code implementations that allow you to train a model using deep metric learning. It is possible to change and study the impact of hyperparameters and the loss function that is used in training the model. The project was created for master's thesis titled "Deep Metric Learning Techniques".

**Abstract**: This paper aims to analyze the effect of different loss functions and the number of training epochs on the performance of a neural network model using deep learning metrics in an image classification task. The study was conducted on the MNIST dataset, and the evaluation of the model's accuracy was performed using the k nearest neighbor algorithm. Several different loss functions were used in the study, including neighborhood component analysis loss, margin loss, proxy anchor loss, contrastive loss and triple loss. For each loss function, the model was trained at the number of epochs: $1$, $5$, $10$, $20$ and $50$. The goal was to understand how the loss functions and the number of training epochs affect the quality of the classification results. Analysis of the results shows that the choice of an appropriate loss function has a significant impact on the achieved accuracy of the model. It turned out that there is no single universally best loss function for all cases, but their effectiveness depends on the specifics of the problem. Moreover, the effect of the number of training epochs on the results varies - longer training does not always translate into better results. Cases of model overlearning were observed, leading to lower accuracy with too much training. The conclusions of the study confirm the effectiveness of deep learning metrics in image classification tasks. Selecting appropriate loss functions and adjusting the number of training epochs are key to achieving optimal accuracy. These results have important implications for the design and optimization of neural network models in classification applications.

## Requirements
- Python 3.10
- Conda

## Instalation
- Open conda terminal in repository directory
- Type following command:

```
conda env create -f environment.yml
```
- Open souce code in IDE
- Select python interpreter: **deep_metric_learning**
- Enjoy ;)

## Credits
Special thanks to Kevin Musgrave, who created the Pytorch Metric Learning library, without which this work would never have been written.