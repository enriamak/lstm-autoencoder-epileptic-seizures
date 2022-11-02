# Anomaly Detection of Epileptic Seizures with Autoencoders

<p align="center">
<img width="600" alt="image" src="https://user-images.githubusercontent.com/65355690/199611545-82bc8503-704e-4cba-ba07-adbce08f65b9.png">
</p>

## Description
In this work, it is proposed to carry out machine learning for the detection of anomalies in cases
of epilepsy through the analysis of EEG signals. A study of the impact/benefit of incorporating
temporality into LSTM neural network models, mainly Autoencoders, capable of classifying anomalies,
will be carried out.

Its prevention would improve the quality of life of patients by reducing the number and severity of
injuries caused by epilepsy, in order to take sufficient preventive measures.

Different models have been previously applied (Random Forest, SVM, KNN,...) that allow
classifications to be made with precision between 80% and 98%. The results of using Autoencoders
show an 85.6% accuracy in the classification of anomalies in EEG signals.

## Index
---
- [Summary](#summary)
- [Context](#context)
  - [What is epilepsy?](#what-is-epilepsy)
  - [Requirement](#requirement)
  - [Conceptual framework](#conceptual-framework)
  - [Current situation](#current-situation)
- [Main paradigms](#main-paradigms)
- [Methodology](#methodology)
  - [Data treatment](#data-treatment)
  - [Implementation](#implementation)
  - [Threshold Selection](#threshold-selection)
  - [Network optimization](#network-optimization)
  - [Evaluation](#evaluation)
- [Results and comparison](#results-and-comparison)
- [Conclusions](#conclusions)
- [Used tools](#used-tools)

## Summary
- Anomaly detection: EEG signal analysis
- Impact/benefit study: Inclusion of temporality in LSTM
- Use of Autoencoders: Signal classification
- Comparative: Evaluation against other models

## Context
### What is epilepsy?
---
Epilepsy is a central nervous system (neurological) disorder in which normal brain activity is disrupted, causing seizures or periods of unusual behavior and sensations, and sometimes loss of consciousness.

These uncontrolled seizures could cause brain damage, serious injury, or even death to his patients in cases of traffic accidents or risky circumstances.

Epileptic seizures usually last only a few seconds or a few minutes, after which they are over and the brain returns to normal function. How a seizure manifests depends on the part of the brain affected and the cause of the epilepsy.

### Requirement
---
- Most common neurological disorder -> Worldwide, 50 million people suffer from epilepsy
- Death risk -> Premature death in people with epilepsy up to three times higher than in the general population.
- Tractable -> 70% of people with epilepsy could live without seizures if properly diagnosed and treated

### Conceptual framework
---
- KNN (K-Nearest Neighbor classifier)
- SGD (Stochastic Gradient Descendant)
- Linear SVM (Support vector machine)
- Random Forest (Decision tree)
- CNN (Convolutional Neural Network)

### Current situation
---
|            Author            |         Model        |   Precision   |
|:----------------------------:|:--------------------:|:-------------:|
|        Shekokar (2021)       |          LSTM        |     98,50%    |
|       Almustafa  (2020)      |     Random Forest    |     97,08%    |
|          Nandy (2019)        |          SVM         |     97,05%    |
|        Chirasani (2022)      |          CNN         |     96,34%    |
|     Ashok Sharmila (2019)    |          KNN         |     95,23%    |
|        Almustafa (2020)      |          SGD         |     81,92%    |

## Main paradigms
---
<p align="center">
<img width="992" alt="image" src="https://user-images.githubusercontent.com/65355690/189505771-5273c98d-c4a3-441b-9633-02ada140a011.png">
<img width="992" alt="image" src="https://user-images.githubusercontent.com/65355690/189505781-450242ce-9f9b-468c-aba8-f30ea66e7873.png">
</p>

## Methodology
---
### Data treatment
- CHB-MIT Database


![image](https://user-images.githubusercontent.com/65355690/189506192-b42c9184-d9b8-41d2-ae13-e9a460346260.png)

- Anomalous Samples (2%)
- Layout [NSamples, LSeq, LSignal]
- Channel Dimension Removal


### Implementation
---
- Random parameters
- Combinations with 3600 samples
- Train 80% Test 20%
- Layout [3600, 5, 128] = [NSamples, LSeq, LSignal]

<img width="600" alt="image" src="https://user-images.githubusercontent.com/65355690/189506242-d982d24c-58d0-43d9-9f2c-09d9fe6a66c4.png">
<img width="600" alt="image" src="https://user-images.githubusercontent.com/65355690/189506238-b67eabb0-72cf-449e-b294-e102ca290882.png">

### Threshold Selection 
---
<img width="397" alt="image" src="https://user-images.githubusercontent.com/65355690/189506290-a4259823-4ec3-47e1-9bd5-985fa8cdcd9c.png">

### Network Optimization
---
We will attend to the following criteria:
- Temporary windows of different lengths
- Internal parameters of the network
- Increase the number of samples and epochs
- Loss function

### Evaluation
---
- KFOLD 5


![image](https://user-images.githubusercontent.com/65355690/189506565-31b74822-21a9-4527-9fe4-85f0a0956c5d.png)

- Quality metrics:
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix


## Results and comparison
---
### Tests V1
![image](https://user-images.githubusercontent.com/65355690/189506905-e1a37c68-6f4c-426f-8aa7-f3a0cb5b0b25.png)

- Combination Lists
  - LIST LEARNING RATE = [0.01, 0.001, 0.0001]
  - LIST ENCODING DIM = [2, 7, 64]
  - LIST SEQL = [2, 5, 10]
---
### Tests V2(optimized)
![image](https://user-images.githubusercontent.com/65355690/189506910-11bca2c2-f53b-43a4-9ed8-107eba4719f5.png)

- Parameters
  - LEARNING RATE = 0.001
  - ENCODING DIM = 7
  - SEQL = [2, 5]
  - 10,000 Samples and 150 Epochs


### Comparison
Quality Metrics:
|         Class       |     Precision    |     Recall    |     F1-score    |
|:-------------------:|:----------------:|:-------------:|:---------------:|
|      0 (normal)     |        0,86      |      0,87     |       0,86      |
|      1 (seizure)    |        0,87      |      0,86     |       0,86      |
|                     |                  |               |                 |
|       Accuracy      |       0,865      |      0,865    |       0,86      |
|       Macro avg     |       0,865      |      0,865    |       0,86      |
|     Weighted avg    |       0,865      |      0,865    |       0,86      |

---
Past Work:
|         Models       |     Precision    |
|:--------------------:|:----------------:|
|          LSTM        |       98,50%     |
|     Random Forest    |       97,08%     |
|          SVM         |       97,05%     |
|          CNN         |       96,34%     |
|          KNN         |       95,23%     |
|          SGD         |       81,92%     |
---
Boxplot:

![image](https://user-images.githubusercontent.com/65355690/189506974-cef2be9d-6a21-4ad8-8a89-72409cf0985f.png)

---
Confusion Matrix:

![image](https://user-images.githubusercontent.com/65355690/189506976-072f2be0-29f4-4dd6-8c84-7415f59b46ff.png)

## Conclusions
- Positive Results -> Close to 90% accuracy
- Autoencoder as classifier -> One-class training
- Deep learning knowledge -> Refinement within the framework of deep learning

## Used tools
- Language: Python
- Libraries: Pytorch, Numpy, Sklearn, Skimage
- Software: Spyder, Jupyter Notebook, Google Colab
- Hardware: Ryzen 5 2600x CPU, Azure GPU Virtual Machine, Google Colab TPU
