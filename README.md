# Project Name - Quantum Image Classification

## Code Method Report
### Overview
This repository contains code for a quantum kernel-based classification method using scikit-learn, TensorFlow, and Qiskit. The method aims to classify handwritten digits from the MNIST dataset using support vector machines (SVMs) with quantum kernels.

### Code Structure
The repository consists of the following main components:

1. **Imports:** The necessary libraries and modules are imported at the beginning of the code. These include scikit-learn, TensorFlow, Qiskit, and other relevant dependencies.
2. **Loading and Preprocessing Data:** The code loads the MNIST dataset using TensorFlow's mnist.load_data() function. It splits the data into training and testing sets and prints the shape of the loaded data for inspection. Additionally, it selects a subset of samples from the training and testing sets for faster processing.
3. **Preprocessing Steps:** Several preprocessing steps are applied to the data before training the classifier. These steps include standardization, dimensionality reduction using PCA, and normalization using min-max scaling. The preprocessing ensures that the data is in an appropriate format and range for subsequent analysis.
4. **Quantum Kernel-based Classification:** The core of the code is the quantum kernel-based classification process. It involves the following steps:
* Iterating over each label in the dataset: The code loops through a predefined set of labels and performs classification for each label against the rest.
* Binary Label Conversion: It converts the original labels into binary labels, where the specific label is assigned a value of 1, and the rest are assigned 0.
* Quantum Feature Map Initialization: The code initializes a quantum feature map object (ZZFeatureMap) from Qiskit. The feature map is defined with parameters such as the feature dimension and entanglement pattern.
* Quantum Kernel Creation: It creates a quantum kernel object (QuantumKernel) using the defined feature map. This quantum kernel is used to compute the similarity between feature vectors.
* SVC Model Initialization: An SVC model with a precomputed kernel and probability estimation enabled is initialized using scikit-learn's SVC class.
* Kernel Evaluation and Model Fitting: The quantum kernel is evaluated on the training samples, and the SVC model is fitted to the resulting kernel matrix and binary labels.
* Validation and Prediction: The quantum kernel is evaluated on the validation and test samples to compute accuracy scores and predict probabilities for the specific label.
* Results and Analysis: The original validation labels, modified binary labels, accuracy of discrimination, and predicted probabilities are printed and stored for further analysis.
5. **Visualization:** Lastly, the code includes a visualization step where it generates a quantum circuit for encoding a single sample using the ZZFeatureMap object. The circuit is then decomposed and visualized using Matplotlib.

## Comparision between Quantum and Classical SVM
### Performance Considerations:
1. Computational Complexity: The Quantum Kernel-based approach involves quantum circuit simulations, which can be computationally intensive and may scale exponentially with the number of qubits and samples. On the other hand, the traditional SVM classifier with PCA has polynomial time complexity and is generally more computationally efficient.
2. Hardware Requirements: The Quantum Kernel-based approach requires access to quantum computing resources or simulators, which may have limitations in terms of qubit availability, noise, and runtime. The traditional SVM classifier with PCA can be implemented on conventional computing hardware.
3. Scalability: As the dataset size increases, the Quantum Kernel-based approach may face challenges in terms of memory and computation. The traditional SVM classifier with PCA can handle larger datasets more efficiently.

### Accuracy Comparison:
1. **Label-specific Accuracy:** The Quantum Kernel-based approach provided accuracy scores for each label against the rest. This allows for a more detailed analysis of the performance for individual classes. In contrast, the traditional SVM classifier with PCA calculates the overall accuracy of the classifier on the test set without considering label-specific accuracies.
2. **Quantum Kernel's Discriminative Power:** The Quantum Kernel-based approach can potentially capture complex relationships and exploit quantum features for classification. It leverages the quantum feature map and quantum kernel to provide a more flexible representation and classification mechanism. However, its performance depends on the quality of the feature map and the suitability of quantum computing for the given task.
3. **PCA's Dimensionality Reduction:** The traditional SVM classifier with PCA performs dimensionality reduction using PCA, which can help capture the most relevant information from the data. By reducing the feature space, it focuses on the most discriminative components and simplifies the classification task. However, it may lose some fine-grained details that the Quantum Kernel-based approach can capture.
