---
title: "CustumQuantumTF: Efficient TensorFlow-Based Quantum Circuit Implementation Using Custom Layers for GPU-Accelerated Machine Learning on Windows"
authors:
  - name: Andrea Carbone
    affiliation: 1
    orcid: 0000-0002-1119-060X
affiliations:
  - name: Department of Civil, Constructional and Environmental Engineering (DICEA), Sapienza University of Rome, Rome, Italy
    index: 1
tags:
  - Python
  - TensorFlow
  - quantum computing
  - quantum machine learning
  - hybrid neural networks
  - GPU computing
  - Windows
date: 2026-04-10
bibliography: paper.bib
---

# Summary

The integration of quantum computing with classical machine learning is emerging as a promising research direction, 
with the potential to significantly enhance model expressivity and computational efficiency [CITES]. However, despite strong 
theoretical foundations, practical implementations are still largely limited to simulation-based approaches due to the
current constraints of quantum hardware [CITE]. A similar situation occurred in the development of artificial neural networks, 
which were first conceptualized in the mid-20th century [CITE] but became practically viable only in recent decades with 
the advent of modern computational resources and large-scale datasets.

This repository introduces **QuLayer**, a TensorFlow-native framework for building and simulating quantum circuits 
directly within classical deep learning pipelines. The primary goal of this software is to simplify the development 
of Hybrid Quantum Neural Networks (HQML) by removing the need for external quantum simulation libraries. Unlike 
existing solutions, QuLayer is implemented entirely using standard TensorFlow operations. This design enables native
GPU acceleration and allows the framework to run efficiently on Windows systems without requiring Linux-based 
environments or specialized dependencies. As a result, the framework lowers the barrier to entry for researchers 
interested in quantum machine learning, including those with limited experience in quantum computing.

The framework provides a modular and extensible set of quantum operations, allowing users to construct flexible 
parameterized quantum circuits and integrate them seamlessly into existing machine learning workflows.


# Statement of Need

HQML has gained increasing attention due to its potential advantages in both computational efficiency and model expressivity [CITE]. 
Existing frameworks such as PennyLane [cite] and TensorFlow Quantum [cite] libraries offer powerful tools for simulating quantum 
circuits and integrating them with classical machine learning models. However, they often introduce several practical challenges, 
including complex dependencies, reliance on external simulation backends, and limited support for GPU acceleration in certain 
configurations. Additionally, many of these tools are primarily optimized for Linux-based environments, making them less accessible 
to users working on native Windows systems.

These limitations can significantly hinder experimentation, particularly for researchers who:
- do not have access to Linux-based infrastructures  
- require efficient GPU-based computation  
- come from non-quantum backgrounds and need simpler tools  

The proposed custum quantum layer (QuLayer.py) addresses these issues by providing a fully TensorFlow-native implementation of 
quantum circuits. By relying exclusively on TensorFlow operations, the framework eliminates external dependencies and enables 
seamless integration with existing deep learning pipelines. This approach makes HQML more accessible, efficient, and easier 
to adopt across a broader research community.


# State of the field

The current landscape of HQML is dominated by well-established frameworks that enable the simulation of quantum circuits 
and their integration with classical deep learning models.

Among these, PennyLane represents one of the most widely adopted solutions. It provides high-level abstractions for quantu
circuit design, including built-in templates and data encoding strategies. A key strength of PennyLane lies in its flexibility, 
as it supports major machine learning frameworks, including TensorFlow [CITE], PyTorch [CITE], and JAX/Haiku [CITE]. This 
interoperability is achieved through dedicated interface modules that translate native PennyLane quantum circuits into 
representations compatible with classical machine learning layers (e.g., qml.qnn.KerasLayer, qml.qnn.TorchLayer). Similarly, 
TensorFlow Quantum provides tight integration with TensorFlow, using Cirq [CITE] as its underlying quantum circuit simulator.

Despite their capabilities, these frameworks share a common design paradigm in which quantum circuits are treated as external 
components that must be interfaced with classical machine learning pipelines. This approach typically requires:
- external simulators responsible for executing quantum circuits
- intermediate wrappers to ensure compatibility with ML frameworks  
- cross-framework data conversion and execution overhead 

In practice, this design can introduce inefficiencies, particularly when targeting GPU acceleration or working in environments 
that are not natively supported (e.g., Windows-based systems).

In contrast, the proposed 'QuLayer' framework is fully implemented within TensorFlow, removing dependencies on external quantum simulation 
engines and enabling direct integration with TensorFlow’s GPU execution pipeline in Windows. As a result, the proposed framework offers 
a lightweight and efficient alternative that reduces system complexity and simplifies the deployment of quantum-inspired models. 


# Software design

## Custom Q-TF Layer

The proposed framework is designed to have the quantum layer as a fully customizable TensorFlow module that provides a set of fundamental
quantum operations as building blocks for constructing arbitrary quantum circuits. Interaction with classical states is supported through both
amplitude and angle embeddings for encoding the quantum state, along with Pauli measurements and probabilistic decoding, enabling flexible 
integration of classical and quantum information. In particular, the layer includes a set of fundamental quantum operations, including 
single-qubit rotation gates around the x-, y-, and z-axes (Rx, Ry, Rz), and controlled-Z (CZ) entangling gates. 

Although the current implementation does not encompass all possible quantum gates, the provided initial set is sufficient to express a wide
class of parameterized quantum circuits, and the modular structure allows additional operations to be incorporated with minimal effort. Indeed, 
by releasing the framework as open-source software, the objective is to encourage collective development and continuous expansion of the available
quantum functionalities over time. A specific quantum circuit configuration is adopted as a representative application, inspired by the 
circuit-centric classifier design [20], based on a parameterized vsariational quantum circuit composed of amplitude encoding followed by strongly 
entangling layers. Although the present study focuses on this particular configuration, the quantum layer is not restricted to it and can be 
readily adapted to alternative circuit architectures.

The design, illustrated in Figure, reports the scheme:
- Each qubit undergoes a sequence of three parameterized single-qubit rotations (`R_z(θ1)`, `R_y(θ2)`, `R_z(θ3)`)
- Qubits are entangled using controlled operations in a ring topology, with progressively skipping connections per layer
- Final quantum state is measured via the Pauli-Z operator on each qubit, producing a continuous vector used as input for classical layers

![Quantum Layer Schematic](../StrongLayers.png)

## HQNN implementation

The proposed HQNN takes as input $28 \times 28$ grayscale images from the MNIST and FashionMNIST datasets. The data is first processed 
by a sequence of convolutional blocks designed to progressively extract hierarchical feature representations. The first hidden stage consists 
of a two-dimensional convolutional layer with $n$ filters and a $2 \times 2$ kernel. The layer uses *same* padding to preserve spatial 
dimensions and applies a ReLU activation function to introduce non-linearity. This is followed by a max pooling operation with a 
$2 \times 2$ pooling window and a stride of $1 \times 1$, which reduces redundancy while retaining the most relevant spatial features.

This convolution–pooling block is repeated three times. In each subsequent repetition, the number of filters is increased progressively 
to $2n$ and $4n$, respectively. This hierarchical expansion allows the network to capture increasingly abstract and high-level r
epresentations of the input data. After the final convolutional stage, the output tensor has dimensions $(\text{batch}, m, m, 4n)$, 
where $m$ denotes the resulting spatial resolution. Since the quantum layer operates on a different data structure, the tensor is reshaped 
into $(\text{batch}, m \cdot m, 4n)$ before being passed to the quantum component. This transformation ensures that spatial information
is reorganized into a format compatible with the quantum encoding process, allowing each feature vector to be processed independently.

The quantum layer employs amplitude encoding to map classical data into quantum states. The number of required qubits $n_Q$ is determined by the dimensionality
of the input feature space $n$:
$$
n_Q = 2 + \log_2(n)
$$
Equivalently, the minimum number of qubits is chosen such that the quantum state space can fully encode the reshaped feature vector without information loss. 
This ensures compatibility between the classical convolutional pipeline and the quantum embedding layer.

# Research impact statement

---

# AI usage disclosure

Generative AI tools were used to assist in improving the clarity and structure of the manuscript and documentation. 
All technical content, experimental design, and implementation details were reviewed and validated by the author.


# Acknowledgements

This work builds upon TensorFlow open-source software ecosystems. 


# References

(References are provided in `paper.bib` using full journal and conference names as required by JOSS.)


---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------

The integration of quantum computing with classical machine learning has become an increasingly relevant research area. 
However, current implementations are primarily simulation-based, as quantum hardware is still limited compared to the 
maturity of the underlying mathematical formulations. A similar gap historically existed in classical neural networks, 
which were theoretically proposed decades ago but only became practically viable with modern computational resources.
A comparable situation is expected for quantum computing, where theoretical models suggest the potential for significant 
improvements in computational efficiency and model expressivity when integrated with neural network architectures.

In this context, several Python-based frameworks have been developed to support quantum circuit simulation and integration 
with classical machine learning workflows, enabling the study of hybrid quantum-classical machine learning (HQML) models.
However, these frameworks typically rely on complex software dependencies and are predominantly optimized for Linux-based 
environments, thereby limiting accessibility and efficient GPU utilization on native Windows platforms. 

To assess the effectiveness of the proposed framework, a comparative analysis is conducted against established implementations, 
including PennyLane combined with Flax, TensorFlow, and PyTorch. The results demonstrate improvements in terms of computational 
efficiency, scalability, and ease of integration within classical machine learning pipelines.  
This software provides a TensorFlow-based implementation of quantum circuits as custom 
layers for hybrid quantum machine learning (HQML). 

In contrast to existing solutions, the proposed quantum circuits is based on TensorFlow custom layers, relying exclusively 
on standard TensorFlow operations (QuLayer.py). The implementation supports GPU acceleration and is designed to run natively 
on Windows systems without requiring external quantum simulation packages. This makes the framework accessible to users 
without access to Linux-based quantum computing environments and to researchers from diverse fields with limited programming 
experience, aiming to lower the barrier to entry for quantum computing and promote its adoption across disciplines, similarly 
to the widespread diffusion of classical machine learning.

The software is demonstrated within hybrid quantum neural networks applied to image classification tasks on the MNIST and FashionMNIST datasets,
and its performance is compared against existing quantum machine learning frameworks.


--------------------------------------------
To better understand these limitations, we consider a representative comparison based on implementations built using 
PennyLane across different machine learning ecosystems. In this context, three common configurations are typically adopted:

- **Pennylane/TF**: quantum circuits defined in PennyLane and integrated into TensorFlow via `qnn.keraslayer`  
- **Pennylane/PyTorch**: quantum circuits interfaced with PyTorch through `qnn.pytorch`  
- **Pennylane/Haiku**: quantum circuits executed via JAX and integrated with DeepMind’s Haiku, enabling just-in-time (JIT) compilation through XLA for improved performance  

While these approaches provide flexibility and cross-platform compatibility, they also highlight a key limitation: 
the reliance on intermediary layers between quantum simulation and classical computation.

### Build vs. Contribute Justification

Given the maturity of existing frameworks, a natural question is whether extending them would be sufficient. 
However, the proposed Q-TF framework adopts a fundamentally different approach.

Instead of interfacing with external quantum simulators, Q-TF models quantum circuits directly as TensorFlow operations. 
This design choice eliminates the need for intermediate wrappers and enables full compatibility with TensorFlow’s execution 
engine, including automatic differentiation and GPU acceleration.

This results in several key distinctions:

| Aspect | Existing frameworks | Q-TF |
|--------|--------------------|------|
| Design paradigm | Quantum-first | ML-first |
| Quantum execution | External simulators | Native TF ops |
| Integration | Adapter layers | Direct |
| GPU utilization | Indirect / limited | Native |
| System compatibility | Mostly Linux-oriented | Windows-native |

Therefore, the contribution of Q-TF is not simply an incremental improvement over existing tools, but rather a shift in perspective. 
By embedding quantum circuit logic directly within a machine learning framework, it enables a more efficient, accessible, 
and tightly integrated approach to HQML research.
