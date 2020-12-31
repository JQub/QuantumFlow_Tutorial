
![](https://raw.githubusercontent.com/weiwenjiang/QML_tutorial/main/Readme_Img/qflow.png)


# Tutorial for Implemeting Neural Network on Quantum Computer [[Tutorial arXiv](https://arxiv.org/pdf/2012.10360.pdf)] [[QuantumFlow arXiv](https://arxiv.org/pdf/2006.14815.pdf)]

[News] [QuantumFlow](https://arxiv.org/pdf/2006.14815.pdf) has been accepted by **Nature Communications**.

[News] Invited to give a talk at ASP-DAC 2021 for this tutorial work.

[News] The tutorial is released on 12/31/2020! Happy New Year and enjoy the following Tutorial.

## Overview
Recently,  we proposed the first neural network and quantum circuit co-design framework, [QuantumFlow](https://arxiv.org/pdf/2006.14815.pdf). Based on the understandings on how to implement neural networks onto quantum computer, we provide a tutorial on implementing neural networks onto quantum circuits, which is based the invited paper at **ASP-DAC 2021**, titled [When Machine Learning Meets Quantum Computers: A Case Study](https://arxiv.org/pdf/2012.10360.pdf). This github repo is for this work, and it will provide the basis to understand  [QuantumFlow](https://arxiv.org/pdf/2006.14815.pdf), the repo for [QuantumFlow](https://arxiv.org/pdf/2006.14815.pdf) will be completed soon at [here](https://github.com/weiwenjiang/QuantumFlow).

## Framework: from classical to quantum
![](https://raw.githubusercontent.com/weiwenjiang/QML_tutorial/main/Readme_Img/Frameworks.png)

In the above figure, on the left-hand, it is the design framework for classical hardware (HW) accelerators. The whole procedure will take three steps: (1) pre-processing data, (2) accelerating the neural computations, (3) post-processing data.

Similarly, on the right-hand side, we can build up the workflow for quantum machine learning. It takes 5 steps to complete the whole computation: (1) **PreP** pre-processing data; (2) **U<sub>P</sub>** data encoding onto quantum, that is quantum-state preparation; (3) **U<sub>N</sub>** neural operation oracle on quantum computer; (4) **M** data readout, that is quantum measurement; (5) **PostP** post-processing data. Among these stpes, (1) and (5) are conducted on classical computer, and (2-4) are conducted on the quantum circuit.

## Tutorial 1: **PreP** + **U<sub>P</sub>**
This tutorial demonstrates how to do data pre-preocessing and encoding it to quantum circuit using Qiskit. 

Let us formulate the problem as follow.

**Given:** (1) One 28\*28 image from MNIST ; (2) The size to be downsampled, i.e., 4\*4

**Do:** (1) Downsampling image; (2) Converting classical data to quantum data that can be encoded to quantum circuit; (3) Create quantum circuit and encode 16 pixel data to log16=4 qubits.

**Check:** Whether the data is correctly encoded.

Details please see **[Tutorial_1_DataPreparation.ipynb](https://github.com/weiwenjiang/QML_tutorial/blob/main/Tutorial_1_DataPreparation.ipynb)**.

## Tutorial 2: **PreP** + **U<sub>P</sub>** + **U<sub>N</sub>**
This tutorial demonstrates how to use the encoded quantum circuit to perform **weighted sum** and **quadratic non-linear** operations, which are the basic operations in machine learning. 

Let us formulate the problem based on the output of Tutorial 1 as follow.

**Given:** (1) A circuit with encoded input data **x**; (2) the trained binary weights **w** for one neural computation, which will be associated to each data.

**Do:** (1) Place quantum gates on the qubits, such that it performs **(x\*w)^2/||x||**.

**Check:** Whether the output data of quantum circuit and the output computed using torch on classical computer are the same.

Details please see **[Tutorial_2_Hidden_NeuralComp.ipynb](https://github.com/weiwenjiang/QML_tutorial/blob/main/Tutorial_2_Hidden_NeuralComp.ipynb)**.


## Tutorial 3: **PreP** + **U<sub>P</sub>** + **U<sub>N</sub>** + **M** + **PostP** 
This is a complete tutorial to demonstrates an end-to-end implementation of a two-layer neural network for MNIST sub-dataset of {3,6}. The first layer (hidden layer) is implemented using the one presented in Tutorial 2, and the second layer (output layer) is implemented using the P-LYR and N-LYR proposed in [QuantumFlow](https://arxiv.org/pdf/2006.14815.pdf). The model is pre-trained and the weights for hidden layer, output layer, and normalization are obtained (details will be provided in [QuantumFlow github repo](https://github.com/weiwenjiang/QuantumFlow)). 

Let us formulate the problem from scratch as follow.

**Given:** (1) An image from MNIST; (2) The trained model.

**Do:** (1) Construct the quantum circuit; (2) Perform the simulation on Qiskit or execute the circuit on IBM Quantum Processor.

**Check:** Whether the prediction is correct.

Details please see **[Tutorial_3_Full_MNIST_Prediction.ipynb](https://github.com/weiwenjiang/QML_tutorial/blob/main/Tutorial_3_Full_MNIST_Prediction.ipynb)**.



## Enviroment Requirement
* Qiskit
* numpy
* torch
* torchvision


## Related work on This Tutorial

The work accepted by Nature Communications.

```
@article{jiang2020co,
  title={A Co-Design Framework of Neural Networks and Quantum Circuits Towards Quantum Advantage},
  author={Jiang, Weiwen and Xiong, Jinjun and Shi, Yiyu},
  journal={arXiv preprint arXiv:2006.14815},
  year={2020}
}
```

The work invited by ASP-DAC 2021.

```
@article{jiang2020machine,
  title={When Machine Learning Meets Quantum Computers: A Case Study},
  author={Jiang, Weiwen and Xiong, Jinjun and Shi, Yiyu},
  journal={arXiv preprint arXiv:2012.10360},
  year={2020}
}
```

## Contact
**Weiwen Jiang**

**Email: wjiang2@nd.edu**

**Web: https://wjiang.nd.edu/**

**Date: 12/31/2020**
