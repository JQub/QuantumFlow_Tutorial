
![](https://raw.githubusercontent.com/weiwenjiang/QML_tutorial/main/Readme_Img/qflow.png)

[![](https://img.shields.io/website?style=plastic&up_message=online&url=https%3A%2F%2Fwjiang.nd.edu%2Fcategories%2Fqf%2F)](https://jqub.github.io/categories/QF/) [![](https://img.shields.io/website?label=paper&style=plastic&up_color=blue&up_message=QuantumFlow&url=https%3A%2F%2Farxiv.org%2Fpdf%2F2006.14815.pdf)](https://arxiv.org/pdf/2006.14815.pdf) [![](https://img.shields.io/website?label=paper&style=plastic&up_color=blue&up_message=Tutorial&url=https%3A%2F%2Farxiv.org%2Fpdf%2F2012.10360.pdf)](https://arxiv.org/pdf/2012.10360.pdf) [![](https://img.shields.io/badge/Colab-online-orange?style=plastic)](#)


# Tutorial for Implemeting Neural Network on Quantum Computer

[News 2021/06] A tutorial proposal on QuantumFlow has been accepted by [**ESWEEK**](https://esweek.org/). See you then virtually on Oct. 08, 2021.

[News 2021/06] The tutorial on QuantumFlow optimization is released on 06/02/2021! See Tutorial 4.

[News 2021/05] The tutorial can be executed on Google CoLab now!

[News 2021/01] Invited to give a talk at **ASP-DAC 2021** for this tutorial work.

[News 2021/01] [QuantumFlow](https://arxiv.org/pdf/2006.14815.pdf) has been accepted by **Nature Communications**.

[News 2020/12] The tutorial is released on 12/31/2020! Happy New Year and enjoy the following Tutorial.

[News 2020/09] Invited to give a talk at **IBM Quantum Summit** for [QuantumFlow](https://arxiv.org/pdf/2006.14815.pdf).


## Overview
Recently, we proposed the first neural network and quantum circuit co-design framework, [QuantumFlow](https://arxiv.org/pdf/2006.14815.pdf), in which we have successfully demonstrated the quantum advatages in performing the basic neural operation from O(N) to O(logN). Based on the understandings from the co-design framework, in this repo, we provide a tutorial on an end-to-end implementation of neural networks onto quantum circuits, which is base of the invited paper at **ASP-DAC 2021**, titled [When Machine Learning Meets Quantum Computers: A Case Study](https://arxiv.org/pdf/2012.10360.pdf). 

This repo aims to demonstrate the workflow of implementing neural network onto quantum circuit and demonstrates the functional correctness. It will provide the basis to understand [QuantumFlow](https://arxiv.org/pdf/2006.14815.pdf). The demonstration of quantum advantage will be included in the repo for [QuantumFlow](https://arxiv.org/pdf/2006.14815.pdf), which will be completed soon at [here](https://github.com/weiwenjiang/QuantumFlow).

## Framework: from classical to quantum
![](https://raw.githubusercontent.com/weiwenjiang/QML_tutorial/main/Readme_Img/Frameworks.png)

In the above figure, on the left-hand, it is the design framework for classical hardware (HW) accelerators. The whole procedure will take three steps: (1) pre-processing data, (2) accelerating the neural computations, (3) post-processing data.

Similarly, on the right-hand side, we can build up the workflow for quantum machine learning. It takes 5 steps to complete the whole computation: (1) **PreP** pre-processing data; (2) **U<sub>P</sub>** data encoding onto quantum, that is quantum-state preparation; (3) **U<sub>N</sub>** neural operation oracle on quantum computer; (4) **M** data readout, that is quantum measurement; (5) **PostP** post-processing data. Among these stpes, (1) and (5) are conducted on classical computer, and (2-4) are conducted on the quantum circuit.

## Tutorial 1: **PreP** + **U<sub>P</sub>**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weiwenjiang/QML_tutorial/blob/main/Tutorial_1_DataPreparation.ipynb)

This tutorial demonstrates how to do data pre-preocessing and encoding it to quantum circuit using Qiskit. 

Let us formulate the problem as follow.

**Given:** (1) One 28\*28 image from MNIST ; (2) The size to be downsampled, i.e., 4\*4

**Do:** (1) Downsampling image; (2) Converting classical data to quantum data that can be encoded to quantum circuit; (3) Create quantum circuit and encode 16 pixel data to log16=4 qubits.

**Check:** Whether the data is correctly encoded.


## Tutorial 2: **PreP** + **U<sub>P</sub>** + **U<sub>N</sub>**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weiwenjiang/QML_tutorial/blob/main/Tutorial_2_Hidden_NeuralComp.ipynb)


This tutorial demonstrates how to use the encoded quantum circuit to perform **weighted sum** and **quadratic non-linear** operations, which are the basic operations in machine learning. 

Let us formulate the problem based on the output of Tutorial 1 as follow.

**Given:** (1) A circuit with encoded input data **x**; (2) the trained binary weights **w** for one neural computation, which will be associated to each data.

**Do:** (1) Place quantum gates on the qubits, such that it performs **(x\*w)^2/||x||**.

**Check:** Whether the output data of quantum circuit and the output computed using torch on classical computer are the same.



## Tutorial 3: **PreP** + **U<sub>P</sub>** + **U<sub>N</sub>** + **M** + **PostP** 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weiwenjiang/QML_tutorial/blob/main/Tutorial_3_Full_MNIST_Prediction.ipynb)

This is a complete tutorial to demonstrates an end-to-end implementation of a two-layer neural network for MNIST sub-dataset of {3,6}. The first layer (hidden layer) is implemented using the one presented in Tutorial 2, and the second layer (output layer) is implemented using the P-LYR and N-LYR proposed in [QuantumFlow](https://arxiv.org/pdf/2006.14815.pdf). The model is pre-trained and the weights for hidden layer, output layer, and normalization are obtained (details will be provided in [QuantumFlow github repo](https://github.com/weiwenjiang/QuantumFlow)). 

Let us formulate the problem from scratch as follow.

**Given:** (1) An image from MNIST; (2) The trained model.

**Do:** (1) Construct the quantum circuit; (2) Perform the simulation on Qiskit or execute the circuit on IBM Quantum Processor.

**Check:** Whether the prediction is correct.



## Tutorial 4: **PreP** + **U<sub>P</sub>** + **Optimized U<sub>N</sub>** + **M** + **PostP**  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weiwenjiang/QML_tutorial/blob/main/Tutorial_4_QAccelerate.ipynb)

This is a complete tutorial to demonstrates QuantumFlow can optimize the quantum circuit with the same function as the one created in Tutorial 3.
We continously use the settings in Tutorial 3. We do the optimization on the U-Layer of two hidden neurons using the algorithm proposed in [QuantumFlow github repo](https://github.com/weiwenjiang/QuantumFlow).  

Let us formulate the problem from scratch as follow.

**Given:** (1) An image from MNIST; (2) The trained model.

**Do:** (1) Construct the quantum circuit with optimized U-Layer; (2) Perform the simulation on Qiskit or execute the circuit on IBM Quantum Processor.

**Check:** Whether the prediction is correct; whether the results are almost the same with the circuit created in Tutorial 3; compare the reduction on circuit depth.


## Related work on This Tutorial

The work published at Nature Communications.

```
@article{jiang2021co,
  title={A co-design framework of neural networks and quantum circuits towards quantum advantage},
  author={Jiang, Weiwen and Xiong, Jinjun and Shi, Yiyu},
  journal={Nature communications},
  volume={12},
  number={1},
  pages={1--13},
  year={2021},
  publisher={Nature Publishing Group}
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

**Email: wjiang8@gmu.edu**

**Web: https://wjiang.nd.edu/ | https://jqub.github.io**

**Date: 06/02/2021**
