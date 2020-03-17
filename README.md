# Neural Hamiltonian Flows

## Introduction

Implementation of [1], and especially Neural Hamiltonian Flows.

## Structure

* `hamiltonian.py`: an `Hamiltonian` class, which provides integration and energy evaluation. 
* `encoder.py`: an `GenericEncoder` class, which provides sampling.  
* `nhf.py`: an `NHF` class, which encodes all the flows features and the training loops. 
* `flow_example.py`: a `FlowExample` class, which provides easy handling of plotting for showing results. 
* `train.py`: some train functions which can be easily extended to any distributions to test.
* `requirements.txt`: the necessary list of packages.
* `utils.py`: various utilities functions.
* `README.md`: this file.

## Todo

* Training suffers from numerical instabilities.
* Study the influence of the complexity of the MLP.
* Add new priors.
* Provide sampling feature.
* Test with additional densities.   

## References

[1] : We introduce a class of generative models that reliably learn Hamiltonian dynamics from high-dimensional observations. The learnt Hamiltonian can be applied to sequence modeling or as a normalising flow.

Abstract: The Hamiltonian formalism plays a central role in classical and quantum physics. Hamiltonians are the main tool for modelling the continuous time evolution of systems with conserved quantities, and they come equipped with many useful properties, like time reversibility and smooth interpolation in time. These properties are important for many machine learning problems - from sequence prediction to reinforcement learning and density modelling - but are not typically provided out of the box by standard tools such as recurrent neural networks. In this paper, we introduce the Hamiltonian Generative Network (HGN), the first approach capable of consistently learning Hamiltonian dynamics from high-dimensional observations (such as images) without restrictive domain assumptions. Once trained, we can use HGN to sample new trajectories, perform rollouts both forward and backward in time, and even speed up or slow down the learned dynamics. We demonstrate how a simple modification of the network architecture turns HGN into a powerful normalising flow model, called Neural Hamiltonian Flow (NHF), that uses Hamiltonian dynamics to model expressive densities. Hence, we hope that our work serves as a first practical demonstration of the value that the Hamiltonian formalism can bring to machine learning. More results and video evaluations are available at: http://tiny.cc/hgn

Keywords: Hamiltonian dynamics, normalising flows, generative model, physics