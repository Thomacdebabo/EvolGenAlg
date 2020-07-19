# Evolutionary Genetic Algorithm for Path Optimization in Python

This is an optimizer which tries to optimize a path on a virtual terrain in regards to energy consumption.
It works by creating simple creatures which contain a brain which is modeled as a very simple neural network.
The training process works by generating a population of creatures which we get by randomly modifying and combining the parameters inside the neural networks of the previous generation.

![Main Image](https://github.com/Thomacdebabo/EvolGenAlg/blob/master/preeeetty_gooood.png)

## World

### The world contains:
- terrain (simple perlin noise as a height map)
- creatures

### For each iteration the world…
 - creates new creatures by breeding, mutating and killing the previous population
 - calculates the trajectory each creature makes over 1000+ steps

### There are a couple of rules:
- a creature dies when it reaches the border of the terrain
- each step has to be a certain size
- for each step a creature loses / gains some energy (uphill loses, downhill gains)
- the final score is a weighted sum of the energy and 1/distance_to_goal

## Creature

### A creature contains:
- id
- starting location
- target location
- current location
- trajectory
- energy score
- brain

-> while walking the brain decides which step to take according to given inputs

### Brain Inputs:
current energy
current height
height difference to last step
x,y distances to target
the heights of 4 points around the creature
how many steps it already took

## Brain

![Brain](https://github.com/Thomacdebabo/EvolGenAlg/blob/master/nn_schema.jpg)

### Implementation:
- input tensor
- n x m matrix for the connections between two layers
- where n,m are the amount of neurons which are connected
- non linear functions after each layer (relu)
- output tensor [x,y] 

### to calculate the next step: 
1. matrix multiplication with input
2. use non linear function to truncate the result
3. do these steps for each hidden layer
4. get output

