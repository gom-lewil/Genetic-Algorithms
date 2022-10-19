# Genetic-Algorithms
 The functions and algorithms were tasks of a course by 
 [Oliver Kramer](https://uol.de/en/computingscience/ci/team/oliver-kramer) 
 in Februar 2022 at the Carl von Ossietzky University of Oldenburg, Germany. 

## Project Description
### Strategies
The functions in the strategies.py script, describe different 
[Genetic Algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm) for Artificial Intelligence.
Included strategies are:
- (1 + λ)
- (μ + λ)
- (μ, λ)
- (1 + 1) Rechenberg
- (μ + λ) Rechenberg
- (1, λ) self adaptation
- (1, λ) de-randomized
- (1, λ) evolution path
- (μ + λ) CMA

### Crossover methods
As crossover method only the calculation of mean over all parents is provided. 
Other methods can be added in the crossover.py file. 

### Using the library
As a sample usage and to compare the results of the different strategies on varing optimization problems 
have a look at and run the main.py file