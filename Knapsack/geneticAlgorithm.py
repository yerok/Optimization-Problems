
from ks import *

def generate(length, popSize):
    pop = [[random.randint(0,1) for i in range(length)] for j in range(popSize)]
    return pop

def getFitness(pop, items, capacity):
    fitness = []
    for i in range(len(pop)):
        value = 0
        weight = capacity+1
        while (weight > capacity):
          value = 0
          weight = 0
          selectedItems = []
          for j in range(len(pop[i])):
            if pop[i][j] == 1:
              weight += calculateWeight(items[j])
              value += calculateValue(items[j])
              selectedItems += items[j]
          if weight > capacity:
            pop[i][selectedItems[random.randint(0, len(selectedItems)-1)]] = 0
        fitness += [value]

    return fitness