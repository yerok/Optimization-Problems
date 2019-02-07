import random
import numpy as np
import math
from os import listdir
from os.path import isfile, join
import os, shutil
import collections, numpy
from maxSatSolver import evalGA
from maxSatSolver import testGenetic

def removeContentOfFolder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def generateParams(popSize):

    popSizeArray = np.arange(20,100,10)
    mutArray = np.arange(0,5100,100)
    rateArray = np.arange(0.5,1,0.1)
    maxGenArray = np.arange(10,210,10)
    # popSizeArray = [30]
    # mutArray = [100]
    # rateArray = [0.9]
    # maxGenArray = [100]

    pop = [{'popSize': random.choice(popSizeArray),'mut': random.choice(mutArray),'rate':random.choice(rateArray),'maxGen':random.choice(maxGenArray)} for j in range(popSize)]
    # pop = [random.choice(popSizeArray),random.choice(mutArray),random.choice(rateArray),random.choice(maxGenArray)]
    return pop

def getFitness(pop,ratio):
    # maxFitScore = 260
    # fitness = np.zeros(len(pop),dtype=np.int32)
    fitness = []

    for i in range(len(pop)):
        fitScore, timer = evalGA("./instancesMAXSAT/",pop[i]['popSize'],pop[i]['mut'],pop[i]['maxGen'],pop[i]['rate'],True,ratio)
        fitness.append([fitScore,timer])
    return fitness

def selectRoulette(pop, fit):
    size = len(pop)

    newFitness = []

    # we inverse fitness because here it's a minimization problem
    for i in range(len(fit)):
        newFitness.append(1/fit[i][1])
    sumFit = int(sum(newFitness))
    selected = random.randint(0, sumFit)
    tempSum = 0
    fit1 = 0
    for i in range(size):
        tempSum += newFitness[i]
        if tempSum >= selected:
            mate1 = pop.pop(i)
            fit1 = newFitness.pop(i)
            break
        
    tempSum = 0
    sumFit = int(sum(newFitness))
    selected = random.uniform(0, sumFit)
    for i in range(len(pop)):
        tempSum += newFitness[i]
        if tempSum >= selected:
            mate2 = pop[i]
            pop.append(mate1)
            newFitness.append(fit1)
            return mate1, mate2

def crossover(mate1, mate2):
    child = {'popSize': 0,'mut': 0,'rate':0,'maxGen':0}

    selected = random.randint(0, len(mate1)-2)
    selected2 = random.randint(1,2)
    if selected == 0:
        if selected2 == 1:
            child['popSize'] = mate1['popSize']
            child['mut'] = mate2['mut']
            child['rate'] = mate2['rate']
            child['maxGen'] = mate2['maxGen']
        else :
            child['popSize'] = mate2['popSize']
            child['mut'] = mate1['mut']
            child['rate'] = mate1['rate']
            child['maxGen'] = mate1['maxGen']
    if selected == 1:
        if selected2 == 1:
            child['popSize'] = mate1['popSize']
            child['mut'] = mate1['mut']
            child['rate'] = mate2['rate']
            child['maxGen'] = mate2['maxGen']
        else :
            child['popSize'] = mate2['popSize']
            child['mut'] = mate2['mut']
            child['rate'] = mate1['rate']
            child['maxGen'] = mate1['maxGen']
    if selected == 2:
        if selected2 == 1:
            child['popSize'] = mate1['popSize']
            child['mut'] = mate1['mut']
            child['rate'] = mate1['rate']
            child['maxGen'] = mate2['maxGen']
        else :
            child['popSize'] = mate2['popSize']
            child['mut'] = mate2['mut']
            child['rate'] = mate2['rate']
            child['maxGen'] = mate1['maxGen']
  
    return child

def mutateDict(gene,i):
    popSizeArray = np.arange(30,110,10)
    mutArray = np.arange(0,5100,100)
    rateArray = np.arange(0.5,1,0.1)
    maxGenArray = np.arange(10,210,10)

    if i == 0:
        gene['popSize'] = random.choice(popSizeArray)
    if i == 1:
        gene['mut'] = random.choice(mutArray)
    if i == 2:
        gene['rate'] = random.choice(rateArray)
    if i == 3:
        gene['maxGen'] = random.choice(maxGenArray)

    return gene

def mutate(gene, mutate):
    for i in range(len(gene)):
        selected = random.randint(1, mutate)
        if selected == 1:
            gene = mutateDict(gene,i)
    return gene

def newPopulation(pop,fit,mut,elitism,maxScore):
    popSize = len(pop)
    newPop = []

    if elitism:
        newPop.append(selectBest(pop, fit,maxScore))
    
    while(len(newPop) < popSize):
        mate1, mate2 = selectRoulette(pop, fit)
        newPop.append(mutate(crossover(mate1, mate2), mut))

    return newPop

#Elitism
def selectBest(pop,fit,maxScore):
    best = 0
    bestFit = math.inf
    for i in range(len(fit)):
        if (fit[i][0] >= maxScore and  fit[i][1] < bestFit):
          best = i
          bestFit = fit[best][1]
    return pop[best]

def selectFinalBest(pop,fit,maxScore):
    best = 0
    bestFit = math.inf
    for i in range(len(fit)):
        if (fit[i][0] >= maxScore and fit[i][1] < bestFit):
            best = i
            bestFit = fit[best][1]
    return pop[best], bestFit

def geneticAlgorithm(popSize,mut,maxGen,rate,elitism):
    

    ratio = 3.5
    # 30,100,100,0.9,True
    # popSize': 40, 'mut': 4500, 'rate': 0.69999999999999996, 'maxGen': 10}
    maxScore,timer = evalGA("./instancesMAXSAT/",40,4500,10,0.69,True,ratio)
    print(maxScore)
    print(timer)

    generation = 0
    pop = generateParams(popSize)
    fitness = getFitness(pop,ratio)

    while(generation < maxGen):
        print("eee")
        generation += 1
        pop = newPopulation(pop,fitness,mut,elitism,maxScore)
        fitness = getFitness(pop,ratio)

    print(fitness)
    bestpop, bestFit = selectFinalBest(pop,fitness,maxScore)
    print("best pop : ",bestpop)
    print("best time : ",bestFit)


    with open("./resultsMAXSATMETAGA/"+"3-SAT_result.txt","a") as outputFile:
        outputFile.write(str(ratio)+" ")
        for key in bestpop:
           outputFile.write(str(bestpop[key])+" ")
        outputFile.write(str(bestFit) + "\n")
    outputFile.close()


def testMetaGA():
    
    # defaultPopSize = 50
    defaultPopSize = 20
    defaultMut = 200
    defaultRate = 0.9 
    defaultMaxGen = 20
    defaultElitism = True


    geneticAlgorithm(defaultPopSize,defaultMut,defaultRate,defaultMaxGen,defaultElitism)


import json

def main():
    testMetaGA()

if __name__ == "__main__":
    main()