
# Knapsack Problem 0/1

import numpy as np
import time
import math
import re
import heapq
import random
from collections import deque
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import subprocess
import os, shutil

def removeLSBs(number,bits):

    for i in range(0,bits):
        number = (number & ~(1 << i)) | (0 << i)

    return number

def calculateWeight(item):
    return item[0]

def calculateTotalWeigh(items):
    totalWeight = 0
    for item in items:
        totalWeight += calculateWeight(item)
    return totalWeight

def calculateValue(item):
    return item[1]

def calculateTotalValue(items):
    totalValue = 0
    for item in items:
        totalValue += calculateValue(item)
    return totalValue

# used to truncate floats
def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

# To add the indexes of the items
def addIndex(items):
    res = [[]]
    i = 1
    for item in items:
        r = np.append(item,i)
        res.append(r)
        i += 1
    return res

#compute the relative error
def getRelativeError(optCost,heuristicCost):
    relativeError = (optCost-heuristicCost)/(optCost)
    return relativeError

def getFPTASRelativeError(nbItems,bits,optCost):
    return nbItems*pow(2,bits)/optCost

# compute the weight / cost ratio
def getRatio(item):
    if item != []:
        ratio = item[1] / item[0]
    else:
        ratio = 0;
    return ratio

def calculateMaxCost(items):
    maxCost = 0

    for item in items:
        if calculateValue(item) > maxCost:
            maxCost = calculateValue(item)
    return maxCost

# add the Cost / Weight Ratio for every item in a set
def addRatio(items):
    res = [[]]
    i = 1
    for item in items:
        # ratio = getratio(item)
        ratio = getRatio(item)
        r = np.append(item,ratio)
        res.append(r)
        i += 1
    return res

# create all the possible subsets (2^n subsets)
def createSets(items):
    res = [[]]
    for item in items:
        newset = [r+[item] for r in res]
        res.extend(newset)
    return res


# brute force algorithm
def knapsackBrutForce(items, maxWeight):

    sets = createSets(items);

    knapsack = []
    bestWeight = 0
    bestValue = 0
    for subset in sets:
        weight = 0
        value = 0
        for item in subset:
            if item != []:
                weight += item[0] 
                value += item[1]
        if value > bestValue and weight <= maxWeight:
            bestValue = value
            bestWeight = weight
            knapsack = subset
    return knapsack, bestValue,bestWeight

# greedy algorithm with cost/weeight ratio heuristic
def knapsackGreedy(items,maxWeight,sortingKey):

    knapsack = []
    bestWeight = 0
    bestValue = 0
    weight = 0
    # we must add the v/c value 
    items = addRatio(items)
    # then sort in by this ratio
    sortedItems = sorted(items,key=sortingKey)
    while (len(sortedItems) > 0):
        item = sortedItems.pop()
        if item != []:
            if (bestWeight + item[0] <= maxWeight):
                knapsack.append(item)
                if item != []:
                    bestWeight += item[0] 
                    bestValue += item[1]
            else:
                break
    return knapsack, bestValue,bestWeight

# Just to print the number (IDs) of the items you should take
def knapsackToIDOfItem(knapsack):
    res = []
    for item in knapsack:
        if item != []:        
            res.append(int(item[2]))
    print("You should take the following items : ",res)


def plotBruteForceGraph():
    plt.plot([4,10,15,20],[0.000520356, 0.05570669220000001,2.9799490838,118.71765811666667])
    plt.xlabel('Number of items')
    plt.ylabel('computing time (s)')
    plt.title('Evolution of brute force algorithm computing time for the 0/1 knapsack Problem')
    plt.show()

def plotBruteForceAndBranchAndBound():
    #bruteForce
    plt.plot([4,10,15,20],[0.000520356, 0.05570669220000001,2.9799490838,118.71765811666667],label='Brute Force Algorithm')

    #B&B
    plt.plot([4,10,15,20,22],[0.0005002,0.00300192, 0.11206269,0.08406114,0.57941222], label='Branch and Bound Algorithm')
    plt.xlabel('Number of items')
    plt.ylabel('computing time (s)')
    plt.legend()
    plt.title('Evolution of computing time for the 0/1 knapsack Problem')
    plt.show()

def plotData(algorithm,numberOfLines,column1,column2,indexStart,numberOfItems,xlabel,ylabel,title):
    resultFiles = [f for f in listdir('./results/') if isfile(join('./results/', f))]
    resultFilePath = './results/'  + algorithm + ".txt"
    # number of items in the knapsack
    solArray = []

    # array reshaping
    myarray = np.fromfile(resultFilePath,dtype=float,sep="\r")

    # number of lines, number of arrays (parameters)
    myarray = np.reshape(myarray, (numberOfLines,9))

    x = []
    y = []
    for i in range(indexStart,indexStart+numberOfItems):
        x.append(truncate(myarray[i][column1],4))
        y.append(truncate(myarray[i][column2],4))

    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.show()

def plotGreedyGraph(results):

    x = []
    y = []

    for i in range(0,len(results)):
        if results[i] != []:
            x.append(results[i][0])
            y.append(results[i][1])
    plt.plot(x,y)
    plt.xlabel('Number of items')
    plt.ylabel('computation time (s)')
    plt.title('Evolution of greedy algorithm computation time for the 0/1 knapsack Problem')
    plt.show()

#used to calculate the upper bound of a node in the B&B algorithm
def calculateUpperBound(takenItems,items,index,capacity):

    upperBound = 0
    totalValue = calculateTotalValue(takenItems)
    totalWeight = calculateTotalWeigh(takenItems)
    bestRatio = getRatio(items[index])  
    upperBound = totalValue + (capacity - totalWeight)*bestRatio

    return upperBound

#used  to treat the test files and solutions files in order to use them in the differents algorithms
def prepareData(i,sol):

    if sol:
        testFiles = [f for f in listdir('./inst/') if isfile(join('./inst/', f))]
        testFilePath = './inst/'  + testFiles[i]
        solFiles = [f for f in listdir('./sol/') if isfile(join('./sol/', f))]
        solFilePath = './sol/'  + solFiles[i]
        solArray = np.fromfile(solFilePath,dtype=int,sep="\r")
         # regex to find the number of elements in one bag and use it to create arrays
        n = re.search("\d{1,2}",testFilePath)
        n = int(n.group(0))
        solArray = np.reshape(solArray, (50,n+3))
    else:
        testFiles = [f for f in listdir('./randomInstances/') if isfile(join('./randomInstances/', f))]
        testFilePath = './randomInstances/'  + testFiles[i]
         # regex to find the number of elements in one bag and use it to create arrays
        n = re.search("\d{1,2}",testFilePath)
        n = int(n.group(0))
        solArray = []

    # array reshaping
    myarray = np.fromfile(testFilePath,dtype=int,sep="\r")
    myarray = np.reshape(myarray, (50,(n*2)+3))

    myArraySize = (myarray.shape[1]-3)//2
    # print(myarray)
    return myarray, solArray, myArraySize, n 

#Priority queue used in the B&B Algorithm
class SimpleQueue(object):
    def __init__(self):
        self.buffer = deque()

    def push(self, value):
        self.buffer.appendleft(value)

    def pop(self):
        return self.buffer.pop()

    def __len__(self):
        return len(self.buffer)

#Node object used in the B&B algorithm
class Node(object):
    def __init__(self, level, selectedItems , cost, weight, bound):
        self.level = level
        self.selectedItems = selectedItems
        self.cost = cost
        self.weight = weight
        self.bound = bound

#used to find items tuples from their ID
def indextoItem(indexes,items):
    print(indexes)
    finalItems = []
    for index in indexes:
        finalItems.append(items[index])

    return finalItems

def BranchAndBound(items,capacity):
   
    priorityQueue = SimpleQueue()
   
    bestSoFar = Node(0, [],0.0, 0.0, 0.0)
    firstNode = Node(0,[],0.0, 0.0, calculateUpperBound([],items,0,capacity))
    priorityQueue.push(firstNode)

    while len(priorityQueue) > 0 :

        currentNode = priorityQueue.pop()
        if currentNode.level >= len(items):
            break
        #if the bound is higher than the best cost that we've found so far, we create a new node 
        # else we do not evaluate this node (pruning)
        if currentNode.bound >= bestSoFar.cost:
            item = items[currentNode.level]
            nextAdded = Node(
                currentNode.level + 1,
                currentNode.selectedItems + [currentNode.level],
                currentNode.cost + calculateValue(item),
                currentNode.weight + calculateWeight(item),
                currentNode.bound
            )
            # if we can still store the item in the knapsack and if the total value of it is better than
            # the previous better one, we add it in the knapsack
            if nextAdded.weight <= capacity:
                if nextAdded.cost > bestSoFar.cost:
                    bestSoFar = nextAdded

                # if the bound of this node is higher than the best cost so far, we push this 
                # node in the queue
                if nextAdded.bound > bestSoFar.cost:
                    priorityQueue.push(nextAdded)

            #We create another node at the same level of the previous one which corresponds to the node where we don't take this item
            nextNotAdded = Node(currentNode.level + 1,currentNode.selectedItems, currentNode.cost,
                                  currentNode.weight, currentNode.bound)

            if nextNotAdded.level >= len(items):
                nextNotAdded.bound = nextNotAdded.cost
            else: 
                nextNotAdded.bound = calculateUpperBound(indextoItem(currentNode.selectedItems,items) ,items, nextNotAdded.level, capacity)
            if nextNotAdded.bound > bestSoFar.cost:
                priorityQueue.push(nextNotAdded)

    takenItems = indextoItem(bestSoFar.selectedItems,items)

    return int(bestSoFar.cost),takenItems


#small errors with this function, created another one because I wans't finding my mistake
def BranchAndBound2(remainingItems,takenItems,index,capacity,bestValue,currentCost,optCost):
    if index == len(remainingItems)  or bestValue == optCost:
        bestValue = max(bestValue, currentCost)    
        return 0,[]
    #pruning
    if calculateUpperBound(takenItems,remainingItems,index,capacity) < bestValue:
        return 0,[]

    res, takenItems = BranchAndBound2(remainingItems,takenItems,index+1,capacity,bestValue,currentCost,optCost)


    if capacity - calculateWeight(remainingItems[index] >= 0): 
        newRes, newTakenItems = BranchAndBound(remainingItems,takenItems,index+1, capacity - calculateWeight(remainingItems[index]),
                            bestValue, 
                            currentCost + calculateValue(remainingItems[index]),optCost)
        newRes += calculateValue(remainingItems[index])
        if newRes > res:
            res = newRes
            takenItems = newTakenItems
            takenItems.append(remainingItems[index])
    return res, takenItems


def DPWeightknapsack(items, capacity):

    table = [[0 for j in range(0,capacity+1)] for i in range(0,len(items)+1)]
    for i in range(1,len(items)+1):
        for j in range(1,capacity+1):
            if j < calculateWeight(items[i-1]):
                table[i][j] = table[i-1][j]
            else:
                table[i][j] = max(table[i-1][j],calculateValue(items[i-1])+table[i-1][j-calculateWeight(items[i-1])])

    n = len(items)
    c = capacity
    takenItems = []
    while c > 0 and n > 0:
        if table[n][c] != table[n-1][c]:
            c -= calculateWeight(items[n-1])
            if c < 0:
                break       
            takenItems.append(items[n-1])
        n -=1

    bestValue = calculateTotalValue(takenItems)

    return bestValue, takenItems

def DPCostknapsack(items,capacity):
    totalValue = calculateTotalValue(items)

    #init of the table
    table = [[0 for j in range(0,totalValue+1)] for i in range(0,len(items)+1)]

    for p in range(1,totalValue+1):
        table[0][p] = math.inf

    #recurrence relation who allows to fill the table
    for i in range(1,len(items)+1):
        for p in range(1,totalValue+1):
            #We find the solution which gets a total value of p with the minimum weight (thus the best solution)
            if calculateValue(items[i-1]) <= p:
                table[i][p] = min(table[i-1][p],calculateWeight(items[i-1])+table[i-1][p - calculateValue(items[i-1])])
            #if we can't find a new solution, we just take the one from the precedent subset
            else:
                table[i][p] = table[i-1][p]

    #we find the best value which correspond to element in the last column,
    # in the field where the weight is less than Capacity and has the maximum index.
    opt = 0
    for p in range(0,totalValue+1):
        bound = table[len(items)][p]
        if bound <= capacity and p>opt:
            opt = p

    #We "walk back" through the table to find which items we should take
    takenItems = []
    i = len(items) 
    while i > 0:
        if calculateValue(items[i-1]) <= opt:
            if calculateWeight(items[i-1]) + table[i-1][opt-calculateValue(items[i-1])] < table[i-1][opt]:
                takenItems.append(items[i-1])
                opt -= calculateValue(items[i-1])
        i -=1
    bestValue = calculateTotalValue(takenItems)
    return bestValue, takenItems
    

def FPTASknapsack(items,capacity,epsilon):

    temp = epsilon*calculateMaxCost(items)/len(items)
    b = math.floor(math.log(temp,2))

    for item in items:
        item[1] = removeLSBs(item[1], b)

    bestValue, takenItems = DPCostknapsack(items,capacity)

    return bestValue, takenItems, b


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
                    selectedItems.append(j)
            if weight > capacity:
                # print(pop[i][selectedItems[random.randint(0, len(selectedItems)-1)]])
                pop[i][selectedItems[random.randint(0, len(selectedItems)-1)]] = 0
        fitness.append(value)
        # print(fitness)

    return fitness

# We Check what percentage of the chromosomes in the population has the same fitness value
def testOfFitness(fit,rate):
    values = set(fit)
    maxCount = 0
    for i in values:
        if maxCount < fit.count(i):
            maxCount = fit.count(i)

    val = float(maxCount)/float(len(fit))
    if  val >= rate:
        return True
    else:
        return False

def crossover(mate1, mate2):
    selected = random.randint(0, len(mate1)-1)
    return mate1[:selected]+mate2[selected:]
  
def mutate(gene, mutate):
    for i in range(len(gene)):
        selected = random.randint(1, mutate)
        if selected == 1:
            gene[i] = bool(gene[i])^1   
    return gene

#Elitism
def selectBest(pop,fit):
    best = 0
    for i in range(len(fit)):
        if fit[i] > fit[best]:
          best = i
    return pop[best]

# Roulette-wheel selection
def selectRoulette(pop, fit):
    size = len(pop)

    selected = random.randint(0, sum(fit))
    tempSum = 0
    fit1 = 0
    for i in range(size):
        tempSum += fit[i]
        if tempSum >= selected:
            mate1 = pop.pop(i)
            fit1 = fit.pop(i)
            break
    tempSum = 0
    selected = random.randint(0, sum(fit))
    for i in range(len(pop)):
        tempSum += fit[i]
        if tempSum >= selected:
            mate2 = pop[i]
            pop.append(mate1)
            fit.append(fit1)
            return mate1, mate2


def newPopulation(pop, fit, mut, elitism):
    popSize = len(pop)
    newPop = []

    if elitism:
        newPop.append(selectBest(pop, fit))
    
    while(len(newPop) < popSize):
        mate1, mate2 = selectRoulette(pop, fit)
        newPop.append(mutate(crossover(mate1, mate2), mut))

    return newPop

def geneticAlgorithm(items,capacity, popSize, mut, maxGen, rate, elitism):

    generation = 0
    pop = generate(len(items),popSize)
    fitness = getFitness(pop,items,capacity)
    while(not testOfFitness(fitness, rate) and generation < maxGen):
        generation += 1
        pop = newPopulation(pop, fitness, mut, elitism)
        fitness = getFitness(pop,items,capacity)

    indexes = selectBest(pop,fitness)
    takenItems = []
    for i in range(len(indexes)):
        if indexes[i] == 1:
            takenItems.append(items[i])

    bestValue = calculateTotalValue(takenItems)

    return bestValue, takenItems

def testAlgorithm(testFiles,solFiles,algorithm):

    if algorithm == "greedy":
        print("\n Greedy Algoritm with cost/weight heuristic : \n")
    elif algorithm == "B&B":
        print("\n Branch & Bound Algoritm : \n")
    elif algorithm == "DPWeight":
        print("\n Dynamic Programming algorithm (By weight): \n")
    elif algorithm == "DPCost":
        print("\n Dynamic Programming algorithm (By cost): \n")
    elif algorithm == "FPTAS":
        print("\n FPTAS Algoritm with a complexity of O(n^3/ε) : \n")
    elif algorithm == "genetic":
        print("\n Genetic Algoritm  : \n")

                # bestValue, knapsack = FPTASknapsack(items,capacity,epsilon)
   # Time computing results / number of items
    results = [[]]

    # loop over the different test files
    if solFiles:
        sol = True
    else:
        sol = False
    for i in range(0,len(testFiles)):
        myarray, solArray, myArraySize, n = prepareData(i,sol)
        # correspond to the computing time w.r.t. the number of items for     
        result = []
        #loop over all the instances
        averageRelativeError = 0
        maximumRelativeError = 0
        averageFPTASRelativeError = 0
        timer = 0

        for i in range(50):
            if sol:
                optCost = solArray[i][2]
            else : 
                optCost = 0                
            capacity =  myarray[i][2]
            testArray = myarray[i][3:myarray.shape[1]]
            testArray = np.split(testArray,myArraySize)
            testArray = np.vstack(testArray)

            items = (addIndex(testArray));

            #remove the empty arrays
            items = [x for x in items if x != []]

            sortingKey = getRatio
            start = time.time()

            if algorithm == "bruteForce":
                knapsack, bestValue, bestWeight = knapsackBrutForce(items,capacity)
            if algorithm == "greedy":
                bestValue, knapsack = DPWeightknapsack(items,capacity)
                optCost = bestValue
                start = time.time()
                knapsack, bestValue, bestWeight = knapsackGreedy(items,capacity,getRatio)
            elif algorithm == "B&B":
                #we sort the items by their ratio
                sortedItems = sorted(items,key=sortingKey,reverse=True)
                bestValue, knapsack = BranchAndBound(sortedItems,capacity)
            elif algorithm == "DPWeight":
                bestValue, knapsack = DPWeightknapsack(items,capacity)
            elif algorithm == "DPCost":
                bestValue, knapsack = DPCostknapsack(items,capacity)
            elif algorithm == "FPTAS":
                epsilon = 0.8
                bestValue, knapsack, b = FPTASknapsack(items,capacity,epsilon)
            elif algorithm == "genetic":
                bestValue, knapsack = DPWeightknapsack(items,capacity)
                optCost = bestValue
                start = time.time()
                bestValue, knapsack = geneticAlgorithm(items,capacity,50,100,myArraySize*10,0.9,True)
                # print("OPtCost", optCost)
                # print("bestValue", bestValue)

            end = time.time()
            timer += float(truncate((end - start),8)) 

            if optCost != 0:
                relativeError = getRelativeError(optCost,bestValue)

                averageRelativeError += relativeError
                if relativeError > maximumRelativeError:
                    maximumRelativeError = relativeError

                if algorithm =="FPTAS":
                    FPTASrelativeError = getFPTASRelativeError(len(items),b,optCost)
                    averageFPTASRelativeError += FPTASrelativeError
            else : 
                relativeError = 0
                maximumRelativeError = 0
                averageFPTASRelativeError = 0

        averageRelativeError /= 50
        timer /= 50
        timer = truncate(timer,4)
        averageRelativeError = truncate(averageRelativeError,3)
        maximumRelativeError = truncate(maximumRelativeError,3)

        result.append(timer)
        result.append(averageRelativeError)

        if algorithm =="FPTAS":
            averageFPTASRelativeError /= 50
            averageFPTASRelativeError = truncate(averageFPTASRelativeError,4)
            # timer = truncate(timer,4)
            # print('Epsilon : ', epsilon, ' -- Number of items : ',n, '-- Computing time', timer, '-- average relative error', averageFPTASRelativeError,)
            # print(timer)
            print(averageFPTASRelativeError)
            result.append(timer)
            result.append(averageFPTASRelativeError)
        else :
            print('Number of items : ',n, '-- Computing time', timer, '-- average relative error', averageRelativeError, '-- max relative error', maximumRelativeError)

    results.append(result)
    return results


def createRandomInstance(numberOfItems,numberOfInstances,ratio,maximumWeight,maximumCost,k,balance):
    # -I  no  integer     Starting ID
    # -n  yes integer     Number of items
    # -N  yes integer     Number of instances
    # -m  yes real        The ratio of the knapsack capacity to the summary weight
    # -W  yes integer     Maximum weight
    # -C  yes integer     Maximum cost
    # -k  yes real    The k exponent.
    # -d  yes -1, 0, 1    -1 … small items are prefered, 1 … big items are prefered, 0 … balanced
    with open("./randomInstances/randomInstance" + numberOfItems + ".txt", "w") as outputFile:
        args = "./knapgen/knapgen/knapgen.exe" + " -n " + numberOfItems + " -N " + numberOfInstances\
         + " -m " + ratio + " -W " + maximumWeight + " -C " + maximumCost + " -k " + k + " -d " + balance 
        subprocess.call(args,stdout=outputFile)


def removeContentOfFolder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def testGenetic(testFiles,solFiles,popSize, mut, maxGen, rate, elitism):
    results = [[]]

    # loop over the different test files
    if solFiles:
        sol = True
    else:
        sol = False
    for i in range(0,len(testFiles)):
        myarray, solArray, myArraySize, n = prepareData(i,sol)
        # correspond to the computing time w.r.t. the number of items for     
        result = []
        #loop over all the instances
        averageRelativeError = 0
        maximumRelativeError = 0
        timer = 0
        for i in range(50):
            if sol:
                optCost = solArray[i][2]
            else : 
                optCost = 0                
            capacity =  myarray[i][2]
            testArray = myarray[i][3:myarray.shape[1]]
            testArray = np.split(testArray,myArraySize)
            testArray = np.vstack(testArray)

            items = (addIndex(testArray));

            #remove the empty arrays
            items = [x for x in items if x != []]

            sortingKey = getRatio
            start = time.time()

            bestValue, knapsack = DPWeightknapsack(items,capacity)
            optCost = bestValue
            start = time.time()
            bestValue, knapsack = geneticAlgorithm(items,capacity, popSize, mut, maxGen, rate, elitism)

            end = time.time()
            timer += float(truncate((end - start),8)) 

            if optCost != 0:
                relativeError = getRelativeError(optCost,bestValue)

                averageRelativeError += relativeError
                if relativeError > maximumRelativeError:
                    maximumRelativeError = relativeError
            else : 
                relativeError = 0
                maximumRelativeError = 0
                averageFPTASRelativeError = 0

        averageRelativeError /= 50
        timer /= 50
        timer = truncate(timer,4)
        averageRelativeError = truncate(averageRelativeError,3)
        maximumRelativeError = truncate(maximumRelativeError,3)

        result.append(timer)
        result.append(averageRelativeError)
        result.append(maximumRelativeError)

    results.append(result)
    return results

def expEvalGenetic():
    # genetic Algorithm
    defaultPopSize = 30
    defaultMut = 100
    defaultRate = 0.9 
    defaultElitism = True

    defaultNumberOfItems = "20"
    defaultRatio = "0.7"
    defaultNumberOfInstances = "50"
    defaultMaximumWeight = "100"
    defaultMaximumCost = "100"
    defaultK = "1.0"
    defaultBalance = "0"

    popSizeArray = [10,20,30,40,50]
    mutArray = [100000,100,50,20,10,5,2]
    rateArray = [50,70,80,90]
    elitismArray = [True,False]

    fh = open("./results/genetic.txt","a") 

    removeContentOfFolder("./randomInstances/")
    createRandomInstance(defaultNumberOfItems,defaultNumberOfInstances,defaultRatio,defaultMaximumWeight,defaultMaximumCost,defaultK,defaultBalance)
    testFiles = [f for f in listdir('./randomInstances/') if isfile(join('./randomInstances/', f))]

    for popSize in popSizeArray:
        results = testGenetic(testFiles,False,popSize,defaultMut,200,defaultRate,defaultElitism)
        result = results.pop()
        time = str(result[0])
        score = str(result[1])
        maximumError = str(result[2])
        string = str(popSize) + " " + str(defaultMut) + " " + str(defaultRate) + " " + str(defaultElitism) + " " + time + " " + score + " " + maximumError +  "\n"
        fh.write(string) 
    for mut in mutArray:
        results = testGenetic(testFiles,False,defaultPopSize,mut,200,defaultRate,defaultElitism)
        result = results.pop()
        time = str(result[0])
        score = str(result[1])
        maximumError = str(result[2])
        string = str(defaultPopSize) + " " + str(mut) + " " + str(defaultRate) + " " + str(defaultElitism) + " " + time + " " + score + " " + maximumError +  "\n"
        fh.write(string) 
    for rate in rateArray:
        results = testGenetic(testFiles,False,defaultPopSize,defaultMut,200,rate,defaultElitism)
        result = results.pop()
        time = str(result[0])
        score = str(result[1])
        maximumError = str(result[2])
        string = str(defaultPopSize) + " " + str(defaultMut) + " " + str(rate) + " " + str(defaultElitism) + " " + time + " " + score + " " + maximumError +  "\n"
        fh.write(string) 
    for elitism in elitismArray:
        results = testGenetic(testFiles,False,defaultPopSize,defaultMut,200,defaultRate,elitism)
        result = results.pop()
        time = str(result[0])
        score = str(result[1])
        maximumError = str(result[2])
        string = str(defaultPopSize) + " " + str(defaultMut) + " " + str(defaultRate) + " " + str(elitism) + " " + time + " " + score + " " + maximumError +  "\n"
        fh.write(string) 


def expEval(algorithm):  

    defaultNumberOfItems = "10"
    defaultRatio = "0.7"
    defaultNumberOfInstances = "50"
    defaultMaximumWeight = "100"
    defaultMaximumCost = "100"
    defaultK = "1.0"
    defaultBalance = "0"
   

    numberOfItemsArray = ["5","10","15","20","25"]
    ratioArray = ["0","0.2","0.5","0.7","0.9"]
    maximumWeightArray = ["50","100","500","2000","5000"]
    maximumCostArray =  ["50","100","500","2000","5000"]
    kArray = ["0","0.3","1","2","10","100"]
    balanceArray = ["-1","0","1"]

    fh = open("./results/"+algorithm+".txt","a") 

    for number in numberOfItemsArray:
        removeContentOfFolder("./randomInstances/")
        createRandomInstance(number,defaultNumberOfInstances,defaultRatio,defaultMaximumWeight,defaultMaximumCost,defaultK,defaultBalance)
        testFiles = [f for f in listdir('./randomInstances/') if isfile(join('./randomInstances/', f))]
        results = testAlgorithm(testFiles,False,algorithm)
        result = results.pop()
        time = str(result[0])
        score = str(result[1])
        string = str(number) + " " + defaultRatio + " " + defaultNumberOfInstances + " " + defaultMaximumWeight + " " +defaultMaximumCost\
         + " " + defaultK + " " + defaultBalance + " " + time + " " + score + "\n"
        fh.write(string) 
    for ratio in ratioArray:
        removeContentOfFolder("./randomInstances/")
        createRandomInstance(defaultNumberOfItems,defaultNumberOfInstances,ratio,defaultMaximumWeight,defaultMaximumCost,defaultK,defaultBalance)
        testFiles = [f for f in listdir('./randomInstances/') if isfile(join('./randomInstances/', f))]
        results = testAlgorithm(testFiles,False,algorithm)
        result = results.pop()
        time = str(result[0])
        score = str(result[1])
        string = defaultNumberOfItems + " " + ratio + " " + defaultNumberOfInstances + " " + defaultMaximumWeight + " " +defaultMaximumCost\
         + " " + defaultK + " " + defaultBalance + " " + time + " " + score + "\n" 
        fh.write(string) 
 
    for maximumWeight in maximumWeightArray:
        removeContentOfFolder("./randomInstances/")
        createRandomInstance(defaultNumberOfItems,defaultNumberOfInstances,defaultRatio,maximumWeight,defaultMaximumCost,defaultK,defaultBalance)
        testFiles = [f for f in listdir('./randomInstances/') if isfile(join('./randomInstances/', f))]
        results = testAlgorithm(testFiles,False,algorithm)
        result = results.pop()
        time = str(result[0])
        score = str(result[1])
        string = defaultNumberOfItems + " " + defaultRatio + " " + defaultNumberOfInstances + " " + maximumWeight + " " +defaultMaximumCost\
         + " " + defaultK + " " + defaultBalance + " " + time + " " + score + "\n" 
        fh.write(string) 
 
    for maximumCost in maximumCostArray:
        removeContentOfFolder("./randomInstances/")
        createRandomInstance(defaultNumberOfItems,defaultNumberOfInstances,defaultRatio,defaultMaximumWeight,maximumCost,defaultK,defaultBalance)
        testFiles = [f for f in listdir('./randomInstances/') if isfile(join('./randomInstances/', f))]
        results = testAlgorithm(testFiles,False,algorithm)
        result = results.pop()
        time = str(result[0])
        score = str(result[1])
        string = defaultNumberOfItems + " " + defaultRatio + " " + defaultNumberOfInstances + " " + defaultMaximumWeight + " " +maximumCost\
         + " " + defaultK + " " + defaultBalance + " " + time + " " + score + "\n" 
        fh.write(string) 
 
    for k in kArray:
        removeContentOfFolder("./randomInstances/")
        createRandomInstance(defaultNumberOfItems,defaultNumberOfInstances,defaultRatio,defaultMaximumWeight,defaultMaximumCost,k,defaultBalance)
        testFiles = [f for f in listdir('./randomInstances/') if isfile(join('./randomInstances/', f))]
        results = testAlgorithm(testFiles,False,algorithm)
        result = results.pop()
        time = str(result[0])
        score = str(result[1])
        string = defaultNumberOfItems + " " + defaultRatio + " " + defaultNumberOfInstances + " " + defaultMaximumWeight + " " +defaultMaximumCost\
         + " " + k + " " + defaultBalance + " " + time + " " + score + "\n" 
        fh.write(string) 
 
    for balance in balanceArray:
        removeContentOfFolder("./randomInstances/")
        createRandomInstance(defaultNumberOfItems,defaultNumberOfInstances,defaultRatio,defaultMaximumWeight,defaultMaximumCost,defaultK,balance)
        testFiles = [f for f in listdir('./randomInstances/') if isfile(join('./randomInstances/', f))]
        results = testAlgorithm(testFiles,False,algorithm)
        result = results.pop()
        time = str(result[0])
        score = str(result[1])
        string = defaultNumberOfItems + " " + defaultRatio + " " + defaultNumberOfInstances + " " + defaultMaximumWeight + " " +defaultMaximumCost\
         + " " + defaultK + " " + balance + " " + time + " " + score + "\n" 
        fh.write(string) 
 
    fh.close() 


def main():

    # We get all the test and solutions files
    testFiles = [f for f in listdir('./inst/') if isfile(join('./inst/', f))]
    solFiles = [f for f in listdir('./sol/') if isfile(join('./sol/', f))]

    # expEval("bruteForce")
    # expEval("B&B")
    # expEval("DPCost")
    # expEval("DPWeight")
    # expEval("greedy")
    expEvalGenetic()
    # plotData(algorithm,numberOfLines,column1,column2,indexStart,numberOfItems,xlabel,ylabel,title)
    # plotData("B&B",29,0,7,0,5)
    # testAlgorithm(testFiles,solFiles,"genetic")
    # testAlgorithm(testFiles,solFiles,"greedy")
    # testAlgorithm(testFiles,solFiles,"DPCost")
    # testAlgorithm(testFiles,solFiles,"DPWeight")
    # testAlgorithm(testFiles,solFiles,"FPTAS")
    # testAlgorithm(testFiles,solFiles,"bruteForce")
    # testAlgorithm(testFiles,solFiles,"B&B")
    # plotBruteForceAndBranchAndBound()
    # plotData("B&B",29,0,7,0,5,"Number of items","Computing time (s)","Computing time as a function of the instance size for the B&B Algorithm ")
    # plotData("DPCost",29,4,7,15,5,"Maximum Cost","Computing time (s)","Computing time as a function of the maximum cost for the DP By Cost")
    # plotData("DPWeight",29,3,7,10,5,"Maximum Weight","Computing time (s)","Computing time as a function of maximum costfor the DP By Weight")


if __name__ == "__main__": main()