import time
from fractions import Fraction
import random
import numpy as np
import math
from os import listdir
from os.path import isfile, join
import os, shutil
import collections, numpy
import matplotlib.pyplot as plt
import pandas as pd



def removeContentOfFolder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def randBool():
    return random.choice([True, False])

def randLit(clause, nbVariables):
    """Return a randomly generated literal from variables not appeared in the clause.
    """
    while (True):
        v = random.randint(1, nbVariables)
        if (v not in clause and -v not in clause):
            break
    return v if randBool() else -v

def randWeight(rangeMax):

    return random.randint(1,rangeMax)


def generateInstances(path,numberOfInstances,k, nbClauses, nbVariables,isSat,rangeMax,remove):
    # Generate random k-SAT instance.

    # Args:
    # k: The size of each clause.
    # nbvar: The number of variables.
    # nbClauses: The number of clauses
    # issat: indicating whether the generated SAT instance will be satisfiable or not.
    # rangeMax : maximum weight

    # c CNF Example
    # c 4 variables, 6 clauses
    # c each clause is terminated by '0' (not by the end of line)
    # p cnf 4 6
    # 1 -3 4 0
    # -1 2 -3 0
    # 3 4 0
    # 1 2 -3 -4 0
    # -2 3 0
    # -3 -4 0

    #We delete all the previously created instances
    if remove:
        removeContentOfFolder(path)

    for i in range(numberOfInstances):
        with open(path+str(k)+"-SAT_"+str(nbClauses)+"_"+str(nbVariables)+"_"+str(i)+".txt","w") as outputFile:
            formula = []
            model = [var if randBool() else -var for var in range(1, nbVariables+1)]

            outputFile.write(" p cnf "+str(nbVariables)+" "+str(nbClauses)+"\n")
            weights = list()
            for i in range(nbVariables):
                weights.append(randWeight(rangeMax))
            outputFile.write(" ".join(map(str, weights)))
            outputFile.write("\n")

            for i in range(nbClauses):
                clause = list()
                if isSat:
                    clause.append(random.choice(model))

                for _ in range(k - len(clause)):
                    clause.append(randLit(clause, nbVariables))

                random.shuffle(clause)
                clause.append(0)
                outputFile.write(" ".join(map(str, clause)))
                outputFile.write("\n")
            
           
        outputFile.close()

def loadInstance(path,i,k,nbClauses,nbVariables):
    instancePath = path + str(k)+"-SAT_"+str(nbClauses)+"_"+str(nbVariables)+"_"+str(i)+".txt"
    instance = np.genfromtxt(instancePath,skip_header=2)

    f = open(instancePath)
    data = f.read()
    first_line = data.split('\n', 1)[0]
    second_line = data.split('\n', 2)[1]
    f.close()

    nbClauses = first_line.split(' ')[4]
    nbVars = first_line.split(' ')[3]

    weights = second_line.split(' ')
    
    return [instance, nbClauses, nbVars, weights]

def loadAllInstances(path,n,k,nbClauses,nbVariables,):     

    instances = []
    for i in range(n):
        instances.append(loadInstance(path,i,k,nbClauses,nbVariables))

    return instances

def generate(nbVars,popSize):
    pop = [[random.randint(0,1) for i in range(nbVars)] for j in range(popSize)]
    return pop


def evalWeightedClause(clause,pop,k):
    weight = 0

    binaryClause = np.zeros(k,dtype=np.int32)
    binaryPop = np.zeros(k,dtype=np.int32)
    pop = np.asarray(pop)
    for i in range(k):
        if clause[i] > 0:
            index = int(clause[i])-1
            value = 1
        else :
            index = int(abs(clause[i])-1)
            value = 0
        binaryClause[i] = value
       
        try:
            binaryPop[i] = pop[index]
        except IndexError as err:
            print(i)
            print(index)

    orArray = (binaryClause == binaryPop)

    if any(orArray):
        return True
    return False

def calculateSumPop(pop,weights):
    try:
        pop = np.array(pop).astype(int)
    except TypeError as err:
        print("pop", pop)
    # pop = np.array(pop).astype(int)
    weights = np.array(weights).astype(int)
    res = weights[np.where(pop > 0)]
    res = np.sum(res)
    return res


def evalWeightedCNF(cnf,pop,nbClauses,nbVars,k,weights):
    weight = 0 
    satisfiedClauses = []
    for i in range(nbClauses):
        clause = cnf[i]
        satisfiedClauses.append(evalWeightedClause(clause,pop,k))

    notSatisfiedClauses = nbClauses - sum(satisfiedClauses)

    if notSatisfiedClauses == 0:
        ratio = 1 
    else : 
        ratio = notSatisfiedClauses+1


    res = calculateSumPop(pop,weights)/ratio
    return res


def getFitness(pop,cnf,nbClauses,nbVars,k,weights):
    fitness = np.zeros(len(pop),dtype=np.int32)
    fitness = []

    for i in range(len(pop)):
        fitness.append(evalWeightedCNF(cnf,pop[i],nbClauses,nbVars,k,weights))
    return fitness

def testOfFitness(fit,rate):
    counter = collections.Counter(fit)
    values = set(fit)
    maxCount = 0

    for i in values:
        if maxCount < counter[i]:
            maxCount = counter[i]

    val = float(maxCount)/float(len(fit))

    # test of convergence
    if  val >= rate:
        return True
    else:
        return False        

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

    selected = random.uniform(0, sum(fit))
    tempSum = 0
    fit1 = 0
    for i in range(size):
        tempSum += fit[i]
        if tempSum >= selected:
            mate1 = pop.pop(i)
            fit1 = fit.pop(i)
            break
    tempSum = 0
    selected = random.uniform(0, sum(fit))
    for i in range(len(pop)):
        tempSum += fit[i]
        if tempSum >= selected:
            mate2 = pop[i]
            pop.append(mate1)
            fit.append(fit1)
            return mate1, mate2

def crossover(mate1, mate2):
    selected = random.randint(0, len(mate1)-1)
    return mate1[:selected]+mate2[selected:]
  
def mutate(gene, mutate):
    if mutate != 0:
        for i in range(len(gene)):
            selected = random.randint(1, mutate)
            if selected == 1:
                gene[i] = bool(gene[i])^1   
        return gene


def newPopulation(pop,fit,mut,elitism):
    popSize = len(pop)
    newPop = []

    if elitism:
        newPop.append(selectBest(pop, fit))
    
    while(len(newPop) < popSize):
        mate1, mate2 = selectRoulette(pop, fit)
        newPop.append(mutate(crossover(mate1, mate2), mut))

    return newPop


def selectFinalBest(pop,fit):
    best = 0
    for i in range(len(fit)):
        if fit[i] > fit[best]:
          best = i
    return pop[best],fit[best]

def geneticAlgorithm(instance,popSize,mut,maxGen,rate,elitism,k):
    cnf, nbClauses, nbVars, weights = instance

    nbClauses = int(nbClauses)
    nbVars = int(nbVars)

    generation = 0
    pop = generate(nbVars,popSize)
    fitness = getFitness(pop,cnf,nbClauses,nbVars,k,weights)
    while(not testOfFitness(fitness,rate) or generation < maxGen):
        generation += 1
        pop = newPopulation(pop,fitness,mut,elitism)
        fitness = getFitness(pop,cnf,nbClauses,nbVars,k,weights)

    bestPop,bestFit = selectFinalBest(pop,fitness)
    return bestPop,bestFit
           
# tun the GA on the given instances
def testGenetic(instances,k):

    # defaultPopSize = 30
    # defaultMut = 100
    # defaultRate = 0.9 
    # defaultElitism = True

    # defaultPopSize = 60
    # defaultMut = 1600
    # defaultRate = 0.89999999 
    # defaultElitism = True

    defaultPopSize = 60
    defaultMut = 1700
    defaultRate = 0.6
    defaultElitism = True

    res = getSumWeight(instances,k)
    fitScore = 0
    start = time.time()
    for instance in instances:
        bestPop,bestFit = geneticAlgorithm(instance,defaultPopSize,defaultMut,10,defaultRate,defaultElitism,k)
        fitScore += bestFit
    end = time.time()
    timer = round(end - start,3)
    print("numberOfInstances : ", len(instances))
    print("Score : ",fitScore)
    print("maximum weight : ",res[0])
    print("Clauses / Vars ratio : ",getClauseVarRatio(int(res[2]),int(res[1])))
    print("Score / maximum weight : ", round((fitScore/res[0]),3))
    print(" time : ", timer," s")


#function used by MetaGA to get the best params
def evalGA(path,popSize,mut,maxGen,rate,elitism,ratio):
    
    numerator, denominator = getFraction(ratio)
    instances = loadAllInstances("./instancesMAXSAT/",10,3,numerator,denominator)

    fitScore = 0

    start = time.time()

    for instance in instances:
        bestPop,bestFit = geneticAlgorithm(instance,popSize,mut,maxGen,rate,elitism,3)
        fitScore += bestFit
    end = time.time()
    timer = round(end - start,3)

    return fitScore,timer

#function used by MetaGA to get the best params on all the available Instances
def evalGAallInstances(path,popSize,mut,maxGen,rate,elitism):
    
    ratios = np.arange(0.5,10,0.5)
    instancesArray = []

    # for all different instances independently of the size or anything else
    for ratio in ratios:
        numerator, denominator = getFraction(ratio)
        instancesArray.append(loadAllInstances("./instancesMAXSAT/",10,3,numerator,denominator))

    instances = [instance for i in instancesArray for instance in i]
    fitScore = 0

    start = time.time()

    for instance in instances:
        bestPop,bestFit = geneticAlgorithm(instance,popSize,mut,maxGen,rate,elitism,3)
        fitScore += bestFit
    end = time.time()
    timer = round(end - start,3)

    return fitScore,timer


def getClauseVarRatio(nbClauses,nbVars):
    return nbClauses/nbVars

#Calculate the total sum of all the instances if they could be all satisfied
def getSumWeight(instances,k):
    weight = 0 
    for instance in instances:
        cnf, nbClauses, nbVars,weights = instance
        for w in weights:
            weight += int(w)
    return [weight, nbVars, nbClauses]

#get a numerator and a denominator corresponding to a fraction, used 
# to get the number of variables and clauses from a ratio

def getFraction(ratio):
    fraction = Fraction(ratio).limit_denominator()

    numerator = fraction.numerator
    denominator = fraction.denominator
    # We can't have a number of variables inferior to 3
    while denominator < 3 :
        numerator *= 2
        denominator *=2
    return numerator, denominator 

# is not used
def generateAllInstances(startingRatio,endingRatio,interval):

    for i in range(startingRatio,endingRatio,interval):
        numerator, denominator = getFraction(i)
        generateInstances("./instancesMAXSAT/",30,3,numerator,denominator,True,10)

# Use to run the GA on generatedInstances
def performTests(popSize,mut,maxGen,rate,elitism,k,alreadyGeneretad):

    ratios = np.arange(0.5,10,0.5)
    instancesArray = []

    if not alreadyGeneretad:
        removeContentOfFolder("./instancesMAXSAT/")
        for ratio in ratios:
            numerator, denominator = getFraction(ratio)
            generateInstances("./instancesMAXSAT/",30,3,numerator*4,denominator*4,False,10,remove=False)

    for ratio in ratios:
        numerator, denominator = getFraction(ratio)
        instancesArray.append(loadAllInstances("./instancesMAXSAT/",30,3,numerator*4,denominator*4))

    for instances in instancesArray:

        fitScore = 0
        res = getSumWeight(instances,k)

        start = time.time()
        for instance in instances:
            bestPop,bestFit = geneticAlgorithm(instance,popSize,mut,maxGen,rate,elitism,k)
            fitScore += bestFit
        end = time.time()
        timer = round(end - start,3)

        print("-------------------------------")
        print("     Clauses / Vars ratio : ",getClauseVarRatio(int(res[2]),int(res[1])))
        print("     numberOfInstances : ", len(instances))
        print("     Score : ",fitScore)
        print("     maximum weight : ",res[0])
        print("     Score / maximum weight : ", round((fitScore/res[0]),3))
        print("     time : ", timer," s")

        with open("./resultsMAXSAT/"+str(k)+"-SAT_6.txt","a") as outputFile:
            outputFile.write(str(len(instances))+" "+str(int(res[2]))+" "+str(int(res[1]))+" "+str(getClauseVarRatio(int(res[2]),int(res[1])))
                +" "+str(fitScore)+" "+str(res[0])+" "+str(round((fitScore/res[0]),3))+" "+str(timer)+"\n")
        outputFile.close()

# used to plot the results of the running of the GA 
def plotResults():

    # path of the result file
    instancePath = "./resultsMAXSAT/3-SAT_3.txt"
    instance = np.genfromtxt(instancePath)

    ratios = instance[:,3]
    times = instance[:,-1]
    scores = instance[:,-2]

    print(ratios)
    fig,ax1 = plt.subplots()

    ax1.plot(ratios, scores, color="red",label='Score')
    ax1.set_xlabel('ratio',fontsize=18)
    ax1.set_ylabel('Score / maximum weight',fontsize=18)

    ax2 = ax1.twinx()
    ax2.plot(ratios, times, color="blue",label='Time')
    ax2.set_ylabel('time (s)',fontsize=18)

    # Now add the legend with some customizations.
    legend = ax1.legend(loc=6, shadow=True)
    legend = ax2.legend(loc=7, shadow=True)
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        # label.set_fontname('Arial')
        label.set_fontsize(16)

    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        # label.set_fontname('Arial')
        label.set_fontsize(16)

    plt.show()

# used to plot the results of the running of the GA 
def plotAllResults():

    # path of the result file
    instancePath1 = "./resultsMAXSAT/3-SAT_3.txt"
    instance1 = np.genfromtxt(instancePath1)
    instancePath2 = "./resultsMAXSAT/3-SAT_4.txt"
    instance2 = np.genfromtxt(instancePath2)
    instancePath3 = "./resultsMAXSAT/3-SAT_5.txt"
    instance3 = np.genfromtxt(instancePath3)
    instancePath4 = "./resultsMAXSAT/3-SAT_6.txt"
    instance4 = np.genfromtxt(instancePath4)

    ratios1 = instance1[:,3]
    times1 = instance1[:,-1]
    scores1 = instance1[:,-2]

    ratios2 = instance2[:,3]
    times2 = instance2[:,-1]
    scores2 = instance2[:,-2]

    ratios3 = instance3[:,3]
    times3 = instance3[:,-1]
    scores3 = instance3[:,-2]

    ratios4 = instance4[:,3]
    times4 = instance4[:,-1]
    scores4 = instance4[:,-2]

    fig,ax1 = plt.subplots()

    ax1.plot(ratios1, scores1, color="red",label='Score')
    ax1.plot(ratios2, scores2, color="red")
    ax1.plot(ratios3, scores3, color="red")
    ax1.plot(ratios4, scores4, color="red")
    ax1.set_xlabel('ratio',fontsize=18)
    ax1.set_ylabel('Score / maximum weight',fontsize=18)

    ax2 = ax1.twinx()
    ax2.plot(ratios1, times1, color="blue",label='Time')
    ax2.plot(ratios2, times2, color="blue")
    ax2.plot(ratios3, times3, color="blue")
    ax2.plot(ratios4, times4, color="blue")
    ax2.set_ylabel('time (s)',fontsize=18)

    # Now add the legend with some customizations.
    legend = ax1.legend(loc=6, shadow=True)
    legend = ax2.legend(loc=7, shadow=True)
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        # label.set_fontname('Arial')
        label.set_fontsize(16)

    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        # label.set_fontname('Arial')
        label.set_fontsize(16)

    plt.show()
    

#used to plot the table of the results of the Meta GA 
def plotTable():
    instancePath = "./resultsMAXSATMETAGA/3-SAT_result.txt"
    results = np.genfromtxt(instancePath)

    headerValues = ['Ratio','PopSize','MutRate', 'Rate of convergence', 'MaxGen', 'time (s)']

    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    df = pd.DataFrame(results, columns=headerValues)
    table = ax.table(cellText=df.values,colLabels=headerValues, loc='center', cellLoc = 'center', rowLoc = 'center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    fig.tight_layout()
    plt.show()

def main():
    # numberOfInstances,k, nbClauses, nbVariables,isSat,rangeMax
    # case with nb Vars == 1 doesn't work
    # generateInstances("./instancesMAXSAT/",30,3,10,5,False,10,remove=True)
    # instances = loadAllInstances("./instancesMAXSAT/",30,3,10,5)
    # print(instances[0])
    # testGenetic(instances,3)

    # defaultPopSize = 60
    # defaultMut = 1700
    # defaultMaxGen = 10
    # defaultRate = 0.6
    # defaultElitism = True

    defaultPopSize = 30
    defaultMut = 4100
    defaultMaxGen = 10
    defaultRate = 0.6
    defaultElitism = True


    performTests(defaultPopSize, defaultMut,defaultMaxGen,defaultRate,defaultElitism,3,False)
    # plotResults()
    # plotAllResults()
    # plotTable()

if __name__ == "__main__":
    main()
