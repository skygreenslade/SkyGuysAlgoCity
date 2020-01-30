import numpy as np
import math
import matplotlib.pyplot as plt

import functions

# function to evaluate the cost of given function for a given vector of n values
def evaluaterecurrence(myfunction, nvec):
    costs = []
    for n in nvec:
        costs.append(funcDict[myfunction](n))
    return costs

# function to generate linear vector of values from given starting number, with given interval
def genNvec(startingN, samples, rateOfChange):
    toRet = []
    n = startingN
    for i in range(samples):
        toRet.append(n)
        n = n+rateOfChange
    return toRet

# function to generate powers of 2 of given starting number
def genNvecPowersOf2(startingN, samples):
    toRet = []
    n = startingN
    for i in range(samples):
        toRet.append(n)
        n = n *2
    return toRet

# dictionary to retrieve function names
funcDict = {
    'mergeSortFunc': functions.mergeSortFunc,
    'f2': functions.hanoi,
    'f3': functions.f3,
    'f4': functions.f4
}

# generate vectors
nvec1 = genNvecPowersOf2(200, 100)
nvec2 = genNvec(350, 20, 1)
nvec3 = genNvecPowersOf2(35, 50)
nvec4 = genNvec(350, 100, 150)

# generate values for analytic solutions
# nlogn
analytic1 = np.zeros(len(nvec1))
for pos in range(len(nvec1)):
    analytic1[pos] = nvec1[pos] * math.log(nvec1[pos])

# 2^n (*1.1 for visibility)
analytic2 = np.zeros(len(nvec2))
for pos in range(len(nvec2)):
    analytic2[pos] = (2**nvec2[pos])*(1.1)

# n^2 (*10 for visibility)
analytic3 = np.zeros(len(nvec3))
for pos in range(len(nvec3)):
    analytic3[pos] = (nvec3[pos]**2)*10

# n^2
analytic4 = np.zeros(len(nvec4))
for pos in range(len(nvec4)):
    analytic4[pos] = (nvec4[pos]**2)

# Plot results
plt.plot(nvec1, evaluaterecurrence('mergeSortFunc', nvec1))
plt.plot(nvec1, analytic1)
plt.legend(['Program Output', 'Analytic Solution: nlogn'])
plt.xlabel("n value")
plt.ylabel("cost")
plt.title("Merge Sort Cost")
plt.show()

plt.figure()
plt.plot(nvec2, evaluaterecurrence('f2', nvec2))
plt.plot(nvec2, analytic2)
plt.legend(['Program Output', 'Analytic Solution: 1.1*2^n'])
plt.xlabel("n value")
plt.ylabel("cost")
plt.title("Tower of Hanoi Cost")
plt.show()

plt.figure()
plt.plot(nvec3, evaluaterecurrence('f3', nvec3))
plt.plot(nvec3, analytic3)
plt.legend(['Program Output', 'Analytic Solution: 10*n^2'])
plt.xlabel("n value")
plt.ylabel("cost")
plt.title("Recursive Algorithm 3 Cost")
plt.show()

plt.figure()
plt.plot(nvec4, evaluaterecurrence('f4', nvec4))
plt.plot(nvec4, analytic4)
plt.legend(['Program Output', 'Analytic Solution: n^2'])
plt.xlabel("n value")
plt.ylabel("cost")
plt.title("Recursive Algorithm 4 Cost")
plt.show()
