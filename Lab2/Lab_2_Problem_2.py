import numpy as np
import time
import matplotlib.pyplot as plt
from Lab_2_Problem_1 import evaluaterecurrence

def recursiveMatrixMultiply(a, b, c, aRowStart, aRowEnd, aColStart, aColEnd,
                            bRowStart, bRowEnd, bColStart, bColEnd, cRowStart, cRowEnd, cColStart, cColEnd):

    # Check that matrix shapes are valid for multiplication
    if (np.shape(a)[1] != np.shape(b)[0]):
        print("Given matrix sizes cannot be multiplied together")
        SystemExit()

    # number of rows and columns in each given matrix
    aRows = aRowEnd-aRowStart
    aCols = aColEnd-aColStart
    bRows = bRowEnd-bRowStart
    bCols = bColEnd-bColStart

    # if one matrix is 1x1, find result (base case)
    if(aRows == aCols == 1):
        c[cRowStart:cRowEnd, cColStart:cColEnd] = c[cRowStart:cRowEnd, cColStart:cColEnd] + a[aRowStart, aColStart]*b[bRowStart:bRowEnd, bColStart:bColEnd]

    elif (bRows == bCols == 1):
        c[cRowStart:cRowEnd, cColStart:cColEnd] = c[cRowStart:cRowEnd, cColStart:cColEnd] + a[aRowStart:aRowEnd, aColStart:aColEnd]*b[bRowStart, bColStart]

    # catch occurrence of 0x0 matrix multiplication
    elif(aRows * aCols * bRows * bCols == 0):
        return
        # do nothing

    # Divide Step
    else:
        aRowSplit = (aRows // 2) + aRowStart # index to divide matrix A rows
        aColSplit = (aCols // 2) + aColStart # index to divide matrix A columns
        bRowSplit = (bRows // 2) + bRowStart # where to divide matrix B columns
        bColSplit = (bCols // 2) + bColStart # where to divide matrix B rows
        cRowSplit = (aRows // 2) + cRowStart # where the splits match up with the resultant matrix
        cColSplit = (bCols // 2) + cColStart

        # Call recursive function to solve each subsection of the matrix
        #c11
        recursiveMatrixMultiply(a, b, c, aRowStart, aRowSplit, aColStart, aColSplit, bRowStart, bRowSplit, bColStart, bColSplit,cRowStart, cRowSplit, cColStart, cColSplit)
        recursiveMatrixMultiply(a, b, c, aRowStart, aRowSplit, aColSplit, aColEnd, bRowSplit, bRowEnd, bColStart, bColSplit, cRowStart, cRowSplit, cColStart, cColSplit)

        #c12
        recursiveMatrixMultiply(a, b, c, aRowStart, aRowSplit, aColStart, aColSplit, bRowStart, bRowSplit, bColSplit, bColEnd, cRowStart, cRowSplit, cColSplit, cColEnd)
        recursiveMatrixMultiply(a, b, c, aRowStart, aRowSplit, aColSplit, aColEnd, bRowSplit, bRowEnd, bColSplit, bColEnd, cRowStart, cRowSplit, cColSplit, cColEnd)

        #c21
        recursiveMatrixMultiply(a, b, c, aRowSplit, aRowEnd, aColStart, aColSplit, bRowStart, bRowSplit, bColStart, bColSplit, cRowSplit, cRowEnd, cColStart, cColSplit)
        recursiveMatrixMultiply(a, b, c, aRowSplit, aRowEnd, aColSplit, aColEnd, bRowSplit, bRowEnd, bColStart, bColSplit, cRowSplit, cRowEnd, cColStart, cColSplit)

        #c22
        recursiveMatrixMultiply(a, b, c, aRowSplit, aRowEnd, aColStart, aColSplit, bRowStart, bRowSplit, bColSplit, bColEnd, cRowSplit, cRowEnd, cColSplit, cColEnd)
        recursiveMatrixMultiply(a, b, c, aRowSplit, aRowEnd, aColSplit, aColEnd, bRowSplit, bRowEnd, bColSplit, bColEnd, cRowSplit, cRowEnd, cColSplit, cColEnd)

    # return nothing


# Brute-Force Matrix Multiplication used in Lab 1
def lab1MM(A, B):
    if (np.shape(A)[1] != np.shape(B)[0]):
        return -1
    Ar, Ac = np.shape(A)
    Br, Bc = np.shape(B)
    C = np.zeros((Ar, Bc))
    for row in range(Ar):
        for col in range(Bc):
            sum_ = 0
            for i in range(Ac):
                sum_ += A[row, i] * B[i, col]
            C[row, col] = sum_
    return C


def MM(A, B):
    if(np.shape(A)[1]!=np.shape(B)[0]):
        return -1
    C = np.zeros((np.shape(A)[0], np.shape(B)[1]))
    recursiveMatrixMultiply(A, B, C, 0, np.shape(A)[0], 0, np.shape(A)[1], 0, np.shape(B)[0], 0, np.shape(B)[1], 0, np.shape(A)[0], 0,  np.shape(B)[1])
    return C

#prompt user for matrix sizes
ar = int(input("Enter A rows value\n"))
ac = int(input("Enter A columns value (also B rows value)\n"))
br = ac
bc = int(input("Enter B columns value\n"))

#generate random matrices for given matrix sizes
a = np.matrix(np.random.rand(ar,ac))
b = np.matrix(np.random.rand(br,bc))

# Run each method and record the time taken
startTime = time.time()
myP = MM(a, b)
myTime = time.time() -startTime
startTime = time.time()
npP = a*b
npTime = time.time()-startTime

#output time taken
if (np.array_equal(np.round(myP * 1000), np.round(npP * 1000))):
    print("Mine took: " + str(myTime) + " seconds")
    print("Numpy took: " + str(npTime) + " seconds")
else:
    print("Product was incorrect")


# run problem for matrix sizes 1x1 through n^nxn^n (128x128)
n = 1
myTimes = []
npTimes = []
sizes = []
for i in range(8):

    # Generate new matrices
    a = np.matrix(np.random.rand(n, n))
    b = np.matrix(np.random.rand(n, n))

    # Perform multiplication and record times
    startTime = time.time()
    myP = MM(a, b)
    myTime = time.time()-startTime

    startTime = time.time()
    npP = a * b
    npTime = time.time() - startTime

    if not np.array_equal(np.round(npP*1000), np.round(myP*1000)):
        continue
    myTimes.append(myTime)
    npTimes.append(npTime)
    sizes.append(n)
    print("finished size " + str(n))
    n = n * 2

    # output time taken
    if (np.array_equal(np.round(myP * 1000), np.round(npP * 1000))):
        print("Mine took: " + str(myTime) + " seconds")
        print("Numpy took: " + str(npTime) + " seconds")
    else:
        print("Product was incorrect")

# calculate predicted times
predicted = evaluaterecurrence('MM', sizes)
scaledPred = []
for num in predicted:
    scaledPred.append(num*(10**-3.25))

# closed-form solution times
closed = []
for num in sizes:
    closed.append((num**3)*(10**-5.25))

# plot results
plt.plot(sizes, myTimes)
plt.plot(sizes, scaledPred)
plt.plot(sizes, closed)
plt.xlabel("n")
plt.ylabel("time (s)")
plt.title("My recursive Matrix Multiplication in python")
plt.legend(['Actual Measured Times', 'Predicted Times', 'Closed-form Solution (n^3)'])
plt.show()
plt.figure()


# compare recursive method to Lab1 brute force method
recursiveTimes = []
bruteForceTimes = []
matrixSizes = []
for n in range(1, 150, 20):
    # Generate new matrices
    a = np.matrix(np.random.rand(n, n))
    b = np.matrix(np.random.rand(n, n))
    matrixSizes.append(n)

    #calculate times
    startTime = time.time()
    MM(a, b)
    recursiveTimes.append(time.time() - startTime)

    startTime = time.time()
    lab1MM(a, b)
    bruteForceTimes.append(time.time() - startTime)


# plot results
plt.plot(matrixSizes, recursiveTimes)
plt.plot(matrixSizes, bruteForceTimes)
plt.xlabel("n")
plt.ylabel("time (s)")
plt.title("Recursive Matrix Multiplication vs Brute Force Method")
plt.legend(['Recursive Measured Times', 'Brute Force Measured Times'])
plt.show()