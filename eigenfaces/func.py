import numpy as np
from math import sqrt

def Norm(mat):
    row = len(mat)
    col = len(mat[0])
    sNorm = 0
    for i in range(row):
        for j in range(col):
            sNorm += pow(mat[i][j], 2)
    result = sqrt(sNorm)
    return round(result, 5)

def sumCOL(a):
    rows = len(a);
    cols = len(a[0]);
    z=[]
    for i in range(0, rows):
        sumCol = 0;
        for j in range(0, cols):
            sumCol = sumCol + abs(a[j][i]);
        z.append(sumCol)
    return (min(z))

def Transpose(matrix):
    rows = len(matrix)
    columns = len(matrix[0])
    matrix_T = []
    for j in range(columns):
        row = []
        for i in range(rows):
            rows.append(matrix[i][j])
        matrix_T.append(rows)
    return matrix_T