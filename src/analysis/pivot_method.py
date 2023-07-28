import numpy as np
import scipy as sp
import scipy.linalg
from src.analysis.jk import jackKnifeCov, jackKnifeList

def pivot_matrix(cmat, t0, td):
    avg = np.mean(cmat, axis=0)

    a = sp.linalg.fractional_matrix_power(avg[t0], -0.5)

    gtilde = a @ avg[td] @ a

    evecs = np.linalg.eig(gtilde)[1]

    return evecs


def pivot_matrix_rotate(cmat, t0, pivotMatrix):
    avg = np.mean(cmat, axis=0)

    a = sp.linalg.fractional_matrix_power(avg[t0], -0.5)
    product = lambda t : pivotMatrix.conjugate().transpose() @ a @ avg[t] @ a @ pivotMatrix

    return [product(t) for t in range(len(avg))]


def get_pivoted_corr(corrMatrix, t0, td):
    pm=pivot_matrix(corrMatrix,t0,td)

    pmr = lambda mat: pivot_matrix_rotate(mat, t0, pm)

    pivotedCorr = jackKnifeCov(pmr, corrMatrix)

    return pivotedCorr

def get_pivoted_corr_samples(corrMatrix, t0, td):
    pm=pivot_matrix(corrMatrix,t0,td)

    pmr = lambda mat: pivot_matrix_rotate(mat, t0, pm)

    return jackKnifeList(pmr, corrMatrix)