from scipy import interpolate
import scipy
import numpy

def get_spline(data):
    y,binEdges = numpy.histogram(data,50)
    y_error = numpy.sqrt(y)
    y_error[y_error == 0] = 1 
    w = 1/y_error
    x = 0.5*(binEdges[1:]+binEdges[:-1])
    return scipy.interpolate.splrep(x, y, w, s=0)

def reweight_prompt(cluster_Inv_E,norm):
    x = cluster_Inv_E**(-2)
    tck = get_spline(x)
    r = scipy.interpolate.splev(x, tck, der=0)
    return norm / r 

def reweight_nonprompt(cluster_Inv_E,norm):
    x = cluster_Inv_E**(-2)
    tck = get_spline(x)
    r = scipy.interpolate.splev(x, tck, der=0)
    # r *= reweight_prompt(cluster_Inv_E,norm)
    return r
