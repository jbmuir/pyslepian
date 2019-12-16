import numpy as np
from scipy import linalg
from scipy import special

#this one is, I think, more readable
# def circ_kernel_2d(x1, x2, K):
#     diff = x2-x1
#     mag = np.sqrt(np.dot(diff,diff))
#     if mag < 1e-12: #x1, x2 are the same point within error
#         return K * K / (4. * np.pi)
#     else:
#         return K * special.j1(K * mag) / (2. * np.pi * mag)

#this one is a straight copy paste of Frederik's code into python. More confusing to read, but
#Using meshgrid is ~ 100 times faster than the list comprehension though so we will use this
def circ_kernel_2d(xx, xxp, yy, yyp, K):
    mag = np.sqrt((xx-xxp)**2 + (yy-yyp)**2)
    with np.errstate(divide = 'ignore'):
        kernel = K * special.j1(K * mag) / (2. * np.pi * mag)
        kernel[np.logical_not(np.isfinite(kernel))] = K * K / (4. * np.pi)
        return kernel


def get_circ_K_2d(shannon, area):
    return np.sqrt(4. * np.pi * shannon / area)

def get_circ_shannon_2d(K, area):
    return K * K * area / (4. * np.pi)

# def circ_kernel_matrix_2d(quad_rules, K):
#     abcissa = quad_rules["abcissa"]
#     return np.array([[circ_kernel_2d(i, j, K) for j in abcissa] for i in abcissa])

def circ_kernel_matrix_2d(quad_rules, K):
    abcissa = quad_rules["abcissa"]
    x = abcissa[:,0]
    y = abcissa[:,1]
    xx, xxp = np.meshgrid(x,x,indexing='xy')
    yy, yyp = np.meshgrid(y,y,indexing='xy')
    return circ_kernel_2d(xx, xxp, yy, yyp, K)

def circ_eig_problem_2d(quad_rules, K):
    D = circ_kernel_matrix_2d(quad_rules, K)
    Wt = np.diag(np.sqrt(quad_rules["weights"]))#using the square root makes the problem symmetric for better conditioning
    w,Vt = linalg.eigh(np.dot(Wt,np.dot(D,Wt)))
    idx = w.argsort()[::-1]#sort by highest to lowest eigenvalue
    w = w[idx]
    Vt = Vt[:,idx]
    V = linalg.solve(Wt,Vt)
    return (w, V)#this solve rescales the problem, after we had previously condioned it by the square root of the weights


def compute_slepians_at_points(quad_rules, shannon, xp, yp):
    K = get_circ_K_2d(shannon,quad_rules["area"])
    w, V = circ_eig_problem_2d(quad_rules, K)
    w = w[:2*shannon]
    V = V[:,:2*shannon]
    abcissa = quad_rules["abcissa"]
    x = abcissa[:,0]
    y = abcissa[:,1]
    xx, xxp = np.meshgrid(x, xp, indexing = 'xy')
    yy, yyp = np.meshgrid(y, yp, indexing = 'xy')
    D = circ_kernel_2d(xx, xxp, yy, yyp, K)
    W = np.diag(quad_rules["weights"])
    if np.max(w) > 1:
        raise ValueError("Maximum concentration value is greater than 1")
    return (w, np.dot(D,np.dot(W,np.dot(V,np.diag([1./l for l in w])))))
