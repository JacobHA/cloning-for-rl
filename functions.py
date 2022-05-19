
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 16:09:58 2021

@author: jacob
"""



import numpy as np
import random
from scipy.sparse.linalg import eigs, ArpackError
from scipy.sparse import csr_matrix
 
def power_iteration(A, n_eigs=1, max_it=10000, start_vec=None, debug=False, tolerance=5e-12):
    A = csr_matrix(A)
    n = A.shape[0]
    eigenvals = []
    eigenvecs = np.empty((n, 0))
    start_vec = np.ones(n) if start_vec is None else start_vec

    for dominant_i in range(1):
        b_k, err, k = start_vec.reshape((n, 1)), 1., 0
        while err > tolerance and k < max_it:
            # calculate the matrix-by-vector product Ab
            b_k1 = A.dot(b_k)

            # calculate the norm
            b_k1_norm = np.linalg.norm(b_k1)

            # re normalize the vector
            b_k1 = b_k1 / b_k1_norm

            err = np.abs(b_k1 - b_k).max()
            b_k = b_k1
            k += 1
        if k == max_it:
            print(f"Warning: failed to converge after {k} iterations. err: {err}")
        elif debug:
            print(f"Converged after {k} iterations")

        # compute the eigenvalue
        l = (b_k.T.dot(A.dot(b_k)) / b_k.T.dot(b_k))[0, 0]
        eigenvals.append(l)
        eigenvecs = np.concatenate([eigenvecs, b_k], axis=1)

    if debug:
        return np.array(eigenvals), eigenvecs, k
    else:
        return np.array(eigenvals), eigenvecs


def largest_eigs_dense(A, n_eigs=1):

    if 'toarray' in dir(A):
        # need to be a dense matrix
        A = A.toarray()

    eigvals, eigvecs = np.linalg.eig(A)
    # try:
    #     eigvals, eigvecs = process_complex_eigs(eigvals, eigvecs)
    # except ValueError:
    #     raise

    return eigvals[:n_eigs], eigvecs[:, :n_eigs]


def largest_eigs_sparse(A, n_eigs=1, max_it=int(1e4), start_vec=None, sigma=1.):
    
    n = A.shape[0]
    start_vec = np.ones(n) if start_vec is None else start_vec
    eigvals, eigvecs = eigs(A, n_eigs, v0=start_vec, which='LM', sigma=sigma, OPpart='r')
    # try:
    #     eigvals, eigvecs = process_complex_eigs(eigvals, eigvecs)
    # except ValueError:
    #     raise

    return eigvals[:n_eigs], eigvecs[:, :n_eigs]


def process_complex_eigs(eigvals, eigvecs):
    
    # sort eigenvalues from largest to smallest
    order = np.argsort(-np.abs(eigvals))
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    if eigvals[0].imag != 0:
        print("W: complex dominant eigenvalue")
        # we require that the dominant eigenvalue is a real number

    # retain only real eigenvalues and corresponding eigenvectors
    realevs = eigvals.imag == 0
    eigvals, eigvecs = eigvals[realevs], eigvecs[:, realevs]

    posievs = eigvals.real > 0
    eigvals, eigvecs = eigvals[posievs], eigvecs[:, posievs]

    return eigvals.real, eigvecs.real


def largest_eigs(A, n_eigs=1, max_it=int(1e4), start_vec=None, method=None,
                 debug=False, sigma=1., tolerance=5e-12):
    nrow, ncol = A.shape
    assert nrow == ncol, "A must be square matrix"

    if method is None:
        try:
            eigvals, eigvecs = largest_eigs_dense(A, n_eigs)
        except Exception as ex:
            print(ex)
            try:
                eigvals, eigvecs = largest_eigs_sparse(A, n_eigs, max_it, start_vec, sigma)
            except Exception as ex:
                print(ex)
                print("W: Could not find eigenvectors. Trying power iteration method")
                eigvals, eigvecs = power_iteration(A, n_eigs, max_it, start_vec, debug, tolerance=tolerance)
    elif method == 'dense':
        eigvals, eigvecs = largest_eigs_dense(A, n_eigs)
    elif method == 'sparse':
        eigvals, eigvecs = largest_eigs_sparse(A, n_eigs, max_it, start_vec, sigma)
    elif method == 'power':
        eigvals, eigvecs = power_iteration(A, n_eigs=n_eigs, max_it=max_it, start_vec=start_vec, debug=debug, tolerance=tolerance)
    else:
        raise ValueError("invalid method value. Please choose one of ['dense', 'sparse', 'power']")

    return eigvals, eigvecs



def row_sums(M):
    return M.sum(axis=1)

def col_sums(M):
    return M.sum(axis=0)

def two_row_sums(M):
    # defintion provided by doi: 10.3934/math.2020047
    # notation: M_i(A)
    row_vec = row_sums(M)
    return M @ row_vec

def avg_two_row_sums(M):
    # defintion provided by doi: 10.3934/math.2020047
    # notation: m_i(A)
    return two_row_sums(M) / row_sums(M)

def w_i_avg_of_avg(M):
    # defintion provided by doi: 10.3934/math.2020047
    # notation: w_i(A), eqn. 1.1
    m_j = avg_two_row_sums(M)
    numerator = M @ m_j
    
    return numerator / m_j # triple check this



def auxiliary_quantities(M):
    # mask out diagonal
    mask = np.ones(M.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    max_nondiag = M[mask].max()
    min_nondiag = M[mask].min()

    diagonal = np.diag(M)
    max_diag, min_diag = diagonal.max(), diagonal.min()
    
    return max_nondiag, min_nondiag, max_diag, min_diag

def PSI_L(M, verbose = False):
    # THEOREM 4
    m_j = avg_two_row_sums(M)
    w_j = w_i_avg_of_avg(M)
    w_j = sorted(w_j, reverse=True) # needed descending per eqn. 2.1
    N__, __, M__, __ = auxiliary_quantities(M)
    assert min(m_j) != 0, "Non-applicable when zero min."
    ratios = np.array([m / m_j for m in m_j])  ## DOUBLE CHECK THIS
    mask = np.ones(ratios.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    b_m = ratios[mask].max()
    res_array = []
    for l in range(len(w_j)):
        w_l = w_j[l]
        sum_term = sum( w_j[:l-1] - w_l)
        if l == 0:
            sum_term = 0
        DELTA_L = (w_l - M__ + N__ * b_m) ** 2 + 4*N__*b_m * sum_term 
        res_array.append(0.5*(w_l + M__ - N__ * b_m + np.sqrt(DELTA_L)))
    
    if verbose: return res_array
    else: return np.array(res_array).min()

def psi_n(M):
    m_j = avg_two_row_sums(M)
    w_j = w_i_avg_of_avg(M)
    w_j = sorted(w_j, reverse=True) # needed descending per eqn. 2.1
    __, T__, __, S__ = auxiliary_quantities(M)
    assert min(m_j) != 0, "Non-applicable when zero min."
    c_m = np.nanmin(m_j) / np.nanmax(m_j)  ## DOUBLE CHECK THIS
    DELTA_N = []
    for l in range(len(w_j)):
        w_l = w_j[l]
        sum_term = sum( w_j[:l-1] - w_l)
        if l == 0:
            sum_term = 0
        DELTA_N.append( (w_l - S__ + T__ * c_m) ** 2 + 4*T__ * c_m * sum_term )
    
    return 0.5*(w_j + S__ - T__ * c_m + np.sqrt(DELTA_N))

def PHI_L_HAT(M, verbose=False):
    # THEOREM 6

    r_i = row_sums(M)
    r_i = sorted(r_i, reverse=True) # needed descending per eqn. 2.1
    N__, __, M__, __ = auxiliary_quantities(M)
    
    res_array = []
    for n in range(len(r_i)):
        r_n = r_i[n]
        sum_term = sum(r_i[:n-1] - r_n)
        if n == 0:
            sum_term = 0

        res_array.append( (r_n + M__ - N__)/2 + np.sqrt( ((r_n - M__ + N__)/2 )**2  + N__ * sum_term) )
    if verbose: return res_array
    else: return np.array(res_array).min() # i.e. the best upper bound
    
def PHI_L_TILDE(M, verbose = False):
    N__, __, M__, __ = auxiliary_quantities(M)

    m_j = avg_two_row_sums(M)
    m_j = sorted(m_j, reverse=True)
    r_j = row_sums(M)
    
    ratios = np.array([r / r_j for r in r_j])  ## DOUBLE CHECK THIS
    mask = np.ones(ratios.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    b = ratios[mask].max()
    
    res_array=[]
    for l in range(len(m_j)):
        m_l = m_j[l]
        sum_term = sum(m_j[:l-1] - m_l)
        if l == 0:
            sum_term = 0
        res_array.append( (m_l + M__ - N__*b)/2 + np.sqrt( ((m_l - M__ + N__*b)/2)**2 + N__*b*sum_term))
    
    if verbose: return res_array
    else: return np.array(res_array).min() # i.e. the best upper bound
    
    
def phi_n_hat(M):
    r_i = row_sums(M)
    r_i = sorted(r_i, reverse=True)
    __, T__, __, S__ = auxiliary_quantities(M)
    for n in range(len(r_i)):
        r_n = r_i[n]
        sum_term = sum(r_i[:n-1] - r_n)
        if n == 0:
            sum_term = 0
    return (r_i + S__ - T__)/2 + np.sqrt( ((r_i - S__ + T__)/2) **2 + T__ * sum_term)
    
def phi_n_tilde(M):
    m_j = avg_two_row_sums(M)
    r_m = row_sums(M)
    __, T__, __, S__ = auxiliary_quantities(M)
    m_j = sorted(m_j, reverse=True)
    for n in range(len(m_j)):
        m_n = m_j[n]
        sum_term = sum(m_j[:n-1] - m_n)
        if n == 0:
            sum_term = 0
    c = min(r_m)/max(r_m)
    return (m_j + S__ - T__*c)/2 + np.sqrt( ((m_j - S__ + T__*c)/2 )**2 +T__*c * sum_term)


def Adam_bounds(M):
    n = M.shape[0] - 1
    ub = min(PSI_L(M), PHI_L_HAT(M), PHI_L_TILDE(M))
    lb = max(psi_n(M)[n], phi_n_hat(M)[n], phi_n_tilde(M)[n])
    
    return lb, ub

def Melman_bounds(M):
    mask = ~np.identity(M.shape[0],dtype=bool)
    
    R = row_sums(M)
    Rp = R - np.diag(M)
    Rpp = Rp - M
    bounds = np.zeros_like(M)
    for i,Mii in enumerate(np.diag(M)):
        
        for j,Mjj in enumerate(np.diag(M)):
            if i != j:
                bounds[i][j] = Mii + Mjj + Rpp[i][j] + np.sqrt( (Mii - Mjj + Rpp[i][j])**2 + 4*M[i][j]*Rp[j])
    np.fill_diagonal(bounds, -np.inf)
    LB = 0.5*bounds.max(axis=0).min()
    np.fill_diagonal(bounds, np.inf)
    UB = 0.5*bounds.min(axis=0).max()
    return bounds #(LB, UB)

def avg_three_row_sum(A):
    M_i = two_row_sums(A)
    return A @ M_i / row_sums(A)

def PHI_L(A):
    n,_ = A.shape
    N, min_nondiag, M, min_diag = auxiliary_quantities(A)
    s_i = avg_three_row_sum(A)
    s_l = sorted(s_i, reverse=True)
    r_j = row_sums(A)
    ratios = np.array([r / r_j for r in r_j])  ## DOUBLE CHECK THIS
    mask = np.ones(ratios.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    b = ratios[mask].max()
    Theta =  M**2 + N**2 * (n-1) - 2*M*N*b - (n-2)*N**2 * b
    PHI_LL = [(s_l[l] + Theta +np.sqrt( (s_l[l] - Theta)**2 +4*(2*M*N + (n-2)*N**2)*b * sum([sk - s_l[l] for sk in s_l])))/2 for l in range(len(s_l))] 
    return(np.sqrt(PHI_LL))
#%%
def twister(A,S,beta=1,new_distro=False):
    N=A*S
    if not new_distro:
        T = np.diag( np.exp(-beta * np.random.rand(N)) )  # twisting matrix
    if new_distro:
        T = np.exp(-beta*np.diag(np.random.rand(N))) - (np.ones((N,N)) - np.identity(N))
    return T

def unif_prior(A,S):
    return np.ones((S, A)) / A

def det_dynamics(A,S):
    N=A*S
    P = np.zeros((N,N))
    choices = np.random.choice(P.size, N**2-N, replace=False) # perhaps not S?
    P.ravel()[choices] = 1
    return P*unif_prior(A,S).reshape((S*A))
    

def dynamic(A,S):
    N=A*S
    P=[]
    for s in range(S):
        row_s = np.random.rand(N)   # random dynamics
        for ac in range(A):
            P.append(row_s)         # uniform prior
    P=np.array(P)
    P /= P.sum(axis=0,keepdims=1)      # Normalize the column sums
    
    return P


def twisted(A,S):
    P = dynamic(A,S)
    T = twister(A,S)
    D = P@T        # twisted dynamics
    return D


def upper(M):
    # upper bound on eigen
    return np.sqrt(np.max(M.sum(axis=1))/np.min(M.sum(axis=1)))

def lower(M):
    # lower bound on eigen
    for col in M.T:
        a=np.max(col)/np.min(col)
    return a

def bounds(M):
    return min(M.sum(axis=1)), max(M.sum(axis=1))

def DOMeigs(M):
  
    vals,vecs = np.linalg.eig(M) # Eigen
    vecs = (vecs.T).real # format and realize
    dict_of_eig = dict(zip(vals,vecs))
    dom_eig_val=sorted(dict_of_eig)[-1]
    dom_eig_vec=dict_of_eig[dom_eig_val]
    
    assert dom_eig_val == dom_eig_val.real, ("Complex Perron root!")
    return dom_eig_val.real, dom_eig_vec


def l_jorma(M):
    # using their notations
    n = M.shape[0]
    a = np.trace(M)
    b = np.trace(M**2)
    
    Co = 1/ (n**2 - n) 
    B = b - a**2 / n
    Co1 = (n-1)/n
    
    return a/n + np.sqrt(Co*B)

def u_jorma(M):
    # using their notations
    n = M.shape[0]
    a = np.trace(M)
    b = np.trace(M**2)

    Co = (n-1) / n
    B = b - a**2 / n
    
    return a/n + np.sqrt(Co*B)

def Perron_info(M):
    # returns rho, u, v
    rho,v = DOMeigs(M)
    rho,u = DOMeigs(M.T)
    v /= sum(v)
    u /= u.dot(v)
    return rho, u, v

def free_energy(M,beta=1):
    exptheta,_ = DOMeigs(M)
    
    return -1/beta * np.log(exptheta)

def avg_energy(P,T, beta=1):
    # N = 1
    M = P@T
    t = -np.log(np.diag(T))
    energies = 1/beta * t
    rho,v = DOMeigs(M)
    rho,u = DOMeigs(M.T)
    v /= sum(v)
    u /= u.dot(v)
    steady_st = v*u
    
    return steady_st.dot(energies)

def avg_energy_low_bound(P,T1,T2,beta=1):
    r1,u1,v1 = Perron_info(P@T1)
    r2,u2,v2 = Perron_info(P@T2)
    # T1 = e^-beta * energy_vector
    eng1=np.diag(T1)
    eng2=np.diag(T2)
    avg_eng=-np.log(eng1*eng2)/2
    stead1=u1*v1
    stead2=u2*v2
    quasi_distro = np.sqrt(stead1*stead2)
    assert len(quasi_distro) == len(avg_eng), "Incorrect input dimensions"
    
    return quasi_distro.dot(avg_eng)

def avg_entropy(P,T, beta=1):
    E = avg_energy(P, T, beta=beta)
    F = free_energy(P@T, beta=beta)
    # F = E - T*S
    return -(F-E)*beta

def policy(D,S,A):
    _,u,_ = Perron_info(D)
    u = u.reshape(S,A)
    pi = u / u.sum(axis=1, keepdims=1)
    return pi
    
def steady_distro(D):
    _,u,v = Perron_info(D)
    return u*v

def det_entropy(D,S,A):
    N=S*A
    pi = policy(D,S,A)
    _,u,v = Perron_info(D)
    def xlogx(x):
        if type(x) != np.ndarray:
            if x > 0:
                return x*np.log(x)
            if x == 0: 
                return 0
            else:
                print(x)

                raise ValueError
        elif type(x) == np.ndarray:
            return [xlogx(i) for i in x]
            
    return ((xlogx(pi)+pi*np.log(A)).reshape(N)).dot(u*v) # possible error??

def uniform_v(S,A):
    return np.ones(S*A)/(S*A)

# def amended_v(S,A,s_reduce, v0=uniform_v(S, A)):
#     v0[s_reduce*A:s_reduce*A+A] = 0

#     return v0/sum(v0) # Don't forget to renormalize!

def Q_func(D,N,beta=1):
    # N=1
    # # twisted_matrix=twisted_matrix.T ### Need to transpose before using
    # rho, u, v = Perron_info(D) # 
    
    # return np.log((rho**N)*u).reshape(S,A)  # N = 1 step
    if beta != 1:
        P,T = PT_decomp(D)
        Tbeta = T**beta
        D=P@Tbeta
    DN=np.linalg.matrix_power(D,N)
    return np.log(col_sums(DN))


def gen_perturbed(P,T, J, perturbation):    
    n=P.shape[0]
    pert = np.identity(n)
    pert[J,J] *= np.exp(-perturbation)
    T_pert = T @ pert
    return P@T_pert
    
def d_v(D,D_pert,S,A):
    # From page 261 of sensitivity analysis book
    LAM = D-D_pert
    rho,u,v = Perron_info(D)
    AAA = np.linalg.matrix_power(rho*np.identity(S*A) - D + v * np.ones(S*A).T @ D, -1) 
    BBB = np.kron(v.T , np.identity(S*A)-v * np.ones(S*A).T)
    return (AAA @ BBB) @ LAM.reshape(S*A * S*A)

def d_u(D,D_pert,S,A):
    # From pg 261 of sensitivity analysis book
    e1 = np.zeros(S*A)
    e1[0]=1
    LAM = D-D_pert
    rho,u,v = Perron_info(D)
    AAA = np.linalg.matrix_power(rho*np.identity(S*A) - D.T + u * e1.T @ D.T, -1) 
    BBB = np.kron(np.identity(S*A)- u * e1.T, u.T)
    return (AAA @ BBB) @ LAM.reshape(S*A * S*A)

    
#%%
    
# From the book: A Survey of Matrix Theory and Matrix Inequalities
# p 307 3.1.5 eq 10
    
def bounds_Marc_Minc(M):
    N = M.shape[0]
    col_sums = (M.T).sum(axis=1)
    row_sums = (M).sum(axis=1)
    # m = M.min()
    alpha = np.diag(M).min()
    M_aug = M - np.diag(np.diag(M))
    k=M_aug.min()

    
    delta_row = np.array([i/row_sums for i in row_sums])
    delta_col = np.array([i/col_sums for i in col_sums])
    delta_row = max(delta_row[delta_row < 1])
    delta_col = max(delta_col[delta_col < 1])
    
    sigma_row = (1/N) * sum(row_sums)
    sigma_col = (1/N) * sum(col_sums)
    
    epsilon_row = (k/(max(row_sums) - alpha))**(N - 1)
    epsilon_col = (k/(max(col_sums) - alpha))**(N - 1)
    
    
    lower_r_result = min(row_sums)+epsilon_row*(sigma_row-min(row_sums))
    lower_c_result = min(col_sums)+epsilon_col*(sigma_col-min(col_sums))
    
    upper_r_result = max(row_sums) - epsilon_row*(max(row_sums) - sigma_row)
    upper_c_result = max(col_sums) - epsilon_col*(max(col_sums) - sigma_col)
    
    return lower_r_result, upper_r_result

def e_(j,N):
    z=np.zeros(N)
    z[j] = 1
    return z

#%%
    
# copied https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_EigenProblem1.html
def __find_p(x):
    return np.argwhere(np.isclose(np.abs(x), np.linalg.norm(x, np.inf))).min()

def __iterate(A, x, p):
    y = np.dot(A, x)       
    u = y[p]      
    p = __find_p(y)     
    error = np.linalg.norm(x - y / y[p],  np.inf)
    x = y / y[p]
    return (error, p, u, x) 

def power_method(A, tolerance=1e-10, max_iterations=10000, left_or_right_str='left'):
    
    n = A.shape[0]
    if left_or_right_str == 'left':
        x = np.ones(n)
    if left_or_right_str == 'right':
        x = np.ones(n)/n # maxent guess
    
    p = __find_p(x)
    error = 1
    
    x = x / x[p]
    
    for steps in range(max_iterations):
        
        if error < tolerance:
            break
            
        error, p, u, x = __iterate(A, x, p)
        
    return u, x

def inverse_power_method(A, given_eigval = 0.5, given_vec = None, tolerance=1e-10, max_iterations=10000):
    
    n = A.shape[0]
    if given_vec is None: x = np.ones(n)
    else: x = given_vec
    I = np.eye(n)
    
    q = given_eigval #np.average(Adam_bounds(A)) #np.dot(x, np.dot(A, x)) / np.dot(x, x)
    
    p = __find_p(x)
    
    error = 1
    
    x = x / x[p]
    
    for steps in range(max_iterations):
        
        if error < tolerance:
            break
            
        y = np.linalg.solve(A - q * I, x)       
        u = y[p]      
        p = __find_p(y)     
        error = np.linalg.norm(x - y / y[p],  np.inf)
        x = y / y[p]
        u = 1. / u + q 
        
    return u, x



def custom_power_it(A, tolerance=1e-10, max_iterations=10_000):
    n = A.shape[0]
    x = np.ones(n)
    I = np.eye(n)
    
    # Single iteration from MaxEnt guess:
    v = A.sum(axis=1)
    v /= sum(v)
    u = A.sum(axis=0)
    u /= u.dot(v)
    rho = u @ A @ v
    # print(rho)
    rho, u = inverse_power_method(A.T, given_eigval=rho, given_vec=u, max_iterations=1)
    u /= u.dot(v)
    # print(rho)
    rho, v = inverse_power_method(A, given_eigval=rho, given_vec=v, max_iterations=1)
    v /= sum(v)
    u/=u.dot(v)
    return rho,u,v
        
    

def pi_diverg(P,T1,T2,S,A,alpha1=0.5,alpha2=0.5,beta=1):
    # JJ HUNT MAXENT POLICY DIVERGENCE CORRECTION
    # S,A=p1.shape
    D1,D2=P@T1,P@T2
    p1,p2=policy(D1,S,A),policy(D2,S,A)

    product = p1**alpha1 * p2**alpha2
    divergence = product.sum(axis=1)
    rewards = np.array([[div_state]*A for div_state in divergence]).flatten()
    twist = np.diag(np.exp(-beta*rewards))
    #r,u,v=Perron_info(P@twist)
    print(rewards)
    return Q_func(P@twist,S,A)

#%%
# PLOTTING
   
def subplots_across_actions(list_of_arrays, list_of_labels, title='', transpose=True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(title, fontsize=16)
    index=0 
    for row in ax:
        for col in row:
            col.set_title(f'Action a={index}')
            for array,label_name in zip(list_of_arrays,list_of_labels):
                if transpose: array = array.T
                col.plot(array[index], label = label_name)
        index += 1
    plt.show()
    plt.legend()

#%%
    
# 6/14/2021 NEW METHOD

def MDP_dynamic(A,S, det=False):
    P=np.zeros((S,A*S))
    if not det:
        for s in range(S):
            row_s = np.random.rand(S*A)   # random dynamics
            P[s] = row_s    
            P /= P.sum(axis=0,keepdims=1)      # Normalize the column sums

    if det:
        for s in range(S):
            sp_transition = np.random.choice(range(S*A))
            row_s=e_(sp_transition,S*A)
            P[s] = row_s
    return P
    
    
def full_dynamic(P, det=False, rando_prior=None):
    S,SA=P.shape
    A=int(SA/S)
    prior_poli = unif_prior(A,S)

    if rando_prior is not None:
        prior_poli = rando_prior #random_prior(A, S)
    
    P_full = np.zeros((S*A,S*A))
    # P = MDP_dynamic(A,S)
    for act in range(A):
        for state in range(S):
            P_full[state*A+act] = P[state] * prior_poli.flatten()
        
    if not det:
        P_full /= P_full.sum(axis=0,keepdims=1)      # Normalize the column sums
    
    return P_full

def random_prior(A,S):
    pi=np.random.uniform(size=(S,A))
    
    return pi / pi.sum(axis=1,keepdims=1) 

def little_q(MDP,T, rando_prior=False):
    S,SA=MDP.shape
    A=int(SA/S)

    prior_poli = unif_prior(A, S)
    if rando_prior:
        prior_poli = random_prior(A,S)
    P=MDP
    expd_rews = T.diagonal()
    # How is T organized? (s0,a0), (s0,a1),...?
    lil_q = np.zeros((S,S))
    
    for sprime,sprime_row in enumerate(P):
        # sum over actions p(s'|s,a)
        for s in range(S):
            # \Chi_{s' s}
            lil_q[sprime][s] = \
                (prior_poli[s] * sprime_row[s*A:s*A+A] * expd_rews[s*A:s*A+A]) \
                    .sum()
    
    return lil_q

        
def v_q(vq,A):
    # ONLY WORKING W UNIF PRIOR
    vq_sa = np.array([[v_s]*A for v_s in vq]).flatten()
    
    return vq_sa / sum(vq_sa)

def u_q(MDP,T):
    S,SA=MDP.shape
    A=int(SA/S)
    q = little_q(MDP,T)
    
    u = np.zeros(S*A)
    r,chi,vq_s=Perron_info(q)
    v = v_q(vq_s,A)
    expd_rews = np.diagonal(T)
    for sa_pair in range(S*A):
        u[sa_pair] = expd_rews[sa_pair]* ((MDP.T)[sa_pair].dot(chi))
        
    return u / u.dot(v)

def uv_one_step_q(MDP,T):
    S,SA=MDP.shape
    A=int(SA/S)
    q = little_q(MDP,T)
    u = np.zeros(S*A)
    v_one = row_sums(q)
    chi_one = col_sums(q)
    
    v = v_q(v_one,A)
    v /= sum(v)
    
    expd_rews = np.diagonal(T)
    for sa_pair in range(S*A):
        u[sa_pair] = expd_rews[sa_pair]* ((MDP.T)[sa_pair].dot(chi_one))
        
    u /= u.dot(v)
    
    return u,v

def driven_matrix(D, r=None, u=None, v=None):
    Ddriv = np.zeros(shape=D.shape)
    if not r and not u and not v: # i.e. none are givens 
        print('calculated eigenspace')
        r,u,v=Perron_info(D)
        # else continue with given values:
    for j, rowj in enumerate(Ddriv):
        for i, coli in enumerate(rowj):
            Ddriv[j][i] = D[j][i] * u[j] / (r * u[i])
                         
            
    # we must normalize the driven matrix!
    Ddriv /= Ddriv.sum(axis=0,keepdims=1) 
            
            
    return Ddriv

def rho_2(D,driven, r=None, u=None, v=None):
    rho_2 = np.zeros(shape=driven.shape)
    if not r and not u and not v:
        print('calculated eigenspace')
        _,_,uv = Perron_info(driven)
    else:
        uv = Perron_info(driven)[2]
        driven = driven_matrix(D,r=r,u=u,v=v)

    for j, rowj in enumerate(rho_2):
        for i, rho2_ji in enumerate(rowj):
            rho_2[j][i] = driven[j][i] * uv[i]
    
    return rho_2

def MDP_rate_func(P,T,driven, r=None, u=None, v=None):
    if not r and not u and not v:
        print('calculated eigvals in mdp_rate_func')
        _,_,uv = Perron_info(driven)
        
    D=P@T
    
    rho2 = rho_2(D,driven,r=r,u=u,v=v)
    integral_A = rho2.sum(axis=0)
    integral_B = rho2.sum(axis=1)
    if np.allclose(integral_A, integral_B):
        return (rho2 * np.log(rho2 / (integral_A*P) ) ).sum()
    else:
        return np.inf

def inf_arg(P,T,r=None, u=None, v=None,beta=1):
    D=P@T
    rew_vec = np.log(T.diagonal())
    if not r and not u and not v:
        print('calculated eigenspace in inf_arg')
        r,u,v=Perron_info(D)
        uv = u*v
        driv = driven_matrix(P@T, r=r, u=u, v=v)

    else:
        driv = driven_matrix(P@T, r=r, u=u, v=v)
        uv = rho_2(P@T,driv,r=r,u=u,v=v).sum(axis=1)

    return beta * (uv.dot(rew_vec)) - MDP_rate_func(P,T,driv,r=r,u=u,v=v)

def uv_one_step_D(D):
    v = row_sums(D)
    u = col_sums(D)
    
    v /= sum(v)
    u /= u.dot(v)
    
    return u,v

def kl_div(opt_distro, est_distro):
    return (opt_distro * np.log(opt_distro / est_distro)).sum()

def policy_from_u(u,S,A):
    u = u.reshape(S,A)
    pi = u / u.sum(axis=1, keepdims=1)
    return pi
    
def q_from_urho(u,rho,A,S,N=1):
    return np.log(rho**N * u).reshape(S,A)

def action_avg_T(T,A):
    # Uniform prior
    diag=np.diagonal(T)
    S=int(diag.shape[0]/A)
    pi=unif_prior(A,S)
    prior_poli=pi
    
    avg_diag = (diag.reshape(S,A) * prior_poli).sum(axis=1)
    
    return avg_diag
    
def avg_pooling(A, acts, custom_func=np.mean):
    # assuming uniform prior. otherwise must build in function
    # to call rather than np.mean below
    import skimage.measure
    return skimage.measure.block_reduce(A,(acts,acts), func=custom_func)*acts
    # /acts above because of acts repetition? Not 100% sure.


def max_pooling(A, acts):
    # assuming uniform prior. otherwise must build in function
    # to call rather than np.mean below
    import skimage.measure
    return skimage.measure.block_reduce(A,(acts,acts), func=np.max)*acts
    # /acts above because of acts repetition? Not 100% sure.
def gen_dotprod(weights, arrays):
    return sum([weights[i] * arrays[i] for i in range(len(weights))])


def softmax(weights, arrays):
    return np.log(gen_dotprod(weights, np.exp(arrays)))
        
def alpha_i(A):
    N = A.shape[0]
    alphais=[]
    for i,row_i in enumerate(A):
        mask = 1-np.array([int(k) for k in e_(i,N)])
        alphais.append((row_i*mask).max())
    return np.array(alphais)
    

def cheng_rao_bounds(A,B):
    aii = np.diagonal(A)
    bii = np.diagonal(B)
    rA, _, _ = Perron_info(A)
    rB, _, _ = Perron_info(B)
    
    return max(aii*bii + alpha_i(A)*rB - alpha_i(A)*bii)
    

def vector_norm(vec, p):
    return (vec**p).sum() ** (1/p)

class Spectrum():
    def __init__(self, matrix, beta=1):
        self.matrix = matrix
        self.Ndim, _ = self.matrix.shape
        self.beta = beta
        # _, self.reigs = largest_eigs(self.matrix, n_eigs=self.Ndim)
        # self.eigvals, self.leigs = largest_eigs(self.matrix.T, n_eigs=self.Ndim)
        # self.reigs = self.reigsT.T
        # self.leigs = self.leigsT.T # un: un-normalized
 
        
        # self.reigs[0] /= np.real_if_close(self.reigs[0].sum())
        # self.leigs[0] /= np.real_if_close(self.reigs[0].dot(self.leigs[0])) # prevents divide by zero blowups from tiny imaginary parts

        # for v in self.reigs:
        #     v /= np.real_if_close(v.sum())
        # for u,v in zip(self.leigs, self.reigs):
        #     u /= np.real_if_close(sum(u*v))
    
        ## Helper function to get S,A out of repeating v values:
        self.A = 0
        for i in range(len(self.matrix)):
            if np.allclose(self.matrix[0], self.matrix[i]):
                self.A +=1
            else:
                break
        self.A = i
        self.S = self.Ndim // self.A
            
    def EVsumEV(self, N_steps = 1):
        res = np.zeros(self.matrix.shape[0], dtype = 'complex128')
        for i in range(len(self.eigvals)):
            res += ( self.eigvals[i] ** N_steps) * self.leigs[i]
            
        return np.real_if_close(res)
    
    def nth_Term(self, n=0):
        return self.eigvals[n] * self.leigs[n]
    
    def Qfunc(self, N_steps=1):
        # return np.real_if_close(  (1/self.beta) * np.log(self.EVsumEV(N_steps))  )
        return np.log( np.linalg.matrix_power(self.matrix, N_steps).sum(axis=1)) # sum over j
    def Vfunc(self, N_steps=1):
        return np.real_if_close( self.Qfunc(N_steps).reshape(self.S, self.A).sum(axis=1))
    


def printf(label, i, out_of = 0, add_1=True):
    if add_1:
        index = i + 1
    else:
        index = i
    if out_of != 0:
        print(f'\r{label}: {index}/{out_of}', flush=True, end='')
    else:
        print(f'\r{label}: {index}', flush=True, end='')
    if out_of != 0:
        if i == out_of-1:
            print('\nFinished.')
    return

def PT_decomp(A):
    colsums = A.sum(axis=0)
    T = np.diag(colsums)
    P = A/colsums
    
    assert np.allclose(A, P@T), "Decomposition does not agree with original matrix"
    return P, T

def Fnorm(A):
    return np.sqrt((A**2).sum())

def gamma_inf(A,B):
    N = A.shape[0]
    rA,x,y=Perron_info(A)

    result = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                result += x[i] * y[i] * (B[i][i] - A[i][i])
            else:
                result += (A[i][j] * x[i] * y[j] ) * np.log(B[i][j]/A[i][j])

    return rA + result
                
def C_pq(x, y, p, q):
    m1,M1 = x.min(), x.max()
    m2,M2 = y.min(), y.max()

    Cpq = (M1**p * M2**q - m1**p * m2**q) / ((p*(M1*m2*M2**q - m1*M2*m2**q))**(1/p) * (q*(m1*M2*M1**p - M1*m2*m1**p))**(1/q) )

    return Cpq

def Aji(x, y, p, q, N, A_before_help=None):
    if N == 1:
        return np.ones_like(x)
    
    if N > 1:
        D = np.zeros_like(x)
        dim = D.shape[0]
        for j in range(dim):
            for i in range(dim):
                term1 = np.linalg.matrix_power(x, N - 1)[j,:] * x[:,i]
                term2 = np.linalg.matrix_power(y, N - 1)[j,:] * y[:,i]

                D[j][i] = 1/C_pq(term1, term2, p, q)
        if A_before_help is None:
            return D * (Aji(x, y, p, q, N - 1)).max(axis=1)
        else:
            return D * A_before_help.max(axis=1)

    
def lb_comp(D1,D2, a1, a2, N):
    return (DGM_bound(np.linalg.matrix_power(D1,N),np.linalg.matrix_power(D2,N), 1/a1,1/a2))**(1/N)

def log_convex(f, N=4, neg_rwds=True, sublog=True):
    x_i = np.linspace(-5,0,100)
    if neg_rwds:
        print("testing with all negative rewards")
        pass
    else:
        print("testing with all positive rewards")
        x_i *= -1
    xis=[]
    for x in range(N):
        xis.append(np.random.choice(x_i))
    xis = np.array(xis)
    lhs = f(np.log( np.exp(xis).sum() ) )

    rhs = np.log(  (np.exp(f(xis))).sum()  )
    if sublog:
        print("testing sublog convexity")
        return lhs <= rhs
    if not sublog:
        print("testing superlog convexity")
        return lhs >= rhs

def count_real_eigs(A):
    evs = np.real_if_close(np.linalg.eigvals(A))
    real_evs = np.real(evs[np.isreal(evs)])
    
    return len(real_evs)

def second_eval_real(A):
        return np.isreal(all_evals(A)[1])
    
def maze_rwds(A,S, yield_T=False, beta=1):
    rwds = [-1]*S
    goal = np.random.choice(list(range(S)))
    rwds[goal] = 0
    rwds_array = np.array([ [i]*A for i in rwds]).flatten()

    if not yield_T:
        return rwds_array
    else:
        return np.diag(np.exp(beta*rwds_array))
        
def dembele_algo(A, n_iter=1):
    dim = A.shape[0]
    An = np.zeros_like(A)
    for n in range(n_iter):
        rowsums = col_sums(A)
        for i in range(dim):
            for j in range(dim):
                An[i][j] = rowsums[j] * A[i][j] / rowsums[i]
                
        A = An
    return A

def linear(x, A, b):
    return A*x + b    

def kolo_bounds(D):
    import matplotlib.pyplot as plt
    alphas = np.linspace(0,1,100)
    rs = row_sums(D)
    lbs=[]
    ubs=[]
    for alpha in alphas:
        ri_rj = np.kron(rs**alpha,rs**(1-alpha))
        lbs.append(ri_rj.min())
        ubs.append(ri_rj.max())
    print(np.unique(lbs))
    plt.plot(alphas, lbs)
    plt.plot(alphas, ubs)
    plt.plot(alphas, [Perron_info(D)[0]]*len(alphas), 'k-')


def Song_lower_bound(A,x):
    # https://doi.org/10.1016/0024-3795(92)90183-B
    n = A.shape[0]
    assert n == x.shape[0]
    q = - np.diag(A).min()
    # q must be greater than or equal to that value^
    prod1 = 1
    for i in range(n):
        for j in range(n):
            if i <= j:
                prod1 *= (A[i][j]*A[j][i]) ** (x[i] * x[j])
            else:
                prod1 *= 1
            # prod1 *= A[i][j] ** (x[i] *x[j])
            
    prod2=1
    for i in range(n):
        prod2 *= (A[i][i] + q)**(x[i]**2)
        
    # return prod1 / (x*x).sum()
    return prod1 * prod2 / (x*x).sum() - q


class TwistedMatrix:
    def __init__(self, P=None, T=None, beta = 1, S=None, A=None, det=False, MDP=None):
           
        self.A = A
        self.S = S
        self.P = P
        self.beta = beta
        self.det = det
        self.MDP = MDP
        
        if T is None:
            self.T = twister(A,S, beta = beta)
        else:          
            self.T = T
            
        self.rwds = np.log(np.diag(self.T))/self.beta
                        
        if self.P is None:
            if self.MDP is None:
                self.MDP = MDP_dynamic(self.A,self.S, det=self.det)
            self.P = full_dynamic(self.MDP, det = self.det)
            
        if self.MDP is None:
            pass
        else:
            self.Dss = little_q(self.MDP, self.T)
            self._gather_ss_perron()
            
        
        self.D = self.P @ self.T
        self.SA = self.D.shape[0]
        

        self._gather_perron()
    
    def update_beta(self, beta):
        self.D = self.P @ np.diag( np.exp(beta * self.rwds) )
        self.beta = beta
        
        self._gather_perron()
        
        
    def _gather_perron(self):
        self.rho, self.u, self.v = Perron_info(self.D)
        
    def _gather_ss_perron(self):
        _, self.chi, _ = Perron_info(self.Dss)
        
    def all_evals(self):
        return np.real_if_close(np.linalg.eigvals(self.D))
    
    def mixing_ratio(self):
        evs = self.all_evals()
        return np.abs(evs[1])/evs[0]
    
    def _gersh_circles(self, A):
        centers = []
        radii = []
        for i in range(A.shape[0]):
            center = A[i][i]
            radius = A[:][i].sum() - center
            centers.append(center)
            radii.append(radius)
        return centers, radii
    
    def _subdom_bound(self):
        result = []
        for x in [self.v]: #  [self.u,self.v]:
            x /= x.sum()
            assert (x > 0).all(), "Input vector must be non-negative"
            S = np.diag(x)
            S_inv = np.diag(1/x)
            B = S_inv @ self.D @ S
            
            centers, radii = self._gersh_circles(B)
            result.append(max(centers+radii))
        return result
    
    def generate_ratios_beta_plot(self):
        import matplotlib.pyplot as plt
        self.ratios = []
        betas = np.linspace(0.05,10,1000)
        for beta in betas:
            self.update_beta(beta)
            r0, r1 = self.all_evals()[:2]
            self.ratios.append(np.log(r1/r0))
        plt.figure()
        plt.title('Spectral Gap vs. Temperature', fontsize=30)
        plt.plot(betas, self.ratios, 'bo-')
        
        
    def generate_theta_beta_plot(self, linear_fit=False):
        if linear_fit:
            from scipy.optimize import curve_fit
        import matplotlib.pyplot as plt
        thetas = []
        betas = np.linspace(0.001,10,1000)
        for beta in betas:
            self.update_beta(beta)
            thetas.append(-np.log(self.rho))
            # thetas.append(self.rho)
        plt.figure()
        plt.title(r'$\Theta$ vs. $t\beta$', fontsize=30)
        plt.plot(betas[1:], thetas[1:], 'bo-')
        popt,pcov = curve_fit(linear, betas[1:], thetas[1:])
        plt.plot(betas, linear(betas, *popt), 'r-')
        print(popt)
        
        
    def CandSbounds(self, LB_mat=None, UB_root=None, opt_choice=False, debug=False):
        if LB_mat is None:
            if opt_choice:
                LB_mat = self.D
            else:
                LB_mat = np.diag(self.T).min() * self.P
        else:
            pass
    
        if UB_root is None:
            if opt_choice:
                UB_root = self.rho
            else:
                UB_root = self.T.max()
        
        Z = np.linalg.matrix_power(UB_root*np.identity(self.SA) - LB_mat, -1)
        Omega, Sigma = col_sums(Z), row_sums(Z)
        Zr = np.diag(1/Omega) @ Z
        Zc = np.diag(1/Sigma) @ Z
        if debug:
            return Zr, Zc
        else:
            return Zc.min(axis=1), Zc.max(axis=1)
    
    def Q_fn(self, N):
         return np.log(col_sums(np.linalg.matrix_power(self.D, N))) 
     