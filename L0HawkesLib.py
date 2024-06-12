# -*- coding: utf-8 -*-
"""
Originally created on Sun Nov 21 23:55:15 2021.
Redeveloped on 06/2024 as independent code. 

@author: Tsuyohi (Ide-san) Ide (tide@us.ibm.com) 
"""
import numpy as np

def L0HawkesNh(timestamps, event_types, Nh, sparse_level=0.75,
               decayfunc='power', prior = 'gg', 
               itr_max=50, err_threshold=1e-4, reporting_interval=10,
               nu_mu=0.1, nu_beta=0.1, nu_A=0.1, epsilon=0.01,
               a_mu=1.001, b_mu=0.01 ,a_beta=1.01, b_beta=0.01,eta=2):
    '''
    Learns L0Hawkes model (Ide et al., NeurIPS 21).

    Parameters
    ----------
    timestamps : 1D numpy array
        Array of timestamps. Assumed to have N+1 events, where the 0-th event is the 
        genesis event and treated differently from the others. 
    event_types : 1D numpy array
        Array of event types in the original names of the N+1 events. 
    Nh : int
        The number of event instances kept in the history. Nh <= N must hold. 
        For the first Nh instances, the history is shorter than Nh.
    sparse_level : double
        Level of sparsity for the impact matrix. Must be in [0.5,1).
        Corresponds to the probability of getting zero in Bernoulli distribution.
        The default is 0.75.
    decayfunc : string
        Specifies the name of the decay function. Either power or exponential. 
        Only the first three letters (pow, exp) matters. Case insensitive.
    prior : string
        Specifies the prior distribution (or regularization). Currently, gauss
        (Gaussian or L2 regularization), gamma (Gamma distribution), 
        or gg (L2+gamma) is allowed. 
    itr_max : double, optional
        The maximum number of MM iterations. The default is 50.
    err_threshold : double, optional
        Threshold for the residual from the previous round. Used to check 
        convergence. The default is 1e-4.
    reporting_interval : int, optional
        How often you want to get residual feedback. The default is every 10 
        iterations. 
    nu_mu : double, optional
        L2 regularization strength for mu. Will be ignored when prior=gamma. 
        The default is 0.1.
    nu_beta : double, optional
        L2 regularization strength for beta. Will be ignored when prior=gamma. 
        The default is 0.1.
    nu_A : double, optional
        L2 regularization strength for A. Will not be affected by "prior". The 
        default is 0.1.
    epsilon : double, optional
        Threshold parameter for the epsilon-sparsity theory. The default is 0.01.
    a_mu : double, optional
        The shape parameter of the Gamma prior for mu. Must be greater than 1.
        The default is 1.001.
    b_mu : double, optional
        The rate parameter of the Gamma prior for mu. Must be positive. The 
        default is 0.01.
    a_beta : double, optional
        The shape parameter of the Gamma prior for beta. Must be greater than 1.
        The default is 1.01.
    b_beta : double, optional
        The rate parameter of the Gamma prior for beta. Must be positive. The 
        default is 0.01.
    eta : double, optional
        Exponent of the power decay. Ignored if decayfunc is not power. 
        The default is 2.

    Returns
    -------
    obj : dictionary{'event_list','learned_params','regularizer_params','training_data'}      
        event_list : 1D numpy array
            Mapping from event type index to the original event type name
        learned_params : dictionary
            mu : 1D numpy array
                Baseline intensities
            beta : 1D numpy array
                Decay parameters
            A : 2D numpy array
                Impact matrix as the final solution. Use this (not A_epsilon)
                for practical purposes. This is computed by post-processing 
                A_epsilon.
            A_epsilon : 2D numpy array
                Raw impact matrix before postprocess. This is only for 
                mathematical investigations. Use A instead for applications.
            l0norm : double
                L0 norm of A
            qself : 1D numpy array
                Self-triggering probabilities
            qhist : 1D numpy array
                Triggering probabilities. n-th row is for n-th event instance
            l0l2sol : dictionary
                Object returned from L0L2sqlog_plus_linear
            loglik : list
                log likelihood values
        regularizer_params : dictionary
            decayfunc : string
                decayfunc
            prior : string
                prior
            nu_A : double 
                nu_A
            sparse_level : double 
                sparse_level
            tau : double 
                tau = ln(sparse_level/(1-sparse_level))
            epsilon : double 
                epsilon
            nu_mu : double
                nu_mu
            nu_beta : double
                nu_beta
            a_mu : double
                a_mu
            b_mu : double
                b_mu
            a_beta : double
                a_beta
            b_beta : double
                b_beta
        training_data : dictionary
            timestamps : 1D numpy array
                timestamps
            event_types : 1D numpy array
                event types as original names
            event_types_idx : 1D numpy array
                event types as indices
    '''

    INT_NAN = np.iinfo(np.int64).max # for integer nan (What's the standard way?)
    
    #===================== data setup
    timestamps,event_types = verify_timeorder(timestamps, event_types,Nh) 
    verify_input(decayfunc,sparse_level,prior,nu_mu,nu_beta,nu_A,
                 a_mu,b_mu,a_beta,b_beta) 
    
    # event_list defines the mapping between event type indices (integer)
    # and the original event type names, which can be string.
    # event_types[k] is the original event type name of the event type index k
    event_list = np.unique(event_types) 
    
    # N is the total number of ACTIVE events. Note that the 0th one is 
    # the genesis event and is not included in N
    N = len(event_types) -1 # Num of instances. 
    D = len(event_list) # Num of event types
    tau = np.log(sparse_level /(1-sparse_level))
    print('---- N(#events)={}, D(#types)={}, Nh(hist.size)={}'.format(N,D,Nh))
    
    # What instances does each event type have? 
    indices_of = [] # indices_of[k] is the set of type-k instance indices
    zero2N = np.arange(0,len(event_types))
    for event in event_list:
        mask = (event_types == event)
        indices_of.append(zero2N[mask])
    
    # Will use an integer version instead of the original event_types
    event_types_idx = np.zeros(N+1,dtype='uint64')
    for k in range(len(event_list)):
        mask = (event_types==event_list[k])
        event_types_idx[mask] = k
         
    #===================== Pre-computation
    Delta = np.repeat(np.nan,(N+1)*Nh).reshape([N+1, Nh])
    ddtt = np.zeros([N+1])
    
    # ddtt: t(n) - t(n-1) for n = 1,..., N
    ddtt[0] = np.nan
    ddtt[1:] = timestamps[1:] - timestamps[:N]
       
    # Delta: t(n)-t(i), where i is in the n-th history
    for n in range(1,(N+1)):
        history_n = history_of(n, Nh) # set of instance indices in the history
        for i in range(len(history_n)):
            Delta[n,i] = timestamps[n] - timestamps[history_n[i]] 

    # types_of[n,i] is the event type of the i-th event in the n's history
    '''
    If memory size is critical, comment out below and use update_A() and 
    update_beta() instead of update_A3() and update_beta3, which will save 
    ~30% memory size but slower.
    '''
    # from here ----
    types_of = np.repeat(INT_NAN,(N+1)*Nh).reshape([N+1,Nh])        
    for n in range(N+1): # n=0, 1.., N
        if n >= Nh:
            types_of[n,:] = event_types_idx[np.arange((n-Nh),n)] 
        elif n < Nh:
            types_of[n,0:n] =  event_types_idx[np.arange(0,n)]  
    #---- to here
    
    #===================== Initialization
    mu = initialize_mu(timestamps,event_types_idx,a_mu,b_mu)
    beta = mu.copy()
    A = initialize_A(timestamps,event_types_idx,nu_A,tau)
    qself,qhist = initialize_q(timestamps, event_types_idx, Nh,
                               decayfunc, beta=beta,eta=eta)

    #===================== MM iteration
    print('---- itr_max={},residual threshold={}'.format(itr_max,err_threshold))
    digits = 1 + int(np.abs(np.log10(err_threshold))) # for showing progress
    Q = np.zeros([D,D])
    H = np.zeros([D,D])
    loglik = list()
    beta_old = 2*beta.copy(); mu_old = 2*mu.copy(); A_old = 2*A.copy()
    for itr in range(itr_max):
        # update baseline intensities
        mu = update_mu(mu,prior, qself,indices_of,ddtt, nu_mu,a_mu,b_mu)
        
        # update impact matrix (class-level causal graph)
        # update_A3 is ver.1
        A, l0norm, l0l2sol = update_A3(A, Q, H,types_of,event_types_idx,
                                      indices_of, qhist, nu_A, tau, epsilon, 
                                      decayfunc, eta, beta, Delta, ddtt)
        ''' # update_A is ver.0
        A, l0norm, l0l2sol = update_A(A, Q, H,event_types_idx,
                                      indices_of, qhist, nu_A, tau, epsilon, 
                                      decayfunc, eta, beta, Delta, ddtt)
        '''
        # update decay parameters 
        # update_beta is faster for large Ns.
        beta=  update_beta(beta,decayfunc,eta,prior,nu_beta,a_beta,b_beta,
                           event_types_idx,A,indices_of,
                           qself, qhist, Delta, ddtt)   
        '''# update_beta3 is faster only when N ~< 1000.
        beta = update_beta3(beta,decayfunc,eta,prior,nu_beta,a_beta,b_beta,
                        types_of,event_types_idx,A,indices_of,
                        qself, qhist, Delta, ddtt)
        '''
        # update triggering probabilities (instance-level causality)
        qself,qhist = update_q(qself,qhist,event_types_idx,
                               mu,beta,Delta,eta,A,decayfunc)
        
        # Computing log likelihood -----
        loglik0 = 0 
        loglik1 = 0
        for n in range(1,N+1): # n=1,...N
            d_n = event_types_idx[n]
            mu_n = mu[d_n]
            A_n = A[d_n, event_types_idx[history_of(n,Nh)]]
            Delta_n = Delta[n,0:len(A_n)]
            beta_n = beta[d_n]
            decay_n = decay(beta_n, Delta_n,decayfunc,eta=eta)
            lambda_n = mu_n + (A_n * decay_n).sum()
            loglik0 = loglik0 + np.log(lambda_n)
        
            Delta_n1 = Delta_n - ddtt[n]
            h_n = decay_integral(beta_n, Delta_n, decayfunc, eta=eta) \
                - decay_integral(beta_n, Delta_n1, decayfunc, eta=eta)
            loglik1 = loglik1 - ddtt[n]*mu_n - (A_n*h_n).sum()            

        if prior.lower().startswith('gau'):
            ln_Gauss_mu = - 0.5*nu_mu*(mu*mu).sum()
            ln_gamma_mu = 0.
            ln_Gauss_beta = - 0.5*nu_beta*(beta*beta).sum()        
            ln_gamma_beta = 0.
        elif prior.lower().startswith('gam'):
            ln_Gauss_mu = 0.
            ln_gamma_mu = ln_gamma_sum(mu,a_mu,b_mu)
            ln_Gauss_beta = 0.
            ln_gamma_beta = ln_gamma_sum(beta,a_beta,b_beta)
        elif prior.lower().startswith('gg'):
            ln_Gauss_mu = - 0.5*nu_mu*(mu*mu).sum()
            ln_gamma_mu = ln_gamma_sum(mu,a_mu,b_mu)
            ln_Gauss_beta = - 0.5*nu_beta*(beta*beta).sum()
            ln_gamma_beta = ln_gamma_sum(beta,a_beta,b_beta)
        
        ln_Gauss_A = -0.5*nu_A*(A*A).sum()
        ln_Bernoulli_A = - tau*l0norm
        
        loglik_reg = ln_Gauss_mu + ln_gamma_mu + ln_Gauss_beta + ln_gamma_beta\
            + ln_Gauss_A + ln_Bernoulli_A
        
        loglik_total = loglik0 + loglik1 + loglik_reg
        loglik.append(loglik_total)        
        
        # Checking convergence -----
        errb = 1 - (beta*beta_old).sum()/np.sum(beta**2)        
        errb = np.abs(errb)
        errm = 1 - (mu*mu_old).sum()/np.sum(mu**2)
        errm = np.abs(errm)
        errA = 1 - (A*A_old).sum()/np.sum(A**2)
        errA = np.abs(errA)        
        if  errb < err_threshold and errm < err_threshold and errA < err_threshold:
            print('---- Converged(th={}) at itr={}'.format(err_threshold,itr+1))
            print('\tfinal residual(mu,beta,A)=({:.4g},{:.4g},{:.4g}),loglik={}'\
                  .format(errm,errb,errA,loglik_total))
            break
        elif (itr+1) % reporting_interval == 0 and itr != 0:
            print('{:4d}:residual(mu,beta,A)=('.format(itr+1),end='')
            print('{:{dd}.{digits}f},'.format(errm,dd=digits+2,digits=digits),end='')
            print('{:{dd}.{digits}f},'.format(errb,dd=digits+2,digits=digits),end='')
            print('{:{dd}.{digits}f}'.format(errA,dd=digits+2,digits=digits),end='')
            print('),loglik={}'.format(loglik_total))
        beta_old[:] = beta[:]
        mu_old[:] = mu[:]
        A_old[:,:] = A[:,:]
        
    #===== Reporting results =====    
    print('\tsparse_level={}(tau={:.4g}), nu_A={}, eps={}'.\
          format(sparse_level,tau,nu_A,epsilon))
    print('\tdecayfunc={}, nu_beta={}, a_beta={}, b_beta={}'.\
          format(decayfunc,nu_beta,a_beta,b_beta))    
    print('\tprior={}, nu_mu={}, a_mu={}, b_mu={}, final loglik={}'.\
          format(prior,nu_mu,a_mu,b_mu,loglik_total))
    
    if tau > 0:
        A_final =l0l2sol.get('x_sparse').reshape([D,D],order='C')  
    else:
        A_final = A.copy()
        
    regularizer_params = {'decayfunc':decayfunc,'prior':prior,
                          'nu_A':nu_A,'sparse_level':sparse_level,'tau':tau,
                          'epsilon':epsilon,'nu_mu':nu_mu, 'nu_beta':nu_beta,
                          'a_mu':a_mu, 'b_mu':b_mu, 
                          'a_beta':a_beta,'b_beta':b_beta}
    learned_params = {'mu':mu,'beta':beta,'A':A_final,
                      'A_epsilon':A,'l0norm':l0norm,
                      'qself':qself,'qhist':qhist, 
                      'l0l2sol':l0l2sol,'loglik':loglik}
    training_data = {'timestamps':timestamps,'event_types':event_types,
                         'event_types_idx':event_types_idx}
    obj = {'event_list':event_list, 'learned_params':learned_params,
           'regularizer_params':regularizer_params,
           'training_data':training_data}
    return obj


def verify_input(decayfunc, sparse_level, prior, nu_mu, nu_beta, nu_A,
                 a_mu, b_mu, a_beta, b_beta):
    '''
    Check consistency of the parameters fed to L0HawkesNh.

    Parameters
    ----------
    decayfunc : string
        Specifies the name of the decay function. Either power or exponential. 
        Only the first three letters (pow, exp) matters. Case insensitive.
    sparse_level : double
        Level of sparsity for the impact matrix (A). Must be in [0.5,1).
        Corresponds to the probability of getting zero in Bernoulli distribution.
    prior : string
        Specifies the prior distribution (or regularization). Currently, gauss
        (Gaussian or L2 regularization), gamma (Gamma distribution), 
        or gg (L2+gamma) is allowed. 
    nu_mu : double
        L2 regularization strength for mu. Will be ignored when prior=gamma.
    nu_beta : double
        L2 regularization strength for beta. Will be ignored when prior=gamma.
    nu_A : double
        L2 regularization strength for A. Will not be affected by prior.
    a_mu : double
        The shape parameter of the Gamma prior for mu. Must be greater than 1.
    b_mu : double
        The rate parameter of the Gamma prior for mu. Must be positive.
    a_beta : double
        The shape parameter of the Gamma prior for beta. Must be greater than 1.
    b_beta : double
        The rate parameter of the Gamma prior for beta. Must be positive.

    Raises
    ------
    ValueError
        Raised when invalid parameters are given.

    Returns
    -------
    None.

    '''
    L2strengths = [nu_mu,nu_beta,nu_A]
    Gamma_shapes = [a_mu,a_beta]
    Gamma_rates = [b_mu, b_beta]       
        
    if not decayfunc.lower().startswith(('exp','pow')):
        raise ValueError('{}:decayfunc invalid'.format(decayfunc))
    elif sparse_level < 0.5 or sparse_level >=1:
       raise ValueError('f{sparse_level}:sparse_level must be in [0.5,1)')
    elif not prior.lower().startswith(('gauss','gamma','gg')):
        raise ValueError('f{prior}:prior invalid')
    elif any(ge <=0 for ge in L2strengths):
       raise ValueError('L2 strengths must be positive')
    elif any(ge < 1 for ge in Gamma_shapes):
       raise ValueError('Gamma prior shape must not be less than 1')
    elif any(ge <= 0 for ge in Gamma_rates):
       raise ValueError('Gamma prior rates must be positive')


def verify_timeorder(timestamps,event_types,Nh):
    '''
    Checks if provided training data are consistent.

    Parameters
    ----------
    timestamps : 1D numpy array
        timestamps of the event instances.
    event_types : 1D numpy array
        event types of the event instances.
    Nh : int
        The max. number of event instances in the history. Nh <= N must hold.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    timestamps : 1D numpy array
        Properly ordered timestamps.
    event_types : 1D numpy array
        Properly time-ordered event types.

    '''
    
    if len(event_types) != len(timestamps):
        raise ValueError('Length differs:timestamp and event_types')
    elif Nh > len(event_types) -1:
        raise ValueError('Nh is too large (cannot be greater than N)')
    elif Nh < 0:
        raise ValueError('Nh cannot be negative)')
    elif not isinstance(timestamps,np.ndarray):
        raise ValueError('timestamps must be numpy 1D array')
    elif not isinstance(event_types, np.ndarray) :
       raise ValueError('event_types must be numpy 1D array')

    isTimeOrdered = False
    dt_timestamps = (timestamps[1:-1] - timestamps[0:-2])

    isTimeOrdered = ((dt_timestamps < 0).sum() == 0 )

    if ~isTimeOrdered:
        raise ValueError('timestamps must be time-ordered.')
    
    if not isinstance(timestamps[0],float):
        timestamps = np.array(timestamps,dtype='float64')    
    if not (timestamps == sorted(timestamps)).all():
        idxes = timestamps.argsort()
        timestamps = timestamps.copy()[idxes]
        event_types = event_types.copy()[idxes]  
    
    return timestamps,event_types

def history_of(n,Nh):
    '''
    For the n-th instance, this function returns the set of event indices 
    in the history. The maximum number of history events is limited to Nh.
    Note that for n=1,..,Nh, the set size is smaller than Nh. 

    Parameters
    ----------
    n : int
        Event instance index.
    Nh : int
        The maximum size of event history kept.

    Raises
    ------
    ValueError
        Raised when a negative n or n=0 is passed.

    Returns
    -------
    history_instance_indices : 1D numpy array
        History event indices of the n-th instance.

    '''
    if n >= Nh:
        history_instance_indices = np.arange((n-Nh),n-1+1) # ending n-1
    elif 0<n and n < Nh:
        history_instance_indices =  np.arange(0,n-1+1) # ending n-1
    else:
        raise ValueError('{}:Invalid index'.format(n))
    return history_instance_indices

def exclude_zeroth_instance(instance_index_array):
    '''
    Removes 0 from an instance index set. The zero-th event is the genesis event.
    Since it does not have any history, the likelihood is not defined for that
    event. 

    Parameters
    ----------
    instance_index_array : 1D numpy array of indices
        Instance index set.

    Returns
    -------
    1D numpy array of indices
        Instance index set not containing 0, if any.

    '''
    zero_idx = np.where((instance_index_array==0))
    return np.delete(instance_index_array,zero_idx)
        
def decay(beta,u,func,eta,**kwargs):
    '''
    Decay function 

    Parameters
    ----------
    beta : double
        Decay parameter.
    u : double or numpy array
        Time variable.
    func : string
        Type of decay function. Currently, power or exponential is allowed.
    eta : double
        The exponent of the power distribution. Ignored when func=exp. 
    **kwargs : TYPE
        For future extensions. Currently not used. 

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    double or numpy array
        Value of decay function.

    '''
    if func.lower().startswith('exp'):
        return decay_exponential(beta,u)
    elif func.lower().startswith('pow'):
        return decay_power(beta,u,eta)
    else:
        raise ValueError('{}:undefined decay func type'.format(func))
    
def decay_integral(beta,u,func,eta,**kwargs):
    '''
    Indefinite integral of the decay function.

    Parameters
    ----------
    beta : double
        Decay parameter.
    u : double or numpy array
        Time-variable.
    func : string
        Decay function name. Currently, power or exponential is allowed. 
    eta : double
        Exponent of the power distribution. Ignored when func=exp.
    **kwargs : TYPE
        Currently unused. For future extensions. 

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    double or numpy array
        Value(s) of the indefinite integral.

    '''
    if func.lower().startswith('exp'):
        return decay_integral_exponential(beta,u)
    elif func.lower().startswith('pow'):
        return decay_integral_power(beta,u,eta)
    else:
        raise ValueError('{}:undefined decay func type'.format(func))
        
def dh_derivative(beta,u,func,eta,**kwargs):
    '''
    Derivative w.r.t. beta of the indefinite integral of the decay function. 

    Parameters
    ----------
    beta : double
        Decay parameter.
    u : double or numpy array
        Time variable(s).
    func : string
        Decay function name. Currently, power or exponential is allowed. 
    eta : double
        Exponent of the power distribution. Ignored when func=exp..
    **kwargs : TYPE
        Currently unused. For future extensions.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    double or numpy array
        Value(s) of the derivative w.r.t. beta of the indefinite integral 
        of the decay function.

    '''
    if func.lower().startswith('exp'):
        return dh_derivative_exponential(beta,u)
    elif func.lower().startswith('pow'):
        return dh_derivative_power(beta,u,eta)
    else:
        raise ValueError('{}:undefined decay func type'.format(func))    
        
def ln_decay_derivative(beta,u,func,eta,**kwargs):
    '''
    Logarithm of the derivative of the decay function

    Parameters
    ----------
    beta : double
        Decay parameter.
    u : double or numpy array
        Time variable(s).
    func : string
        Decay function name. Currently, power or exponential is allowed. 
    eta : double 
        The exponent of the power decay. Ignored when func=exp.
    **kwargs : TYPE
        Currently unused. For future extensions

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        Value(s) of the log of the derivative of the decay function.

    '''
    if func.lower().startswith('exp'):
        return ln_decay_derivative_exponential(beta,u)
    elif func.lower().startswith('pow'):
        return ln_decay_derivative_power(beta,u,eta)
    else:
        raise ValueError('{}:undefined decay func type'.format(func))        

def decay_exponential(beta,u):
    '''
    Exponential decay distribution beta*exp(-beta*u).

    Parameters
    ----------
    beta : double
        Decay parameter.
    u : double or numpy array
        Time variable(s).

    Returns
    -------
    TYPE
        Value(s) of decay function.

    '''
    return beta*np.exp(-beta*u)

def decay_power(beta,u,eta):
    '''
    Power decay distribution

    Parameters
    ----------
    beta : double
        Decay parameter.
    u : double or numpy array
        Time variable(s).
    eta : double
        The exponent of the power decay. Ignored when func=exp.

    Returns
    -------
    TYPE
        Value(s) of decay function.

    '''
    return eta*beta*(1 + beta * u)**(-eta-1)

def decay_integral_exponential(beta,u):
    '''
    Indefinite integral of the exponential decay function.

    Parameters
    ----------
    beta : double 
        Decay parameter.
    u : double or numpy array
        Time variable(s).

    Returns
    -------
    double or numpy array
        Value(s) of the indefinite integral of the exponential decay function.

    '''
    return -np.exp(-beta*u)

def decay_integral_power(beta,u,eta):
    '''
    Indefinite integral of the power decay function.

    Parameters
    ----------
    beta : double 
        Decay parameter.
    u : double or numpy array
        Time variable(s).
    eta : double
        Exponent of the power distribution.

    Returns
    -------
    double or numpy array
        Value(s) of the indefinite integral of the power decay function.

    '''
    return -(1 + beta * u)**(-eta)

def dh_derivative_exponential(beta,u):
    '''
    Derivative w.r.t. beta of the indefinite integral of the exponential
    decay function. 

    Parameters
    ----------
    beta : double 
        Decay parameter.
    u : double or numpy array
        Time variable(s).

    Returns
    -------
    double or numpy array
        Function values.

    '''
    return u*np.exp(-beta*u)

def dh_derivative_power(beta,u,eta):
    '''
    Derivative w.r.t. beta of the indefinite integral of the power 
    decay function. 

    Parameters
    ----------
    beta : double 
        Decay parameter.
    u : double or numpy array
        Time variable(s).
    eta : double
        Exponent of the power distribution.

    Returns
    -------
    double or numpy array
        Function values.

    '''
    return eta*u*(1 + beta*u)**(-eta-1)


def ln_decay_derivative_exponential(beta,u):
    '''
    Logarithm of the derivative of the exponential decay function.

    Parameters
    ----------
    beta : double
        Decay parameter.
    u : double or numpy array
        Time variable(s).

    Returns
    -------
    double or numpy array
        Function value(s).

    '''
    return -u
    
def ln_decay_derivative_power(beta,u,eta):
    '''
    Logarithm of the derivative of the power function.

    Parameters
    ----------
    beta : double
        Decay parameter.
    u : double or numpy array
        Time variable(s).
    eta : double
        Exponent of the power distribution.

    Returns
    -------
    double or numpy array
        Function value(s).

    '''
    return -(eta + 1)*u/(1 + beta*u)

def update_beta(beta,decayfunc,eta,prior,nu_beta,a_beta,b_beta,
                event_types_idx,A,indices_of,
                qself, qhist, Delta, ddtt):
    '''
    Updates beta (decay parameter) in the Minorization-Maximization algorithm.
    Note that denom_beta_k ($D^\beta_k$ in the paper) is a function of beta 
    itself. Initialization is necessary. 

    Parameters
    ----------
    beta : numpy array
        Decay parameters.
    decayfunc : string
        Decay function name. Currently, power or exponential is allowed. 
    eta : double
        The exponent of the power decay. Ignored if decayfunc=exp
    prior : string
        Specifies the prior distribution. Gauss, gamma, or gg. 
    nu_beta : double
        L2 regularization strength for beta.
    a_beta : double
        Shape parameter for beta. Must not be less than 1.
    b_beta : double
        Rate parameter for beta. Must be positive.
    event_types_idx : 1D numpy array
        Array of event types represented in terms of integer indices, 
        rather than the original event type names that could be string.
    A : 2D numpy array
        Impact matrix .
    indices_of : 1D numpy array
        Set of event instance indices belonging to each event type.
        All the instances of indices_of[1] have the same event type 1. 
    qself : 1D numpy array
        Self triggering probabilities. Note qself[0]=np.nan as the 0-th event
        is the genesis event. 
    qhist : 2D numpy array
        NxNh matrix whose n-th row is the triggering probabilities of the n-th
        event instance.
    Delta : 2D numpy array
        Matrix of t(n)-t(i), where t(i) is a history event of the n-th instance.
    ddtt : 1D numpy array
        For n=1,...N, ddtt[n]=t(n)-t(n-1).

    Returns
    -------
    beta : 1D numpy array
        Updated decay parameters.

    '''
    Nh = qhist.shape[1]
    for k in range(len(beta)):
        beta_old_k = beta[k]
        idx_of_k = indices_of[k]
        if 0 in idx_of_k:
            idx_of_k = exclude_zeroth_instance(idx_of_k.copy())
        
        numerator_beta_k = (1-qself[idx_of_k]).sum()

        denom_beta_k = 0 
        for n in idx_of_k: 
            history_of_n = history_of(n, Nh)
            # denominator 1st term
            Delta_n = Delta[n,0:len(history_of_n)]
            D1vec_n = ln_decay_derivative(beta_old_k,Delta_n, decayfunc,eta)
            qhist_n = qhist[n,0:len(history_of_n)]
            D1_kn = (D1vec_n*qhist_n).sum() # sum over the history of n
            
            # denominator 2nd term
            A_k = A[k,event_types_idx[history_of_n]]  
            Delta_n1 = Delta_n - ddtt[n]        
            dhdbeta_k = \
                dh_derivative(beta_old_k, Delta_n, decayfunc, eta=eta)\
                    - dh_derivative(beta_old_k, Delta_n1, decayfunc, eta=eta)        
            D2_kn = (A_k * dhdbeta_k).sum()
            denom_beta_k = denom_beta_k + (-D1_kn + D2_kn)
        
        if prior.lower().startswith('gauss'): # Gaussian prior or L2 reg.
            beta[k] = (-denom_beta_k + \
                       np.sqrt(denom_beta_k**2 \
                               + 4.*numerator_beta_k*nu_beta))/(2.*nu_beta) 
        elif prior.lower().startswith('gamma'): # Gamma prior
            beta[k] = (numerator_beta_k + a_beta -1)/(denom_beta_k + b_beta)
        elif prior.lower().startswith('gg'): # L2 + Gamma
            denom_beta_k = denom_beta_k + b_beta
            numerator_beta_k = numerator_beta_k + a_beta -1 
            beta[k] = (-denom_beta_k
                       + np.sqrt(denom_beta_k**2 
                                 + 4.*numerator_beta_k*nu_beta))/(2.*nu_beta)    
    return beta

def update_beta3(beta,decayfunc,eta,prior,nu_beta,a_beta,b_beta,
                types_of,event_types_idx,A,indices_of,
                qself, qhist, Delta, ddtt):
    '''
    A variant of update_beta(). For N<1000, this is faster than update_beta().
    Updates beta (decay parameter) in the Minorization-Maximization algorithm.
    Note that denom_beta_k ($D^\beta_k$ in the paper) is a function of beta 
    itself. Initialization is necessary. 

    Parameters
    ----------
    beta : numpy array
        Decay parameters.
    decayfunc : string
        Decay function name. Currently, power or exponential is allowed. 
    eta : double
        The exponent of the power decay. Ignored if decayfunc=exp
    prior : string
        Specifies the prior distribution. Gauss, gamma, or gg. 
    nu_beta : double
        L2 regularization strength for beta.
    a_beta : double
        Shape parameter for beta. Must not be less than 1.
    b_beta : double
        Rate parameter for beta. Must be positive.
    types_of : 2D numpy array
        [n,i] element specifies the event type (index) of the i-th event in the
        n's history.
    event_types_idx : 1D numpy array
        Array of event types represented in terms of integer indices, 
        rather than the original event type names that could be string.
    A : 2D numpy array
        Impact matrix .
    indices_of : 1D numpy array
        Set of event instance indices belonging to each event type.
        All the instances of indices_of[1] have the same event type 1. 
    qself : 1D numpy array
        Self triggering probabilities. Note qself[0]=np.nan as the 0-th event
        is the genesis event. 
    qhist : 2D numpy array
        NxNh matrix whose n-th row is the triggering probabilities of the n-th
        event instance.
    Delta : 2D numpy array
        Matrix of t(n)-t(i), where t(i) is a history event of the n-th instance.
    ddtt : 1D numpy array
        For n=1,...N, ddtt[n]=t(n)-t(n-1).

    Returns
    -------
    beta : 1D numpy array
        Updated decay parameters.

    '''
    # A variant of update_beta() without no sums over n

    for k in range(len(beta)):
        idx_of_k = indices_of[k]
        if 0 in idx_of_k:
            idx_of_k = exclude_zeroth_instance(idx_of_k.copy())
        
        numerator_beta_k = (1-qself[idx_of_k]).sum()

        Delta_k = Delta[idx_of_k,:]
        mask = ~np.isnan(Delta_k)
        
        dlndb = ln_decay_derivative(beta[k],Delta_k[mask],decayfunc,eta)
        D1k = (qhist[idx_of_k,:][mask]*dlndb).sum()
        
        Delta1k = Delta_k - ddtt[idx_of_k,np.newaxis]
        dhdb = dh_derivative(beta[k],Delta_k[mask],decayfunc,eta)\
            - dh_derivative(beta[k],Delta1k[mask],decayfunc,eta)
            
        idx_right = types_of[idx_of_k,:]
        Andi = A[k,idx_right[mask]]
        
        D2k = (dhdb*Andi).sum()
        
        denom_beta_k = (-D1k + D2k)
        
        if prior.lower().startswith('gauss'): # Gaussian prior or L2 reg.
            beta[k] = (-denom_beta_k + \
                       np.sqrt(denom_beta_k**2 \
                               + 4.*numerator_beta_k*nu_beta))/(2.*nu_beta) 
        elif prior.lower().startswith('gamma'): # Gamma prior
            beta[k] = (numerator_beta_k + a_beta -1)/(denom_beta_k + b_beta)
        elif prior.lower().startswith('gg'): # L2 + Gamma
            denom_beta_k = denom_beta_k + b_beta
            numerator_beta_k = numerator_beta_k + a_beta -1 
            beta[k] = (-denom_beta_k
                       + np.sqrt(denom_beta_k**2 
                                 + 4.*numerator_beta_k*nu_beta))/(2.*nu_beta)    
    return beta

def update_mu(mu,prior, qself,indices_of,ddtt, nu_mu,a_mu,b_mu):
    '''
    Update mu (baseline intensity) in the Minorization-Maximization algorithm

    Parameters
    ----------
    mu : 1D numpy array
        Baseline intensities.
    prior : string
        Specifies the prior distribution (or regularization). 
        Currently, gauss, gamma, or gg is allowed. 
    qself : 1D numpy array
        Self triggering probabilities. Note qself[0]=np.nan as the 0-th event
        is the genesis event. 
    indices_of : 1D numpy array
        Set of event instance indices belonging to each event type.
        All the instances of indices_of[1] have the same event type 1. 
    ddtt : 1D numpy array
        For n=1,...N, ddtt[n]=t(n)-t(n-1).
    nu_mu : double
        L2 regularization strength for mu.
    a_mu : double
        Shape parameter for mu. Must not be less than 1..
    b_mu : double
        Rate parameter for mu. Must be positive.

    Returns
    -------
    mu : 1D numpy array
        Updated baseline intensities.

    '''
    for k in range(len(mu)):
        idx_of_k = indices_of[k]
        if 0 in idx_of_k:
            idx_of_k = exclude_zeroth_instance(idx_of_k.copy())
        denom_mu_k = ddtt[idx_of_k].sum()
        numerator_mu_k = qself[idx_of_k].sum()
        
        if prior.lower().startswith('gau'): # Gaussian prior or L2 reg.
            mu[k] = (-denom_mu_k
                       + np.sqrt(denom_mu_k**2 
                                 + 4.*numerator_mu_k*nu_mu))/(2.*nu_mu) 
        elif prior.lower().startswith('gam'): # Gamma prior
            mu[k] = (numerator_mu_k + a_mu -1)/(denom_mu_k + b_mu)
        elif prior.lower().startswith('gg'): # L2 + Gamma
            denom_mu_k = denom_mu_k + b_mu
            numerator_mu_k = numerator_mu_k + a_mu -1 
            mu[k] = (-denom_mu_k
                       + np.sqrt(denom_mu_k**2 
                                 + 4.*numerator_mu_k*nu_mu))/(2.*nu_mu)        
    return mu



def update_A(A, Q, H, event_types_idx,indices_of,qhist,nu_A, tau, epsilon, 
                              decayfunc, eta, beta, Delta, ddtt):
    '''
    Updates A (impact matrix) in the Minorization-Maximization algorithm.
    If the memory size is not an issue, use the faster version update_A3().

    Parameters
    ----------
    A : 2D numpy array
        Impact matrix.
    Q : 2D numpy array
        Coefficient matrix of the optimization problem for A.
    H : 2D numpy array
        Coefficient matrix of the optimization problem for A..
    event_types_idx : 1D numpy array
        The event type sequence represented in terms of integer indices.
    indices_of : 1D numpy array
        Set of event instance indices belonging to each event type.
        All the instances of indices_of[1] have the same event type 1. 
    qhist : 2D numpy array
        NxNh matrix whose n-th row is the triggering probabilities of the n-th.
    nu_A : double
        L2 regularization strength for A.
    tau : double
        L0 regularization strength for A.
    epsilon : double
        Threshold parameter for the epsilon-sparsity theory.
    decayfunc : string
        Decay function name. Currently, power or exponential is allowed. 
    eta : double
        Exponent of the power distribution. Ignored when decayfunc=exp
    beta : 1D array
        Decay parameters.
    Delta : 2D numpy array
        Delta[n,i]=t(n)-t(i), where t(i) is in the history set of n.
    ddtt : 1D numpy array
        Array of t(n)-t(n-1) for n = 1, ..., N.

    Returns
    -------
    A : 2D numpy array
        Impact matrix.
    l0norm : double
        L0 norm of A.
    l0l2sol : dictionary
        Returned result of L0L2sqlog_plus_linear.

    '''
    Nh = qhist.shape[1]
    for k in range(A.shape[0]):
        idx_of_k = indices_of[k]
        if 0 in idx_of_k:
            idx_of_k = exclude_zeroth_instance(idx_of_k.copy())
       
        for l in range(A.shape[1]):
            Q[k,l] = 0
            H[k,l] = 0
            for n in idx_of_k:
                history_n = history_of(n,Nh)
                mask = (event_types_idx[history_n] == l)
                qhist_n = qhist[n,0:len(history_n)]
                Q[k,l] = Q[k,l] + qhist_n[mask].sum()
                
                # Correction: Replaced the following 4 lines.
                # This achieves ~30% speedup. But still slower than update_A3.
                # This is new
                Delta_n = Delta[n,0:len(history_n)][mask]
                Delta_n1 = Delta_n - ddtt[n]        
                h_n = decay_integral(beta[k], Delta_n, decayfunc, eta=eta)\
                        - decay_integral(beta[k], Delta_n1, decayfunc, eta=eta)
                H[k,l] = H[k,l] + h_n.sum()
                ''' # This is old
                Delta_n = Delta[n,0:len(history_n)]
                Delta_n1 = Delta[n,0:len(history_n)] - ddtt[n]        
                h_n = decay_integral(beta[k] , Delta_n, decayfunc, eta=eta)\
                        - decay_integral(beta[k] , Delta_n1, decayfunc, eta=eta)
                H[k,l] = H[k,l] + h_n[mask].sum()
                '''
    
    # A --- Solving l0/l2 regularized problem
    qvec = Q.flatten(order='C')
    hvec = H.flatten(order='C')
    if tau >0:
        l0l2sol = L0L2sqlog_plus_linear(qvec=qvec, hvec=hvec, 
                                       tau=tau, nu=nu_A, epsilon=epsilon)
        avec = l0l2sol.get('x')
        l0norm = l0l2sol.get('l0norm')
    else:
        avec = (-hvec+ np.sqrt(hvec**2 + 4.*qvec*nu_A))/(2.*nu_A) 
        l0norm = (avec != 0).sum()       
    A[:,:] = avec.reshape(A.shape,order='C')   
    return A, l0norm, l0l2sol    

def update_A3(A, Q, H, types_of, event_types_idx,indices_of,qhist,nu_A, 
              tau, epsilon,decayfunc, eta, beta, Delta, ddtt):
    '''
    Updates A (impact matrix) in the Minorization-Maximization algorithm.
    This is an alternative version of update_A(). If the memory size is an 
    issue, use update_A() instead, but this update_A3() is a few times faster.
    
    The difference is that update_A3() uses a (N+1)*Nh matrix "types_of", 
    which stores the event type of each instance in the history, to remove the
    summation over n.     
    

    Parameters
    ----------
    A : 2D numpy array
        Impact matrix.
    Q : 2D numpy array
        Coefficient matrix of the optimization problem for A.
    H : 2D numpy array
        Coefficient matrix of the optimization problem for A.
    types_of : 2D numpy array
        types_of[n,:] is the list of the event type indices of n's history events.
        types_of[n,i] returns the event type (index) of the i-th history instance.
        of the n-th event instance.
    event_types_idx : 1D numpy array
        The event type sequence represented in terms of integer indices.
    indices_of : 1D numpy array
        Set of event instance indices belonging to each event type.
        All the instances of indices_of[1] have the same event type 1. 
    qhist : 2D numpy array
        NxNh matrix whose n-th row is the triggering probabilities of the n-th.
    nu_A : double
        L2 regularization strength for A.
    tau : double
        L0 regularization strength for A.
    epsilon : double
        Threshold parameter for the epsilon-sparsity theory.
    decayfunc : string
        Decay function name. Currently, power or exponential is allowed. 
    eta : double
        Exponent of the power distribution. Ignored when decayfunc=exp
    beta : 1D array
        Decay parameters.
    Delta : 2D numpy array
        Delta[n,i]=t(n)-t(i), where t(i) is in the history set of n.
    ddtt : 1D numpy array
        Array of t(n)-t(n-1) for n = 1, ..., N.

    Returns
    -------
    A : 2D numpy array
        Impact matrix.
    l0norm : double
        L0 norm of A.
    l0l2sol : dictionary
        Returned result of L0L2sqlog_plus_linear.

    '''

    for k in range(A.shape[0]):
        idx_of_k = indices_of[k]
        if 0 in idx_of_k:
            idx_of_k = exclude_zeroth_instance(idx_of_k.copy())
        
        # compute all the h_n,i values for n in idx_of_k
        Delta_k = Delta[idx_of_k,:]
        Delta1k = Delta_k - ddtt[idx_of_k,np.newaxis]
        H_k = decay_integral(beta[k],Delta_k,decayfunc,eta)\
            - decay_integral(beta[k],Delta1k,decayfunc,eta)
       
        for l in range(A.shape[1]):
            #mask = (types_of[idx_of_k,:]==l)
            mask = (types_of[idx_of_k,:]==l)
            
            Q[k,l] = qhist[idx_of_k,:][mask].sum()
            H[k,l] = H_k[mask].sum()
    
    # A --- Solving l0/l2 regularized problem
    qvec = Q.flatten(order='C')
    hvec = H.flatten(order='C')
    if tau >0:
        l0l2sol = L0L2sqlog_plus_linear(qvec=qvec, hvec=hvec, 
                                       tau=tau, nu=nu_A, epsilon=epsilon)
        avec = l0l2sol.get('x')
        l0norm = l0l2sol.get('l0norm')
    else:
        avec = (-hvec+ np.sqrt(hvec**2 + 4.*qvec*nu_A))/(2.*nu_A) 
        l0norm = (avec != 0).sum()       
    A[:,:] = avec.reshape(A.shape,order='C')   
    return A, l0norm, l0l2sol 

def L0L2sqlog_plus_linear(qvec,hvec,tau,nu,epsilon):  
    '''
    Solve sum_k(qk ln xk - hk xK -(1/2)nu xk**2) - tau ||x||_0.

    Parameters
    ----------
    qvec : 1D array
        qk of sum_k(qk ln xk - hk xK -(1/2)nu xk**2) - tau ||x||_0.
    hvec : 1D array
        hk of sum_k(qk ln xk - hk xK -(1/2)nu xk**2) - tau ||x||_0.
    tau : double
        L0 regularization strength
    nu : double
        L2 regularization strength.
    epsilon : double
        Sparsity parameter for the epsilon sparsity.

    Returns
    -------
    obj : dictionary
        l0norm : 
            The optimal l0 norm,
        x : 
            The solution of epsilon sparsity
        x_sparse :
            The solution translated zero-sparsity
        x_noL0 :
            The solution with tau = 0
    '''
    # Computing a_bar
    a_bar = ( -hvec + np.sqrt(hvec**2 + 4.*nu*qvec) )/(2*nu)
    a_bar[a_bar<0] =0 ##### ADDED ######
    x = a_bar.copy()
    
    # 
    mask0 = (a_bar != 0)
    a_bar1 = x[mask0]
    x1 = a_bar1.copy()
    qvec1 = qvec[mask0]
    hvec1 = hvec[mask0]
    
    # Computing the base objective value
    Phi_a_bar = qvec1*np.log(a_bar1) - hvec1*a_bar1 - (1/2)*nu*(a_bar1**2)
    if np.isnan(Phi_a_bar).sum() >0 or np.isinf(Phi_a_bar).sum() >0:
        print('L0L2sqlog_plus_linear: strange value in log')
    
    Phi_eps = qvec1*np.log(epsilon) - hvec1*epsilon - (1/2)*nu*(epsilon**2)
    gain_to_off = Phi_eps - Phi_a_bar + tau
    
    # Checking if 
    mask = (a_bar1 >= epsilon)&(gain_to_off > 0)
    x1[mask] = epsilon
    
    x[mask0] = x1
    
    l0norm = np.sum( x > epsilon )
    x_sparse = x.copy()
    x_sparse[x <= epsilon] = 0
    
    obj = {'l0norm':l0norm, 'x':x,'x_sparse':x_sparse,'x_noL0':a_bar}
    return obj

def update_q(qself,qhist,event_types_idx,mu,beta,Delta,eta,A,decayfunc):
    '''
    Update qself,qhist in the Minorization-Maximization algorithm

    Parameters
    ----------
    qself : 1D numpy array
        Self triggering probabilities. Note qself[0]=np.nan as the 0-th event
        is the genesis event. 
    qhist : 2D numpy array
        NxNh matrix whose n-th row is the triggering probabilities of the n-th.
    event_types_idx : TYPE
        Event type sequence represented in term of event type indices.
    mu : 1D numpy array
        Baseline intensities.
    beta : 1D numpy array
        Decay parameters.
    Delta : 2D numpy array
        Delta[n,i]=t(n)-t(i), where t(i) is in the history set of n.
    eta : double
        Exponent of the power distribution. Ignored when decayfunc=exp
    A : 2D numpy array
        Impact matrix.
    decayfunc : string
        Decay function name. Currently, power or exponential is allowed. 

    Returns
    -------
    qself : 1D numpy array
        Self triggering probabilities. Note qself[0]=np.nan as the 0-th event
        is the genesis event. 
    qhist : 2D numpy array
        NxNh matrix whose n-th row is the triggering probabilities of the n-th.

    '''
    N = len(event_types_idx) -1 
    Nh = qhist.shape[1]
    for n in range(1,N+1): # n=1,...N
        d_n = event_types_idx[n]
        mu_n = mu[d_n]
        A_n = A[d_n, event_types_idx[history_of(n,Nh)]]
        Delta_n = Delta[n,0:len(A_n)]
        beta_n = beta[d_n]
        decay_n = decay(beta_n, Delta_n,decayfunc,eta=eta)
        lambda_n = mu_n + (A_n * decay_n).sum()
        
        qself[n] = mu_n/lambda_n
        qhist[n,0:len(A_n)] = (A_n*decay_n)/lambda_n
    return qself,qhist

def initialize_mu(timestamps,event_types_idx,a_mu,b_mu):
    '''
    Initialize mu based on a naive Poisson model

    Parameters
    ----------
    timestamps : 1D numpy array
        timestamps of the event instances.
    event_types_idx : TYPE
        Event type sequence represented in term of event type indices.
    a_mu : double
        Shape parameter of the Gamma prior. Must not be less than 1.
    b_mu : double
        Rate parameter of the Gamma prior. Must be positive.

    Returns
    -------
    mu : 1D numpy array
        Baseline intensities.

    '''
    T = timestamps[-1] - timestamps[0]
    D = len(np.unique(event_types_idx))
    mu = np.zeros(D)
    for k in range(D):
        Nk = sum(event_types_idx==k)
        mu[k] = (Nk + a_mu - 1)/(T + b_mu)
    return mu

def initialize_A(timestamps,event_types_idx,nu_A,tau):
    '''
    Initialize A as the matrix of ones

    Parameters
    ----------
    timestamps : 1D numpy array
        timestamps of the event instances.
    event_types_idx : TYPE
        Event type sequence represented in term of event type indices.
    nu_A : double
        L2 regularization strength.
    tau : double
        L0 regularization strength.

    Returns
    -------
    A : 2D numpy array
        Initialized impact matrix.

    '''
    D = len(np.unique(event_types_idx))
    A = np.ones([D,D])
    return A

def initialize_q(timestamps,event_types_idx,Nh,decayfunc,beta,eta):
    '''
    Initialize triggering probabilities using the actual time stamps

    Parameters
    ----------
    timestamps : 1D numpy array
        timestamps of the event instances.
    event_types_idx : TYPE
        Event type sequence represented in term of event type indices.
    Nh : int
        Maximum number of event instances in the history.
    decayfunc : string
        Decay function name. Currently, power or exponential is allowed. 
    beta : 1D numpy array
        Decay parameters.
    eta : double
        Exponent of the power distribution. Ignored when decayfunc=exp

    Returns
    -------
    qself : 1D numpy array
        Self triggering probabilities. Note qself[0]=np.nan as the 0-th event
        is the genesis event. 
    qhist : 2D numpy array
        NxNh matrix whose n-th row is the triggering probabilities of the n-th

    '''
    N = len(timestamps)-1
    
    qself = np.ones(N+1)  # self-triggering prob.
    qself[0] = np.nan
    
    qhist = np.repeat(np.nan,(N+1)*Nh).reshape([N+1, Nh])

    for n in np.arange(1,N+1): # n=1,...,N
        tn = timestamps[n]
        dn = event_types_idx[n]
        histry_of_n = history_of(n,Nh)
        for idx in range(len(histry_of_n)):
            i = histry_of_n[idx]
            ti = timestamps[i]
            qhist[n,idx] = decay(beta[dn],tn-ti,decayfunc,eta=eta)
        rowsum = qself[n] + qhist[n,~np.isnan(qhist[n,:])].sum()
        qself[n] = qself[n]/rowsum
        qhist[n,:] = qhist[n,:]/rowsum
    return qself,qhist

def ln_gamma_sum(x,a,b):
    '''
    log likelihood of independent Gamma distributions sharing the same (a,b)

    Parameters
    ----------
    x : numpy array
        Random variable.
    a : double
        Shape parameter.
    b : double 
        Rate parameter.

    Returns
    -------
    loglik : double 
        Log likelihood.

    '''
    if a == 1:
        loglik = - b*x.sum()
    else:
        loglik = (a-1)*np.log(x).sum() - b*x.sum()
    return loglik
