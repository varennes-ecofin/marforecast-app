import numpy as np
import pandas as pd
import warnings
import scipy.stats as sts
from scipy.optimize import minimize, SR1
from sympy import Symbol, Poly, series, apart, fraction, expand

class stablemar:
    """
    Class representing various stable models
    
    Attributes:
        order: a vector (r, s) of orders of causal and noncausal lags resp.
        model (str): the model to be used, 'MAR', 'MARST', 'ARCMAR' or 'RWCMAR' (optional, defauts to 'MAR').
        par: a list of model parameters (optional, defauts to []).

    """
    
    def __init__(self, order, model = 'MAR', par = []):
        '''
        :order: a vector (r, s) of orders of causal and noncausal lags resp.
        :model: the model to be used, 'MAR', 'MARST', 'ARCMAR' or 'RWCMAR' (optional, defauts to 'MAR')
        '''
        
        self.order = order
        self.model = model
        self.par = par
        
        if model == "ARCMAR" or model == "RWCAR":
            warnings.warn("Deconvolution holds only for MAR(0,1)", UserWarning)    
        
    def fit(self, data, start, method = "free"):
        """
        Estimate the stable model 
        
        Arguments:
            data: a univariate numpy array.
            start: a numpy list of initial values for the optimisation problem.
            method: type of optimisation, 'free' (trust-constr, 3-point, and SR1) or 'bounded' (L-BFGS-B).

        """    
        self.method = method

        cfestep = 0.5
        cferange = max(abs(data))
        
        if self.model == "MAR" or self.model == "MARST":
            def optimfct(theta):
                return self.stablemle(data, theta)[0]  
        elif self.model == "ARCMAR":
            def optimfct(theta):
                return self.stablecfe(data, theta, cferange, cfestep, 1)  
        elif self.model == "RWCMAR":
            warnings.warn("Random Walk deconvolution is not implemented yet", UserWarning)
        
        if self.method == "bounded": # constrained estimation is recommended for MAR and MARST
            bounds = self.bndcheck(-0.99, 0.99)
            optimum = minimize(optimfct, start, bounds=bounds, method='L-BFGS-B')
        elif self.method == "free": # unconstrained estimation is recommended for ARCMAR
            callback4score.gradients = []
            optimum = minimize(optimfct, start, method = 'trust-constr', jac = '3-point', hess = SR1(),
                                options={'disp': False,
                                        'maxiter': 50,
                                        'barrier_tol': 1e-8,
                                        'initial_tr_radius': 1},
                                callback = callback4score)
            
        thetahat = optimum.x
        
        if self.model == "MAR" or self.model == "MARST":        
            ll, mut, vt, et = self.stablemle(data, thetahat)
            keys = ['Parameters','Likelihood','Trend','Bubble','Residuals']
            objects = [thetahat, ll, mut, vt, et]                
            results={}
            index=0
            for i in keys:
                results[i]=objects[index]
                index+=1  
                
        elif self.model == "ARCMAR":
            mindist = self.stablecfe(data, thetahat, cferange, cfestep, 1)
            keys = ['Parameters','Distance']
            objects = [thetahat, mindist]  
            results={}
            index=0
            for i in keys:
                results[i]=objects[index]
                index+=1    
                
        self.results = results
        return self     
    
    # Stable characteristic function based estimator for deconvolution
    def stablecfe(self, x, theta, rge, step, lam):
        """
        Evaluate the minimum distance objective function of stable/gaussian deconvolution problem
        
        Arguments:
            x: a univariate numpy array.
            theta: a numpy list of parameters
            rge: a scalar parameter indicating the scale of the data, typically, max(abs(x))
            step: an integer setting the distance integral approximation
            lam: a scalar tuning parameter modifying the integral grid (often set to 1) 

        """       
        distance = []
        n = len(x)
        x = x - np.median(x)

        for i in np.arange(-rge, rge + step, step):
            u = i
            v = lam * u
            
            distance.append(abs(self.tcf(theta,u,v) - self.ecf(u,v,x,n))**2)
            
        integral = sum(distance) # approximate the integral of the L2 distance (objectif function)
        
        return integral
    
    # Theoretical CF of convoluted Gaussian/Stable processes
    def tcf(self, theta, u, v): 
        
        # In this function, gamma stands for the stable scale parameter and sigma for the gaussian variance
        phi, alpha, beta, gamma, rho, sigma = theta
        
        # Gaussian component CF
        logxcf = - ((u * rho + v)**2) * (sigma**2)/(2*(1-rho**2))
        logecf = - (u**2) * (sigma**2)/2
        
        # Stable component CF
        if alpha == 1:
            logzcf = - gamma * (1 + beta * 1j * np.sign(u + v * phi) * (2 / np.pi) * np.log(abs(u + v * phi)))
            logvcf = - gamma * (1 + beta * 1j * np.sign(v) * (2 / np.pi) * np.log(abs(v))) * abs(v)
        else:
            logzcf = - (gamma**alpha) * (1 - beta * 1j * np.sign(u + v * phi) * np.tan(np.pi * alpha / 2))
            logvcf = - (gamma**alpha) * (1 - beta * 1j * np.sign(v) * np.tan(np.pi * alpha / 2)) * abs(v)
            
        logzcf = logzcf * (abs(u + v * phi)**(alpha)) / (1 - abs(phi))
        
        varphi = np.exp(logxcf + logecf + logzcf + logvcf)
        
        return varphi
    
    # Compute the Empirical Characteristic Function (ECF)
    def ecf(u, v, y, n): 
        
        y0 = y[1:]
        y1 = y[:-1]
        
        varphi = 1/n * np.sum(np.cos(u*y0 + v*y1) + 1j*np.sin(u*y0 + v*y1))
        
        return varphi
        
    # Stable MLE estimator for MAR and MARST model
    def stablemle(self, x, theta):
        
        n = len(x)
        r, s = self.order
        p = r + s
        
        psi, phi, alpha, beta, sigma, mu, rho, gamma = parcheck(theta, self.order, self.model)
                
        if self.model != "MARST":
            mu = 0
            gamma = 0 
            
        mut = np.zeros(n + 1)
        mut[0:1 + p] = x[0:1 + p]        
        
        et = np.zeros(n)
        vt = np.zeros(n)
        ut = np.zeros(n)
        ll0 = np.zeros(n - p)
        
        if p == 0:
            for i in np.arange(0, n):
                et[i] = x[i] - mut[i]
                xt = et[i] / sigma
                
                mut[i + 1] = mut[i] + mu + xt * gamma
                
                ll0[i] = sts.levy_stable.logpdf(et[i], alpha, beta=beta, loc=0, scale=sigma)
                
            ll = np.sum(ll0)
        else:
            for i in np.arange(p, n):
                vt[i] = x[i] - mut[i]
                ut[i - s] = vt[i - r] - np.sum(phi * vt[i - s + 1:i + 1]) 
                et[i - 1] = ut[i - s] - np.sum(psi * ut[i - p: i - s]) 
                xt = et[i - 1] / sigma
                
                mut[i + 1] = mut[i] + mu + xt * gamma
                
                ll0[i - p] = sts.levy_stable.logpdf(et[i - p], alpha, beta=beta, loc=0, scale=sigma)
                
            ll = np.sum(ll0)
            
        et = et[np.max([p - 1, 0]):-1]
        
        return -ll, mut, vt, et    
    
    def splitfit(self, data, start, K = 2, H = 2):  
        
        r, s = self.order
        gcovstart = start[0:r+s]
        def optimfct1(theta):
            return self.gcov(data, theta, r, s, K, H)[0]
        
        bounds = self.bndcheck(-0.99, 0.99, split = "AR")
        optimum = minimize(optimfct1, gcovstart, bounds=bounds, method='L-BFGS-B')
        marhat = optimum.x
                
        ls, ps = self.gcov(data, marhat, r, s, K, H)
        
        stablehat = self.snoisecfe(ps)
        
        thetahat = [*marhat, *stablehat]
        
        keys = ['Parameters','GStatistic','Pseudo']
        objects = [thetahat, ls, ps]                
        results={}
        index=0
        for i in keys:
            results[i]=objects[index]
            index+=1  
        
        self.results = results
        return self 
        
    def gcov(self, x, theta, r, s, K, H):
        
        n = x.shape[0]
        eps = np.zeros((n - 2, 2))
        psi = theta[0:r]
        phi = theta[r:r + s]
        pseudo = np.zeros(n)
        
        for t in range(r, n-s):
            # Backward filter
            pseudo[t] = x[t]
            for i in range(r):
                pseudo[t] -= psi[i] * x[t-i-1]
                
            # Forward filter
            for j in range(s):
                pseudo[t] -= phi[j] * x[t+j+1]
                
            # Cross filter
            for i in range(r):
                for j in range(s):
                    pseudo[t] += psi[i] * phi[j] * x[t-i+j]
                    
        pseudo = pseudo[r:(n - s)]
        kpowers = np.arange(1, K + 1)
        # eps = np.log(abs(pseudo[:, np.newaxis])) ** kpowers
        # eps = abs(pseudo[:, np.newaxis]) ** (1/kpowers)
        eps = pseudo[:, np.newaxis] ** kpowers
        eps = eps - np.mean(eps, axis=0)
        
        # 1/Gamma(0) computation
        n = eps.shape[0]
        vcv = np.dot(eps.T, eps) / n
        ivcv = np.linalg.inv(vcv)
        
        # Gamma(h) computation
        vcvmat = np.array([np.dot(eps[i:].T, eps[:-i]) / (n - i) for i in range(1, H + 1)])
        
        # Loss statistic L(theta) = sum(Tr(R^2)), with R^2 = cov * invv * cov.T * invv
        ls = np.sum([np.trace(np.linalg.multi_dot([h, ivcv, h.T, ivcv])) for h in vcvmat])
        
        return ls, pseudo
    
    def snoisecfe(self, data, step = 0.01, m = 1):
    
        n = len(data)
        u = np.arange(step, m + step, step)
        q75 = np.quantile(data, 0.75)
        q25 = np.quantile(data, 0.25)
        iqr = q75 - q25
        sigma0 = iqr/2
        data = data/sigma0 # normalize data
        
        ecf = np.zeros(len(u))
        ecf1 = np.zeros(len(u))
        ecf2 = np.zeros(len(u))
        
        for i, ui in enumerate(u):
            ecf1[i] = (1/n) * sum(np.cos(ui * data))
            ecf2[i] = (1/n) * sum(np.sin(ui * data))
            
        ecf = ecf1 + 1j * ecf2 
            
        x = np.log( - np.log(abs(ecf)))
        y = np.log(abs(u))
        parhat = np.polyfit(y, x, 1)
        
        alpha = parhat[0]
        sigma = np.exp(parhat[1]/alpha)
        if alpha != 1:
            eta = np.tan(np.pi * alpha / 2) * (abs(u) - abs(u)**alpha) #drop the sign(u) term as u>0
        else:
            eta = 2 / np.pi * u * np.log(abs(u))
        
        x = np.arctan2(ecf2, ecf1)
        y = - (sigma**alpha) * eta   
        parhat = np.polyfit(x, y, 1)
        beta = parhat[0]
        
        sigma = sigma0 * sigma # unnormalizase scale
        
        thetahat = [alpha, beta, sigma]
        thetahat = [float(x) for x in thetahat]
        
        return thetahat
    
    def simulate(self, n, par, errors = []):
        """
        Simulate various type of stable models
        
        Arguments:
            n: an integer sample size.
            par: a numpy list of model parameters (can be different from self.par).
            errors: a univariate numpy array of errors of size 3*n (optional, defauts to []).

        """          
        r, s = self.order # r := c order / s := nc order 
        p = r + s
        
        # psi are causal params and phi are noncausal ones
        
        psi, phi, alpha, beta, sigma, mu, rho, gamma = parcheck(par, self.order, self.model)   
        
        if len(errors) == 0 or len(errors) != n*3:
            esim = sts.levy_stable.rvs(alpha, beta, scale = sigma, size = n * 3)
        elif len(errors) == n*3:
            esim = errors  
            
        u0 = sts.levy_stable.rvs(alpha, beta, scale = sigma, size = 1) # nc part init
        
        m = 50
        xsim = np.ones(n + m)
        u = np.full(n * 3, u0)
        
        # Double filter methodology   
        # for Psi(L)Phi(F) y(t) = e(t), we define u(t):=Psi(L)y(t) and hence Phi(F)u(t)=e(t)
        # hence we simulate u(t) forward
        # and it follows that y(t) = 1/Psi(L)u(t) where 1/Psi(L)~sum_j^m(lambda_j)
        # where the lambda are obtained by a Laurent series development
        for t in range(n*3 - p - 1, n - s - m, -1):
            u[t] = esim[t] + np.sum(phi * u[t+1:t+s+1])
            
        if r>0:
            lam = laurentSeries(psi, m)
            for t in range(n, (n * 2 + m)):
                xsim[t - n - s]  = np.sum(lam * np.flip(u[t-m:t])) 
        else:
            for t in range(n, (n * 2 + m)):
                xsim[t - n - s]  = u[t] 
                
        xsim = xsim[:-m]    
        xsim = pd.Series(xsim)
        esim = pd.Series(esim[n: n * 2])
        self.secondary = pd.Series()
            
        if self.model == "MARST":
            st = esim[:s].tolist() 
            
            for time in range(s, len(esim)):
                u_t = mu + st[-1] + gamma*(esim[time - s]/sigma)
                st.append(u_t)    
                
            xsim = xsim + pd.Series(st)
            self.secondary = pd.Series(st)
            
        elif self.model == "ARCMAR":
            eps = np.random.normal(loc = 0, scale = gamma, size = n)
            arc = np.zeros(n)
            
            for t in range(1, n):
                arc[t] = rho * arc[t-1] + eps[t]  
                
            xsim = xsim + pd.Series(arc)
            self.secondary = pd.Series(arc)
            
        elif self.model == "RWCMAR":
            eps = np.random.normal(loc = 0, scale = gamma, size = n)
            rwc = np.zeros(n)
            
            for t in range(1, n):
                rwc[t] = rwc[t-1] + eps[t]  
                
            xsim = xsim + pd.Series(rwc)
            self.secondary = pd.Series(rwc)
            
        self.trajectory = xsim  
        self.innovation = esim     
        
        return self
    
    def generate(self, n, errors = []):
        """
        Simulate a stable MAR(r, s) model in a more efficient way than simulate (but restricted to MAR)
        self.par has to be defined before using this function
        
        Arguments:
            n: an integer sample size.
            errors: a univariate numpy array of errors of size 3*n (optional, defauts to []).

        """          
        alpha, beta, sigma = distcheck(self.par, self.order)
        
        m = 50
        ntilde = n + 2*m + 1
        deltas = self.mafilter(m)
        deltas = np.flip(deltas)
        
        if len(errors) == 0 or len(errors) != ntilde:
            esim = sts.levy_stable.rvs(alpha, beta, scale = sigma, size = n * 3)
        elif len(errors) == ntilde:
            esim = errors  
            
        xsim = np.ones(ntilde)*esim[0]
        
        for t in range(m, ntilde):
            xsim[t] = np.sum(deltas*esim[t-m:t+m])

        xsim = xsim[m:-m]    
        xsim = pd.Series(xsim)
        esim = pd.Series(esim[m:-m])
            
        self.trajectory = xsim  
        self.innovation = esim     

        return self
    
    def bndcheck(self, dn, up, split = "None"):
    
        r, s = self.order # r := c order / s := nc order 
        p = r + s
        
        if split == "None":
            bounds = [(dn, up) for _ in range(p)] # bounds for MAR parameters
            bounds += [(0.01, 1.99)] # bounds for alpha 
            bounds += [(-0.99, 0.99)] # bounds for beta 
            bounds += [(0.01, np.inf)] # bounds for sigma 
        elif split == "AR":
            bounds = [(dn, up) for _ in range(p)] # bounds for MAR parameters
        elif split == "Stable":
            bounds = [(0.01, 1.99)] # bounds for alpha 
            bounds += [(-0.99, 0.99)] # bounds for beta 
            bounds += [(0.01, np.inf)] # bounds for sigma
        
        if self.model == 'MARST':
            bounds += [(-np.inf, np.inf)] # bounds for mu
            bounds += [(0.01, np.inf)] # bounds for gamma
        elif self.model == 'ARCMAR':
            bounds += [(dn, up)] # bounds for rho
            bounds += [(0.01, np.inf)] # bounds for gamma
    
        return bounds
    
    def rndinit(self):
    
        r, s = self.order # r := c order / s := nc order 
        p = r + s
        arinit = sts.uniform.rvs(loc=0.1, scale=0.8, size=p)
        ainit = sts.uniform.rvs(loc=1.1, scale=0.8, size=1)
        binit = sts.uniform.rvs(loc=-0.2, scale=0.6, size=1)
        sinit = sts.uniform.rvs(loc=0.5, scale=1, size=1)
        
        if self.model == "ARCMAR":
            minit = 0
            rinit = sts.uniform.rvs(loc=0.1, scale=0.8, size=1)
            ginit = sts.uniform.rvs(loc=0.5, scale=1, size=1)
            name = np.array(["psi(B)"] * r + ["phi(F)"] * s + ["alpha", "beta", "sigma", "rho", "gamma"])
            init = [*arinit, *ainit, *binit, *sinit, *rinit, *ginit]
        elif self.model == "MARST":
            minit = sts.uniform.rvs(loc=0, scale=1, size=1)
            ginit = sts.uniform.rvs(loc=0.5, scale=1, size=1)
            name = np.array(["psi(B)"] * r + ["phi(F)"] * s + ["alpha", "beta", "sigma", "mu", "gamma"])
            init = [*arinit, *ainit, *binit, *sinit, *minit, *ginit]
        else:
            name = np.array(["psi(B)"] * r + ["phi(F)"] * s + ["alpha", "beta", "sigma"])
            init = [*arinit, *ainit, *binit, *sinit]
            
        dispinit = [(name, val) for name, val in zip(name, init)]
    
        return init, name, dispinit
    
    def mafilter(self, m):
        
        psi, phi = marcheck(self.par, self.order) # Return Psi (causal) and Phi (non-causal) parameters
        deltas = np.full(2*m, np.nan)
        
        for k in range(-m, m):
            deltas[k+m] = madelta(psi, phi, k) 
            
        deltas = np.flip(deltas) 
        
        return deltas
    
    def forecast(self, x, tau, m, h, k0, vartheta, trunc = 50):
        
        # Compute the pattern-based forecasts for a MAR(p,q), q>=2
        # INPUT:
        #     x: échantillon observé
        #     phi_nc: vecteur de 2 coefficients AR noncausaux
        #     t: indice de la dernière observation InSample
        #     m: profondeur temporelle du segment (Xt-m,...,Xt)
        #     k0: indice du coefficient MA dans la somme infinie
        #     h: horizon de prédiction (h = k0 + 1 normalement)  
        #     vartheta: -1 (bulle négative) ou 1 (bulle positive)
        
        if trunc <= m:
            trunc = 2*m
            
        dk = self.mafilter(trunc)
            
        dw = trunc - k0 - m
        up = trunc - k0 + h
        dkmh = dk[dw:up]             
        dsnorm = (vartheta * dkmh / np.sqrt(np.sum(dkmh[:m] ** 2))).T
        
        xmh = np.full(m+1, np.nan)
        for i in range(0, m + 1):
            xmh[i] = float(x.iloc[tau - i])
            
        xmh = np.flip(xmh)
        
        xfore = np.full(len(x)+h, np.nan)
        xfore[tau - m + 1:tau + h + 1] = dsnorm * np.sqrt(np.sum(xmh ** 2))
        
        return xfore
    
    def pathfinder(self, x, tau, m, kmax, vartheta, trunc = 50):
        
        # Recherche k0
        # INPUT:
        #     x: échantillon observé
        #     phi_nc: vecteur de coefficients AR noncausaux
        #     m: profondeur temporelle du segment (Xt-m,...,Xt)
        #     h: horizon de prédiction (0 normalement car plus tard on fixe h = k0 + 1?)
        #     A: Borélien sous forme d'un scalaire car augmenté dans la fonction à la dimension m+1
        #     tau: indice de la dernière observation InSample (important si l'échantillon x va au-delà)
        #     kmax: valeur max recherchée pour k0 (kmin étant fixé à 1)
        #     vartheta: -1 (bulle négative) ou 1 (bulle positive)
        
        x = x.to_numpy()
        xm = np.full(m + 1, np.nan)
        for i in range(0, m + 1):
            xm[i] = x[tau - i]
        xm = np.flip(xm)
        xsnorm = xm / np.sqrt(np.sum(xm[:m + 1] ** 2))
        
        ngA = np.full(kmax + 1, np.nan)
        dkm = np.full((kmax + 1, m + 1), np.nan)
        dsnorm = np.full((kmax + 1, m + 1), np.nan) 
        
        dk = self.mafilter(trunc)
        
        for k in range(kmax, -1, -1):
            dw = trunc - k - m - 1
            up = trunc - k
            dkm[k, :] = dk[dw:up]            
            dsnorm[k, :] = vartheta*(dkm[k, :] / np.sqrt(np.sum(dkm[k, :] ** 2)))
            
        ngA = np.sum(np.abs(xsnorm - dsnorm), axis=1) 
        k0 = np.argmin(ngA) + 1 # add 1 as index start at 0 in python
        return k0, ngA, dsnorm, xsnorm, xm, dkm
    
    def foreprob(self, x, k0, h, maxp, t, vartheta):
            
        # Compute the probability-based forecasts for a MAR(1,1)
        # INPUT:
        #     x: échantillon observé
        #     t: indice de la dernière observation InSample
        #     k0: indice du coefficient MA dans la somme infinie
        #     h: horizon de prédiction (h = k0 + 1 normalement)  
        #     maxp: above maxp, the crash prob is set to 1 when forecasting the path
        #     vartheta: -1 (bulle négative) ou 1 (bulle positive)
        # ETENDRE AUX MAR(r,1) : identique sauf pour h >= k dans ftraj
        
        phi_c, phi_nc = marcheck(self.par, self.order)
        alpha, _, _ = distcheck(self.par, self.order)
        
        proba = np.empty((h,4))
        ftraj = np.ones(h+1)
        
        for i in range(1,h+1):
            proba[i-1,0] = i
            proba[i-1,1] = (abs(phi_nc[0]) ** (alpha*(i-1)))*(1 - abs(phi_nc[0]) ** alpha)
            proba[i-1,2] = (abs(phi_nc[0]) ** (alpha*(i)))
            proba[i-1,3] = 1 - (abs(phi_nc[0]) ** (alpha*(i)))
            
        for i in range(1, h+1):
            if proba[i-1,3] < maxp:
                ftraj[i] = ftraj[i-1]*(1/abs(phi_nc[0]))
            else:
                ftraj[i] = ftraj[i-1]*abs(phi_c[0])
        
        ftraj = ftraj*abs(x.iloc[t])*vartheta #*abs(x.iloc[t])
        
        proba[0,1] = np.nan
        proba[0:k0-1,2:4] = np.nan
        dfprob = pd.DataFrame(proba, columns=['h', 'Crash Before h', 'Survive at h', 'Crash at h'])
        dfprob.set_index('h', inplace = True)
        
        dftraj = np.full(len(x)+h, np.nan)
        dftraj[t:t + h + 1] = ftraj
        
        return dfprob, dftraj
        
# Package tool

def madelta(cvec, ncvec, k):
    """
    Compute the infinite MA coefficient at lag k on any side of the sum, 
    for cvec and ncvec, vectors or causal and noncausal AR coefficients.

    Args:
    cvec: vector of causal AR coefficients.
    zeta: vector of noncausal AR coefficients.
    k: a positive or negative integer.

    Returns:
    The MA coefficient delta at lag k.
    """
    if cvec != 0:
        lam = 1 / np.roots(np.flip(np.concatenate(([1], -np.array(cvec)))))
        lam = lam.real # lambda values for the causal part
        r = len(lam)
    else:
        lam = 0
        r = 0

    if ncvec != 0:      
        zeta = 1 / np.roots(np.flip(np.concatenate(([1], -np.array(ncvec)))))
        zeta = zeta.real # lambda values for the noncausal part (denoted zeta)
        s = len(zeta)
    else:
        zeta = 0
        s = 0

    delta = 0

    if k>=0:
        for j in range(s):
            numerator = zeta[j] ** ((s - 1) + k)
            denominator1 = 1 if s == 1 else np.prod([zeta[j] - zeta[i] for i in range(s) if i != j])
            denominator2 = np.prod([zeta[j] * lam[i] - 1 for i in range(r)])

            if s % 2 == 0: # Multiply by -1 if s is even
                delta -= numerator / (denominator1 * denominator2)  
            else: # Multiply by 1 if s is odd
                delta += numerator / (denominator1 * denominator2)
                
    else:
        for j in range(r):
            numerator = lam[j] ** ((r - 1) - k)
            denominator1 = 1 if r == 1 else np.prod([lam[j] - lam[i] for i in range(r) if i != j])
            denominator2 = np.prod([lam[j] * zeta[i] - 1 for i in range(s)])

            if r % 2 == 0:
                delta -= numerator / (denominator1 * denominator2)  
            else:
                delta += numerator / (denominator1 * denominator2) 
                
    if (r % 2 != 0) and (s % 2 != 0):
        delta = -delta
    elif (r % 2 == 0) and (s % 2 == 0):
        delta = -delta
        
    return delta

def pfd(cvec, ncvec):
    """Perform Partial Fraction Decomposition of MAR(r,s) transfert function and extract coefficients"""  

    s = len(ncvec)
    L = Symbol('L')

    Phi = 1 # WARNING : in this function, Phi is the causal polynomial
    Psi = 1 # WARNING : in this function, Psi is the non-causal polynomial
    for i, coef in enumerate(cvec):
        Phi -= coef * L**(i + 1)  

    for i, coef in enumerate(ncvec):
        Psi -= coef * L**(-(i + 1))
        
    # MAR Transfert function
    tfunc = 1 / (Phi * (L**s) * (Psi)) #(L**s) * 
    
    pfdpoly = apart(tfunc)
    
    def getbcoeffs(bpoly):

        num, den = fraction(bpoly)
        num = expand(num)
        b = Poly(num, L).all_coeffs()
        d = Poly(den, L).all_coeffs()
        
        return b, d
                
    b1, d1 = getbcoeffs(pfdpoly.args[0])  
    b2, _ = getbcoeffs(pfdpoly.args[1])

    phi4b1 = b1
    psi4b2 = b2
                
    for dval in d1:
        for cval in ncvec:
            if abs(dval) == abs(cval):
                phi4b1 = b2
                psi4b2 = b1
                print('b1 and b2 reassigned')
                return np.flip(phi4b1), np.flip(psi4b2), pfdpoly

    return np.flip(phi4b1), np.flip(psi4b2), pfdpoly

def laurentSeries(arvec, m):
    """
    Approximate 1/Phi(L) or 1/Psi by a Laurent series development.

    Args:
    coefficients: coefficients of the polynomial, the last term is the constant 1.
    m: order of truncation of the infinit sum.

    Returns:
    the Laurent series coefficients
    """
    arvec = -np.array(arvec) # opposite sign in the polynomial
    polycoeff = arvec[::-1] # flip the polynomial coefficients in view of Poly fct
    polycoeff = np.append(polycoeff, 1) # add the constant 1 in the polynomial
    
    # Define the symbolic polynomial 
    L = Symbol('L')
    PInv = 1 / Poly(polycoeff, L)

    # Laurent Series symbolic computation
    ls = series(PInv, L, x0=0, n=m)

    # Conversion to polynomial expression
    polyseries = ls.as_poly()

    # Extract the Laurent coefficients
    lc = polyseries.coeffs()
    if len(lc)>m:
        lc = lc[:m]

    # Flip the Laurent coefficients (from j = 0 to m)
    return lc[::-1]

def rootcheck(arvec):
    
    poly = np.insert(-np.array(arvec),0,1)
    roots = np.roots(poly[::-1])
    lams = 1/roots
    
    circle = any(abs(root) <= 1 for root in roots)
        
    return circle, roots, lams

def callback4score(x, state):
    score = state.grad
    callback4score.gradients.append(score)

    return False

def marcheck(par, order):
    
    r, s = order # r := c order / s := nc order 
    p = r + s

    if r>0:
        psi = par[0:r] # Psi are causal parameters
        # circle, roots, _ = rootcheck(psi)
        # if circle:
        #     warnings.warn("At least one causal root of Psi lies in the unit circle", UserWarning)
        #     #print("Roots are",roots)
    else:
        psi = 0

    if s>0:
        phi = par[r: p] # Phi are non-causal parameters
        # circle, roots, _ = rootcheck(phi)
        # if circle:
        #     warnings.warn("At least one non-causal root of Phi lies in the unit circle", UserWarning)
        #     #print("Roots are",roots)
    else:
        phi = 0     

    return psi, phi
    
def distcheck(par, order):
    
    r, s = order # r := c order / s := nc order 
    p = r + s

    alpha = par[p]
    beta = par[p + 1]
    sigma = par[p + 2] 

    return alpha, beta, sigma

def modelcheck(par, order, model):
    
    mu = 0
    rho = 0
    gamma = 0

    r, s = order # r := c order / s := nc order 
    p = r + s

    if model == "MAR":
        if len(par) < p + 3:
            warnings.warn(("MAR(r,s) model requires only r+s+3 parameters."
                        "The number of parameters is not good. A Cauchy white noise as been generated"), UserWarning)

    if model == "MARST":
        mu = par[-2] # drift in the ST equation
        gamma = par[-1] # scale parameter in the ST equation
        if len(par) < p + 5:
            warnings.warn(("MARST model requires two additional parameters."
                            "As the number of parameters is not good, only the MAR part has been considered"), UserWarning)
            mu = 0
            gamma = 0
        
    if model == "ARCMAR":
        rho = par[-2] # autogressive parameter in the AR(1) Gaussian equation
        gamma = par[-1] # variance parameter in the AR(1) Gaussian equation
        if len(par) < p + 5 or s > 1 or r > 1:
            warnings.warn(("ARCMAR model requires two additional parameters and exists only for MAR(0,1)."
                            "As the number of parameters is not good, only the MAR part has been considered"), UserWarning)
            rho = 0
            gamma = 0
        
    if model == "RWCMAR":
        gamma = par[-1] # variance parameter in the AR(1) Gaussian equation
        if len(par) < p + 4 or s > 1 or r > 1:
            warnings.warn(("RWCMAR model requires one additional parameter."
                            "As the number of parameters is not good, only the MAR part has been considered"), UserWarning)
            mu = 0
            gamma = 0
        
    return mu, rho, gamma

def parcheck(par, order, model):

    psi, phi = marcheck(par, order)
    alpha, beta, sigma = distcheck(par, order)
    mu, rho, gamma = modelcheck(par, order, model)

    return psi, phi, alpha, beta, sigma, mu, rho, gamma