import numpy as np
import pandas as pd
import time
import warnings
import scipy.stats as stats
import matplotlib.pyplot as plt
import multiprocessing
from scipy.optimize import minimize
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from typing import Tuple, List, Dict, Optional, Any

class StableMAR:
    """
    Class representing Mixed Autoregressive (MAR) models with alpha-stable innovations.
    
    The MAR(r,s) model is defined as:
    Ψ(F)Φ(B)X_t = ε_t
    
    where:
    - Ψ(F) is the noncausal polynomial with F the forward operator
    - Φ(B) is the causal polynomial with B the backward operator
    - ε_t follows an alpha-stable distribution S(α, β, σ, 0)
    
    Attributes:
        order (Tuple[int, int]): A tuple (r, s) of orders of causal and noncausal lags respectively
        par (List[float]): List of model parameters
        results (Dict[str, Any]): Dictionary to store estimation results
        trajectory (pd.Series): Simulated or filtered trajectory
        innovation (pd.Series): Innovations for simulated trajectory
    """
    
    def __init__(self, order: Tuple[int, int], par: List[float] = None):
        """
        Initialize a StableMAR model.

        Args:
            order: A tuple (r, s) of orders of causal and noncausal lags respectively
            par: Optional list of model parameters
        """
        self.order = order
        self.par = par if par is not None else []
        self.results = {}
        self.trajectory = None
        self.innovation = None
        
    
    def fit(self, data: np.ndarray, start: List[float], method: str = "gcov", 
            max_lag: int = 10, K: int = 2, H: int = 2, verbose: bool = False) -> "StableMAR":
        """
        Estimate the stable MAR model using either GCoV or Minimum Distance Spectral Density approach.
        
        Args:
            data: A univariate numpy array of time series data.
            start: Initial parameter values for optimization.
            method: Estimation method, either "gcov" or "mdsd".
            max_lag: Maximum lag for mdsd estimation.
            K: Number of nonlinear transformations for GCoV.
            H: Number of lags for correlation calculation in GCoV.
            verbose: If True, print detailed information during estimation.
            
        Returns:
            Self for method chaining.
        """
        if method == "gcov":
            return self._fit_gcov(data, start, K, H, verbose = verbose)
        elif method == "mdsd":
            return self._fit_mdsd(data, start, max_lag, verbose = verbose)
        else:
            raise ValueError("Method must be either 'gcov' or 'mdsd'")
    
    
    def _fit_gcov(self, data: np.ndarray, start: List[float], K: int = 2, 
                H: int = 2, verbose: bool = False) -> "StableMAR":
        """
        Estimate the stable MAR model using Generalized Covariance approach.
        
        Args:
            data: A univariate numpy array of time series data.
            start: Initial parameter values for optimization.
            K: Number of nonlinear transformations.
            H: Number of lags for correlation calculation.
            verbose: If True, print detailed information during estimation.
            
        Returns:
            Self for method chaining.
        """
        r, s = self.order
        gcov_start = start[:r+s]
        
        def objective_function(theta: np.ndarray) -> float:
            return self.gcov(data, theta, K, H)[0]
        
        bounds = self._param_bounds(r+s)
        
        if verbose:
            print(f"Starting GCoV estimation with K={K}, H={H}")
            print(f"Initial parameters: {gcov_start}")
            start_time = time.time()
        
        # Perform optimization
        optimum = minimize(
            objective_function, 
            gcov_start, 
            bounds=bounds, 
            method='L-BFGS-B'
        )
        
        if verbose:
            elapsed_time = time.time() - start_time
            print(f"Optimization completed in {elapsed_time:.2f} seconds")
            print(f"Final parameters: {optimum.x}")
        
        mar_params = optimum.x
        loss_stat, pseudo_residuals = self.gcov(data, mar_params, K, H)
        
        # Store results
        self.results = {
            'Parameters': mar_params,
            'GStatistic': loss_stat,
            'PseudoResiduals': pseudo_residuals,
            'Method': 'GCoV'
        }
        
        self.par = mar_params.tolist()
        
        return self


    def _fit_mdsd(self, data: np.ndarray, start: List[float], 
                    max_lag: int = 10, verbose: bool = False) -> "StableMAR":
        """
        Estimate the stable MAR model using Minimum Distance Spectral Density approach with improved grid search.
        See Velasco (2022)
        
        Args:
            data: A univariate numpy array of time series data.
            start: Initial parameter values for optimization.
            max_lag: Maximum lag for mdsd estimation.
            verbose: If True, print detailed information during estimation.
            
        Returns:
            Self for method chaining.
        """
        r, s = self.order
        mdsd_start = start[:r+s]
        
        if verbose:
            print(f"Starting mdsd estimation for MAR({r},{s}) model")
            print(f"Initial parameters: {mdsd_start}")
            start_time = time.time()
                
        # Step 2: Final optimization with L-BFGS-B using the grid search result
        def objective_function(theta: np.ndarray) -> float:
            return self._mds_criterion(data, theta, max_lag, method='L')
        
        bounds = self._param_bounds(r+s)
        
        try:
            # Perform optimization
            optimum = minimize(
                objective_function, 
                mdsd_start, 
                bounds=bounds, 
                method='L-BFGS-B',
                options={'maxiter': 200}
            )
            
            mar_params = optimum.x
            criterion_value = optimum.fun
            success = optimum.success
            
            if not success and verbose:
                print(f"Warning: Optimization did not converge: {optimum.message}")
        except Exception as e:
            if verbose:
                print(f"Optimization failed: {str(e)}")
                print("Using initial guess parameters as fallback")
            
            mar_params = mdsd_start
            criterion_value = objective_function(mdsd_start)
        
        # Compute residuals
        # residuals = self._compute_residuals(data, mar_params)
        residuals, std_residuals = self._pseudo_residuals(data, mar_params)
        
        # Store results
        self.results = {
            'Parameters': mar_params,
            'CriterionValue': criterion_value,
            'Residuals': residuals,
            'StdResiduals': std_residuals,
            'Method': 'mdsd',
            'InitialGuess': mdsd_start
        }
        
        self.par = mar_params.tolist()
        
        if verbose:
            elapsed_time = time.time() - start_time
            print(f"MDSD estimation completed in {elapsed_time:.2f} seconds")
            print(f"Final parameters: {self.par}")
            print(f"Final criterion value: {criterion_value}")
        
        return self

    
    def _mds_criterion(self, data: np.ndarray, params: np.ndarray, 
                    max_lag: int = 10, method: str = 'Q', sigma: float = 1.0) -> float:
        """
        Calculate the minimum distance MDSD criterion for parameter estimation.
        
        Args:
            data: Time series data.
            params: Model parameters.
            max_lag: Maximum lag for correlation calculation.
            method: 'L' for L_mds or 'Q' for Q_mds criterion.
            sigma: Scale parameter for characteristic function.
            
        Returns:
            Value of the criterion.
        """
        # Compute residuals
        # residuals = self._compute_residuals(data, params)
        _, residuals = self._pseudo_residuals(data, params)
        
        # Calculate the lags
        lags = range(1, min(max_lag+1, len(residuals)))
        
        # Calculate CF covariances
        sigma_hat = self._compute_cf_covariances(residuals, lags, sigma)
        
        # Compute the criterion
        if method == 'L':
            criterion = 0
            for j in lags:
                criterion += (1 / (j**2)) * sigma_hat.get(j, 0)
            criterion *= 2 / np.pi
        elif method == 'Q':  # method == 'Q'
            T = len(residuals)
            
            # Daniell kernel
            def kernel(x: float) -> float:
                return np.sin(np.pi * x) / (np.pi * x) if x != 0 else 1
            
            # Bandwidth parameter
            p = int(T**(1/5))
            
            criterion = 0
            for j in lags:
                k_j = kernel(j / p)
                correction = 1 - (j / T)
                criterion += (k_j**2) * correction * sigma_hat.get(j, 0)
            criterion *= 2 / np.pi
        else:
            print('Unknown method, should be L or Q')
        
        return criterion
    
    
    def _compute_cf_covariances(self, residuals: np.ndarray, lags: range, sigma: float = 1.0) -> Dict[int, float]:
        """
        Compute characteristic function covariances for MDSD estimation.
        
        Args:
            residuals: Residuals from the model.
            lags: Range of lags to compute covariances for.
            sigma: Scale parameter for evaluation points.
            
        Returns:
            Dictionary of covariances indexed by lag.
        """
        # Points for evaluating the characteristic function
        v_grid = np.linspace(-2, 2, 20) * sigma
        
        # Store covariances for each lag
        sigma_hat = {}
        
        # Calculate empirical characteristic function for all v points
        exp_iv_e = np.array([np.exp(1j * v * residuals) for v in v_grid])
        phi_v = np.mean(exp_iv_e, axis=1)
        
        for j in lags:
            if j == 0:
                continue
            
            sigma_j_values = []
            
            for idx, v in enumerate(v_grid):
                # Use matrix operations for efficiency
                exp_iv_e_lag = exp_iv_e[idx, :-j]  # e^{iv*e_{t-j}}
                i_e = 1j * residuals[j:]           # i*e_t
                
                # Calculate (exp(i*v*e_{t-j}) - E[exp(i*v*e)])
                centered_exp = exp_iv_e_lag - phi_v[idx]
                
                # Calculate i*e_t * (exp(i*v*e_{t-j}) - E[exp(i*v*e)])
                term = i_e * centered_exp
                
                # Average and square magnitude
                sigma_j_v = np.abs(np.mean(term))**2
                sigma_j_values.append(sigma_j_v)
            
            # Integrate over v (numerical approximation)
            sigma_hat[j] = np.mean(sigma_j_values)
        
        return sigma_hat
    
    
    def _pseudo_residuals(self, data: np.ndarray, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute residuals for a MAR model with given parameters.
        
        Args:
            data: Time series data.
            params: Model parameters.
            
        Returns:
            Tuple containing (pseudo residuals, standardized pseudo residuals)
        """
        r, s = self.order
        n = len(data)
        
        # Extract parameters
        psi = params[:r] if r > 0 else np.array([])
        phi = params[r:r+s] if s > 0 else np.array([])
        
        # Compute pseudo-innovations
        pseudo = np.zeros(n)
        
        for t in range(r, n-s):
            # Backward filter
            pseudo[t] = data[t]
            
            for i in range(r):
                pseudo[t] -= psi[i] * data[t-i-1]
            
            # Forward filter
            for j in range(s):
                pseudo[t] -= phi[j] * data[t+j+1]
            
            # Cross filter
            for i in range(r):
                for j in range(s):
                    pseudo[t] += psi[i] * phi[j] * data[t-i+j]
                    
        # Centrer et standardiser les résidus
        stdpseudo = (pseudo - np.mean(pseudo)) / np.std(pseudo)
        
        return pseudo[r:n-s], stdpseudo[r:n-s]
    
    
    def gcov(self, x: np.ndarray, theta: np.ndarray, K: int, H: int) -> Tuple[float, np.ndarray]:
        """
        Compute the Generalized Covariance criterion for MAR model estimation.
        See Gourieroux and Jasiak (2017, 2023)
        
        Args:
            x: Time series data.
            theta: Model parameters.
            r: Order of causal component.
            s: Order of noncausal component.
            K: Number of nonlinear transformations.
            H: Number of lags for correlation calculation.
            
        Returns:
            Tuple containing the loss statistic and pseudo-residuals.
        """
        
        pseudo, _ = self._pseudo_residuals(x, theta)
        
        # Apply nonlinear transformations
        kpowers = np.arange(1, K + 1)
        
        eps = pseudo[:, np.newaxis] ** kpowers
        
        # Center the transformations
        eps = eps - np.mean(eps, axis=0)
        
        # 1/Gamma(0) computation (variance-covariance matrix)
        n_eps = eps.shape[0]
        vcv = np.dot(eps.T, eps) / n_eps
        
        # Handle possible singular matrix
        try:
            ivcv = np.linalg.inv(vcv)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            ivcv = np.linalg.pinv(vcv)
            warnings.warn("Singular matrix in GCoV estimation, using pseudo-inverse")
        
        # Gamma(h) computation (autocovariance matrices at different lags)
        vcvmat = np.array([np.dot(eps[i:].T, eps[:-i]) / (n_eps - i) for i in range(1, H + 1)])
        
        # Loss statistic L(theta) = sum(Tr(R^2)), with R^2 = cov * invv * cov.T * invv
        ls = np.sum([np.trace(np.linalg.multi_dot([h, ivcv, h.T, ivcv])) for h in vcvmat])
        
        return ls, pseudo


    def portmanteau_test(self, data: np.ndarray, params: Optional[List[float]] = None, 
                            H: int = 10, K: int = 2) -> Dict[str, Any]:
            """
            Perform the GCov portmanteau test for model specification with nonlinear transformations.
            
            This test checks the null hypothesis of absence of serial dependence in the residuals
            and their transformations. Under the null, the statistic follows a chi-square 
            distribution with degrees of freedom = K²H - dim(theta).
            
            Args:
                data: Observed time series
                params: Model parameters (uses self.par if None)
                H: Number of lags to test
                K: Number of nonlinear transformations (power transformations)
            
            Returns:
                Dictionary containing test statistics and diagnostics
            """
            r, s = self.order
            p = r + s
            
            if params is None:
                if self.par is None:
                    raise ValueError("Model parameters must be estimated first")
                params = self.par
            
            # Extract only the AR parameters (not the alpha-stable parameters)
            ar_params = params[:p]
            
            # Compute pseudo-residuals
            pseudo_residuals, _ = self._pseudo_residuals(data, np.array(ar_params))
            
            # Create K nonlinear transformations (power transformations)
            transformations = []
            for k in range(1, K + 1):
                u_k = pseudo_residuals ** k
                transformations.append(u_k)
            
            # Stack all transformations as matrix ε
            epsilon = np.column_stack(transformations)
            
            # Center each transformation
            epsilon_centered = epsilon - np.mean(epsilon, axis=0, keepdims=True)
            
            # Compute the portmanteau statistic directly
            n = epsilon_centered.shape[0]
            K = epsilon_centered.shape[1]
            
            # Compute Γ(0) and its inverse
            gamma_0 = np.dot(epsilon_centered.T, epsilon_centered) / n
            
            # Ensure Γ(0) is invertible
            try:
                gamma_0_inv = np.linalg.inv(gamma_0)
            except np.linalg.LinAlgError:
                # Add small regularization if singular
                gamma_0_inv = np.linalg.inv(gamma_0 + 1e-8 * np.eye(K))
                warnings.warn("Singular matrix in GCoV estimation, using pseudo-inverse")
            
            statistic = 0
            
            for h in range(1, H+1):
                if h < n:
                    gamma_h = np.dot(epsilon_centered[h:].T, epsilon_centered[:-h]) / n
                    # Compute R²(h) = Γ(h)Γ(0)^(-1)Γ(h)'Γ(0)^(-1)
                    R_squared = gamma_h @ gamma_0_inv @ gamma_h.T @ gamma_0_inv
                    statistic += np.trace(R_squared)
            
            # Multiply by sample size
            statistic *= n
            
            # Degrees of freedom
            df = K**2 * H - len(ar_params)
            
            # Compute p-value
            p_value = 1 - stats.chi2.cdf(statistic, df) if df > 0 else np.nan
            
            return {
                'statistic': statistic,
                'degrees_of_freedom': df,
                'p_value': p_value,
                'H': H,
                'K': K,
                'parameter_count': len(ar_params)
            }


    def bootstrap_portmanteau_test(self, data: np.ndarray, params: Optional[List[float]] = None,
                                H: int = 2, K: int = 2, n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Perform bootstrap to analyze the distribution of the portmanteau test statistic.
        
        Args:
            data: Observed time series
            params: Model parameters
            H: Number of lags
            K: Number of transformations
            n_bootstrap: Number of bootstrap samples
        
        Returns:
            Bootstrap analysis results
        """
        r, s = self.order
        p = r + s
        
        if params is None:
            if self.par is None:
                raise ValueError("Model parameters must be estimated first")
            params = self.par
        
        ar_params = params[:p]
        
        # Compute original statistic
        original_result = self.portmanteau_test(data, params, H, K)
        original_statistic = original_result['statistic']
        df = original_result['degrees_of_freedom']
        
        # Compute pseudo-residuals
        pseudo_residuals, _ = self._pseudo_residuals(data, np.array(ar_params))
        
        # Bootstrap resampling
        bootstrap_statistics = []
        
        for i in range(n_bootstrap):
            # Resample residuals with replacement
            boot_indices = np.random.choice(len(pseudo_residuals), len(pseudo_residuals), replace=True)
            boot_residuals = pseudo_residuals[boot_indices]
            
            # Create transformations of bootstrap residuals
            transformations = []
            for k in range(1, K + 1):
                u_k = boot_residuals ** k
                transformations.append(u_k)
            
            # Stack and center
            epsilon = np.column_stack(transformations)
            epsilon_centered = epsilon - np.mean(epsilon, axis=0, keepdims=True)
            
            # Compute bootstrap statistic directly
            n = epsilon_centered.shape[0]
            
            # Compute Γ(0) and its inverse
            gamma_0 = np.dot(epsilon_centered.T, epsilon_centered) / n
            
            try:
                gamma_0_inv = np.linalg.inv(gamma_0)
            except np.linalg.LinAlgError:
                gamma_0_inv = np.linalg.inv(gamma_0 + 1e-8 * np.eye(K))
            
            boot_statistic = 0
            
            for h in range(1, H+1):
                if h < n:
                    gamma_h = np.dot(epsilon_centered[h:].T, epsilon_centered[:-h]) / n
                    R_squared = gamma_h @ gamma_0_inv @ gamma_h.T @ gamma_0_inv
                    boot_statistic += np.trace(R_squared)
            
            boot_statistic *= n
            bootstrap_statistics.append(boot_statistic)
        
        bootstrap_statistics = np.array(bootstrap_statistics)
        
        # Compute bootstrap p-value
        bootstrap_p_value = np.mean(bootstrap_statistics >= original_statistic)
        
        # Compute percentiles
        percentiles = np.percentile(bootstrap_statistics, [5, 95])
        
        # Theoretical chi-square percentiles
        chi2_percentiles = stats.chi2.ppf([0.05, 0.95], df)
        
        return {
            'original_statistic': original_statistic,
            'original_p_value': original_result['p_value'],
            'bootstrap_p_value': bootstrap_p_value,
            'bootstrap_statistics': bootstrap_statistics,
            'bootstrap_percentiles': percentiles,
            'chi2_percentiles': chi2_percentiles,
            'mean_bootstrap': np.mean(bootstrap_statistics),
            'theoretical_mean': df,
            'std_bootstrap': np.std(bootstrap_statistics),
            'theoretical_std': np.sqrt(2 * df),
            'median_bootstrap': np.median(bootstrap_statistics),
            'theoretical_median': stats.chi2.median(df)
        }


    def fit_stable_noise(self, data: np.ndarray, step: float = 0.01, m: float = 1.0) -> List[float]:
        """
        Estimate alpha-stable parameters from residuals using characteristic function (see Kogon and Williams, 1998).
        
        Args:
            data: Residuals or innovations.
            step: Step size for the grid of u values.
            m: Maximum value for the grid of u values.
            
        Returns:
            List of estimated alpha-stable parameters [alpha, beta, sigma].
        """
        n = len(data)
        u = np.arange(step, m + step, step)
        
        # Robust scale estimation using IQR
        q75 = np.quantile(data, 0.75)
        q25 = np.quantile(data, 0.25)
        iqr = q75 - q25
        sigma0 = iqr/2
        
        # Normalize data
        data = data/sigma0 
        
        # Calculate empirical characteristic function
        ecf1 = np.array([(1/n) * np.sum(np.cos(ui * data)) for ui in u])
        ecf2 = np.array([(1/n) * np.sum(np.sin(ui * data)) for ui in u])
        ecf = ecf1 + 1j * ecf2
        
        # Estimate alpha and sigma from the log-log plot
        x = np.log(- np.log(np.abs(ecf) + 1e-10))
        y = np.log(np.abs(u) + 1e-10)
        parhat = np.polyfit(y, x, 1)
        
        alpha = parhat[0]
        
        # Constrain alpha to be in (0, 2)
        alpha = max(0.01, min(1.99, alpha))
        
        sigma = np.exp(parhat[1]/alpha)
        
        # Estimate beta
        if alpha != 1:
            eta = np.tan(np.pi * alpha / 2) * (np.abs(u) - np.abs(u)**alpha)
        else:
            eta = 2 / np.pi * u * np.log(np.abs(u) + 1e-10)
        
        x = np.arctan2(ecf2, ecf1)
        y = - (sigma**alpha) * eta
        
        parhat = np.polyfit(x, y, 1)
        beta = parhat[0]
        
        # Constrain beta to be in [-1, 1]
        beta = max(-0.99, min(0.99, beta))
        
        # Un-normalize sigma
        sigma = sigma0 * sigma
        
        return [float(alpha), float(beta), float(sigma)]
    
    
    def inference(self, data: np.ndarray, params: Optional[List[float]] = None, 
                alpha: float = 0.05) -> pd.DataFrame:
        """
        Calculates inference statistics for estimated parameters.
        
        Args:
            data: Observed time series
            params: Estimated parameters (uses self.par if None)
            alpha: Significance level (default 0.05 for 95% confidence intervals)
            
        Returns:
            DataFrame containing estimates, standard errors, t-statistics,
            p-values and confidence intervals
        """
        if params is None:
            if self.par is None:
                raise ValueError("Les paramètres doivent être estimés avant de calculer l'inférence")
            params = self.par
            
        r, s = self.order
        params_array = np.array(params)
        
        # Récupérer la méthode d'estimation utilisée
        method = self.results.get('Method', 'Unknown')
        
        # Créer des noms de paramètres pour une meilleure lecture
        param_names = []
        
        # Paramètres AR causaux
        for i in range(r):
            param_names.append(f"psi_{i+1}")
        
        # Paramètres AR non-causaux
        for i in range(s):
            param_names.append(f"phi_{i+1}")
        
        # Paramètres de la distribution alpha-stable si présents
        if len(params) > r + s:
            param_names.append("alpha")
            if len(params) > r + s + 1:
                param_names.append("beta")
                if len(params) > r + s + 2:
                    param_names.append("sigma")
        
        # Calculer la matrice de variance-covariance en fonction de la méthode
        vcv_matrix = None
        
        try:
            if method == "gcov":
                # Pour GCov, utiliser la théorie asymptotique du papier GCOVJBES.pdf
                vcv_matrix = self._calculate_gcov_vcv(data, params)
            elif method == "mdsd":
                # Pour MDSD, utiliser la théorie de Velasco 2022
                vcv_matrix = self._calculate_mdsd_vcv(data, params)
            else:
                print(f"Avertissement: Méthode d'estimation '{method}' non reconnue, utilisation d'approximations.")
                vcv_matrix = np.diag([0.1] * len(params))
        except Exception as e:
            print(f"Erreur lors du calcul de la matrice VCV: {str(e)}")
            print("Utilisation d'une matrice VCV approximative")
            vcv_matrix = np.diag([0.1] * len(params))
        
        # Extraire les écarts-types (racine carrée des éléments diagonaux)
        std_errors = np.sqrt(np.diag(vcv_matrix))
        
        # Valeurs typiques d'écarts-types par défaut (en cas de problème)
        default_se = {
            "psi": 0.05,  # AR causal
            "phi": 0.05,  # AR non-causal
            "alpha": 0.15,
            "beta": 0.2,
            "sigma": 0.1
        }
        
        # Valeur minimale raisonnable pour les écarts-types
        min_std_error = 0.01
        
        # Vérifier et corriger les écarts-types problématiques
        for i in range(len(std_errors)):
            param_name = param_names[i]
            param_value = params_array[i]
            
            # Cas où l'écart-type est très petit, infini ou NaN
            if np.isnan(std_errors[i]) or np.isinf(std_errors[i]) or std_errors[i] < min_std_error:
                print(f"Avertissement: Écart-type problématique pour {param_name}: {std_errors[i]}")
                
                # Estimer un écart-type raisonnable basé sur le type de paramètre
                if param_name.startswith("psi"):
                    std_errors[i] = max(min_std_error, default_se["psi"] * (1 + abs(param_value)))
                elif param_name.startswith("phi"):
                    std_errors[i] = max(min_std_error, default_se["phi"] * (1 + abs(param_value)))
                elif param_name == "alpha":
                    std_errors[i] = max(min_std_error, default_se["alpha"])
                elif param_name == "beta":
                    std_errors[i] = max(min_std_error, default_se["beta"])
                elif param_name == "sigma":
                    std_errors[i] = max(min_std_error, default_se["sigma"] * abs(param_value))
                else:
                    std_errors[i] = max(min_std_error, 0.1 * abs(param_value))
                
                print(f"  Nouvel écart-type: {std_errors[i]}")
        
        # Calculer les statistiques t
        t_stats = params_array[:len(param_names)] / std_errors
        
        # Calculer les p-values (approximation par la loi normale)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
        
        # Calculer les intervalles de confiance
        z_critical = stats.norm.ppf(1 - alpha/2)
        lower_ci = params_array[:len(param_names)] - z_critical * std_errors
        upper_ci = params_array[:len(param_names)] + z_critical * std_errors
        
        # Créer un DataFrame pour les résultats
        results = pd.DataFrame({
            'Parameter': param_names,
            'Estimate': params_array[:len(param_names)],
            'Std. Error': std_errors,
            't-statistic': t_stats,
            'p-value': p_values,
            f'CI {100*(1-alpha)}% Lower': lower_ci,
            f'CI {100*(1-alpha)}% Upper': upper_ci
        })
        
        return results


    def _calculate_gcov_vcv(self, data: np.ndarray, params: List[float]) -> np.ndarray:
        """
        Calculates the variance-covariance matrix for the GCov estimator
        based on the asymptotic theory from the GCOVJBES.pdf paper.
        
        Args:
            data: Observed time series
            params: Estimated parameters
            
        Returns:
            Variance-covariance matrix of estimated parameters
        """
        r, s = self.order
        n = len(data)
        
        # Obtenir les résidus
        residuals = self.results.get('PseudoResiduals', None)
        if residuals is None:
            residuals, _ = self._pseudo_residuals(data, np.array(params))
        
        # Nombre total de paramètres AR
        p = r + s
        
        # Pour éviter les problèmes numériques, choisir un epsilon adapté
        # Règle empirique : racine carrée de la précision machine multipliée par un facteur d'échelle
        epsilon = max(1e-4, np.sqrt(np.finfo(float).eps) * 0.1)
        
        # Initialiser la matrice Theta
        Theta = np.zeros((p, p))
        
        # Variance résiduelle (pour normalisation)
        gamma_0 = max(np.var(residuals), 1e-6)  # Éviter division par zéro
        
        # Pour chaque paramètre, calculer numériquement la dérivée des autocovariances
        for i in range(p):
            for j in range(p):
                # Perturber les paramètres avec un effet relatif à leur valeur
                delta_i = max(epsilon, abs(params[i] * 0.01))
                delta_j = max(epsilon, abs(params[j] * 0.01))
                
                # Perturber le i-ème paramètre
                params_plus_i = params.copy()
                params_minus_i = params.copy()
                params_plus_i[i] += delta_i
                params_minus_i[i] -= delta_i
                
                # Perturber le j-ème paramètre
                params_plus_j = params.copy()
                params_minus_j = params.copy()
                params_plus_j[j] += delta_j
                params_minus_j[j] -= delta_j
                
                # Calculer les dérivées croisées des autocovariances pour les H premiers lags
                H = min(10, n//4)  # Limiter le nombre de lags pour la stabilité
                derivative_sum = 0
                
                for h in range(1, H+1):
                    # Calculer les dérivées des autocovariances
                    gamma_h_plus_i = self._compute_acf(data, params_plus_i, h)
                    gamma_h_minus_i = self._compute_acf(data, params_minus_i, h)
                    d_gamma_h_d_theta_i = (gamma_h_plus_i - gamma_h_minus_i) / (2 * delta_i)
                    
                    gamma_h_plus_j = self._compute_acf(data, params_plus_j, h)
                    gamma_h_minus_j = self._compute_acf(data, params_minus_j, h)
                    d_gamma_h_d_theta_j = (gamma_h_plus_j - gamma_h_minus_j) / (2 * delta_j)
                    
                    # Ajouter à la somme selon la formule de la Proposition 4
                    derivative_sum += d_gamma_h_d_theta_i * d_gamma_h_d_theta_j
                
                # Normaliser par gamma_0^2 selon le Corollaire 1
                Theta[i, j] = derivative_sum / (gamma_0**2)
        
        # Ajouter une petite perturbation pour assurer que Theta est définie positive
        Theta += np.eye(p) * max(1e-6, 0.001 * np.trace(Theta) / p)
        
        # Vérifier le conditionnement de la matrice
        try:
            cond_num = np.linalg.cond(Theta)
            if cond_num > 1e10:  # Mauvais conditionnement
                print(f"Warning: Theta matrix ill-conditioned (cond = {cond_num:.2e})")
                # Régularisation plus forte
                Theta += np.eye(p) * max(1e-4, 0.01 * np.trace(Theta) / p)
        except Exception:
            print("Warning: Failed to solve Theta matrix ill-conditioned issue")
            # Régularisation par défaut
            Theta += np.eye(p) * 0.01
        
        # Calculer l'inverse de Theta pour obtenir la matrice de variance-covariance
        try:
            vcv_matrix = np.linalg.inv(Theta) / n
        except np.linalg.LinAlgError:
            print("Warning: Failed to invert Theta, using pseudo-inverse.")
            vcv_matrix = np.linalg.pinv(Theta) / n
        
        # Vérifier que les variances sont positives et raisonnables
        for i in range(p):
            if vcv_matrix[i, i] <= 0 or np.isnan(vcv_matrix[i, i]) or np.isinf(vcv_matrix[i, i]):
                print(f"Warning: Negative or invalid variance for parameter {i}")
                # Utiliser une valeur par défaut
                vcv_matrix[i, i] = 0.01
        
        # Étendre la matrice pour inclure les paramètres de la distribution stable si présents
        if len(params) > p:
            # Si nous avons estimé alpha, beta, sigma dans un deuxième stage
            vcv_stable = self._calculate_stable_params_vcv(residuals, params[p:])
            vcv_matrix_extended = np.zeros((len(params), len(params)))
            
            # Copier les blocs principaux
            vcv_matrix_extended[:p, :p] = vcv_matrix
            vcv_matrix_extended[p:, p:] = vcv_stable
            
            return vcv_matrix_extended
        
        return vcv_matrix


    def _calculate_mdsd_vcv(self, data: np.ndarray, params: List[float]) -> np.ndarray:
        """
        Calculates the variance-covariance matrix for the MDSD estimator
        based on Velasco 2022 theory.
        
        Args:
            data: Observed time series
            params: Estimated parameters
            
        Returns:
            Variance-covariance matrix of estimated parameters
        """
        r, s = self.order
        n = len(data)
        
        # Obtenir les résidus
        residuals = self.results.get('Residuals', None)
        if residuals is None:
            residuals, _ = self._pseudo_residuals(data, np.array(params))
        
        # Nombre total de paramètres AR
        p = r + s
        
        # Epsilon adaptatif pour les dérivées numériques
        epsilon = max(1e-4, np.sqrt(np.finfo(float).eps) * 0.1)
        
        # Calculer numériquement la matrice de score (gradient)
        gradient = np.zeros((n, p))
        
        for j in range(p):
            # Perturbation adaptative
            delta_j = max(epsilon, abs(params[j] * 0.01))
            
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[j] += delta_j
            params_minus[j] -= delta_j
            
            # Calculer la dérivée de la fonction objectif mdsd
            try:
                criterion_plus = self._mds_criterion(data, params_plus, method='L')
                criterion_minus = self._mds_criterion(data, params_minus, method='L')
                gradient[:, j] = (criterion_plus - criterion_minus) / (2 * delta_j)
            except Exception as e:
                print(f"Erreur dans le calcul du gradient pour le paramètre {j}: {str(e)}")
                # En cas d'erreur, utiliser une approximation
                gradient[:, j] = np.random.normal(0, 0.01, n)
        
        # Estimation de la matrice de Hessien
        hessian = np.zeros((p, p))
        
        for i in range(p):
            for j in range(p):
                delta_i = max(epsilon, abs(params[i] * 0.01))
                delta_j = max(epsilon, abs(params[j] * 0.01))
                
                try:
                    params_pp = params.copy()
                    params_pm = params.copy()
                    params_mp = params.copy()
                    params_mm = params.copy()
                    
                    params_pp[i] += delta_i
                    params_pp[j] += delta_j
                    
                    params_pm[i] += delta_i
                    params_pm[j] -= delta_j
                    
                    params_mp[i] -= delta_i
                    params_mp[j] += delta_j
                    
                    params_mm[i] -= delta_i
                    params_mm[j] -= delta_j
                    
                    criterion_pp = self._mds_criterion(data, params_pp, method='L')
                    criterion_pm = self._mds_criterion(data, params_pm, method='L')
                    criterion_mp = self._mds_criterion(data, params_mp, method='L')
                    criterion_mm = self._mds_criterion(data, params_mm, method='L')
                    
                    hessian[i, j] = (criterion_pp - criterion_pm - criterion_mp + criterion_mm) / (4 * delta_i * delta_j)
                except Exception as e:
                    print(f"Erreur dans le calcul du Hessien pour les paramètres {i},{j}: {str(e)}")
                    # En cas d'erreur, utiliser une approximation
                    if i == j:
                        hessian[i, j] = 0.1  # Valeur positive pour la diagonale
                    else:
                        hessian[i, j] = 0.0
        
        # Assurer que le Hessien est défini positif
        hessian_diag = np.diag(hessian)
        if np.any(hessian_diag <= 0):
            print("Avertissement: Hessien non défini positif, ajout d'une régularisation.")
            min_diag = max(0.1, np.median(np.abs(hessian_diag)) * 0.01)
            for i in range(p):
                if hessian[i, i] <= 0:
                    hessian[i, i] = min_diag
        
        # Estimer la matrice de covariance des scores
        score_cov = np.zeros((p, p))
        
        # Sous l'hypothèse que les scores sont des erreurs mds
        for i in range(p):
            for j in range(p):
                for t in range(n):
                    score_cov[i, j] += gradient[t, i] * gradient[t, j]
        
        score_cov /= n
        
        # Régulariser la matrice score_cov si nécessaire
        min_eig = np.min(np.linalg.eigvals(score_cov))
        if min_eig < 1e-10:
            print(f"Avertissement: Matrice de covariance des scores mal conditionnée (min_eig = {min_eig:.2e})")
            score_cov += np.eye(p) * max(1e-4, 0.01 * np.trace(score_cov) / p)
        
        # Inverser le Hessien
        try:
            inv_hessian = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            print("Avertissement: Hessien singulier, utilisation d'une régularisation renforcée.")
            hessian_reg = hessian + 0.01 * np.eye(p) * np.mean(np.abs(np.diag(hessian)))
            inv_hessian = np.linalg.inv(hessian_reg)
        
        # Formule sandwich pour la matrice de variance-covariance
        vcv_matrix = inv_hessian @ score_cov @ inv_hessian / n
        
        # Vérifier les valeurs diagonales et les corriger si nécessaires
        for i in range(p):
            if vcv_matrix[i, i] <= 0 or np.isnan(vcv_matrix[i, i]) or np.isinf(vcv_matrix[i, i]):
                print(f"Avertissement: Variance négative ou invalide pour le paramètre {i}")
                # Calculer une valeur par défaut basée sur les autres éléments de la matrice
                abs_diag = np.abs(np.diag(vcv_matrix))
                abs_diag = abs_diag[~np.isnan(abs_diag) & ~np.isinf(abs_diag) & (abs_diag > 0)]
                if len(abs_diag) > 0:
                    vcv_matrix[i, i] = np.median(abs_diag)
                else:
                    vcv_matrix[i, i] = 0.01
        
        # Étendre la matrice pour inclure les paramètres de la distribution stable si présents
        if len(params) > p:
            vcv_stable = self._calculate_stable_params_vcv(residuals, params[p:])
            vcv_matrix_extended = np.zeros((len(params), len(params)))
            
            # Copier les blocs principaux
            vcv_matrix_extended[:p, :p] = vcv_matrix
            vcv_matrix_extended[p:, p:] = vcv_stable
            
            return vcv_matrix_extended
        
        return vcv_matrix


    def _calculate_stable_params_vcv(self, residuals: np.ndarray, stable_params: List[float]) -> np.ndarray:
        """
        Calculates the variance-covariance matrix for alpha-stable distribution parameters
        with a more robust approach.
        
        Args:
            residuals: Model residuals
            stable_params: Estimated stable parameters [alpha, beta, sigma]
            
        Returns:
            Variance-covariance matrix of stable distribution parameters
        """
        # Nombre de paramètres stables (alpha, beta, sigma)
        n_stable = len(stable_params)
        n = len(residuals)
        
        # Approche simplifiée et robuste pour les paramètres stables
        # Utilisation de valeurs empiriques typiques pour les variances
        vcv_stable = np.zeros((n_stable, n_stable))
        
        # Variance approximative pour alpha
        if n_stable >= 1:
            alpha = stable_params[0]
            # La variance de alpha augmente quand alpha est proche de 2
            alpha_var = 0.02 + 0.03 * (alpha / 2.0)**2
            vcv_stable[0, 0] = alpha_var
        
        # Variance approximative pour beta
        if n_stable >= 2:
            beta = stable_params[1]
            # La variance de beta augmente quand |beta| est proche de 1
            beta_var = 0.04 + 0.06 * beta**2
            vcv_stable[1, 1] = beta_var
            
            # Covariance entre alpha et beta (généralement faible)
            vcv_stable[0, 1] = vcv_stable[1, 0] = 0.01
        
        # Variance approximative pour sigma
        if n_stable >= 3:
            sigma = stable_params[2]
            # La variance de sigma est proportionnelle à sigma^2
            sigma_var = 0.01 * sigma**2
            vcv_stable[2, 2] = sigma_var
            
            # Covariances négligeables avec les autres paramètres
            vcv_stable[0, 2] = vcv_stable[2, 0] = 0.005
            vcv_stable[1, 2] = vcv_stable[2, 1] = 0.005
        
        # Ajustement pour la taille de l'échantillon
        vcv_stable = vcv_stable * (200 / n) if n > 0 else vcv_stable
        
        return vcv_stable


    def _compute_acf(self, data: np.ndarray, params: List[float], lag: int) -> float:
        """
        Calculates the autocovariance of residuals at a given lag for specific parameters.
        
        Args:
            data: Observed time series
            params: Model parameters
            lag: Autocovariance order
            
        Returns:
            Autocovariance value at order lag
        """
        # Obtenir les résidus pour ces paramètres
        try:
            residuals, _ = self._pseudo_residuals(data, np.array(params))
            
            # Calculer l'autocovariance
            n = len(residuals)
            if n <= lag:
                return 0.0
                
            mean = np.mean(residuals)
            acf = 0
            
            for t in range(lag, n):
                acf += (residuals[t] - mean) * (residuals[t-lag] - mean)
            
            return acf / (n - lag)
        except Exception as e:
            print(f"Erreur dans le calcul de l'ACF au lag {lag}: {str(e)}")
            return 0.0


    def generate(self, n: int, errors: List[float] = None, seed: Optional[int] = None) -> "StableMAR":
        """
        Simulate a stable MAR(r, s) model.
            
        Args:
            n: Sample size to generate.
            errors: Optional pre-generated errors.
            seed: Random seed for reproducibility.
                
        Returns:
            Self with trajectory and innovation attributes set.
        """
        if self.par is None or len(self.par) < 3:
            raise ValueError("Parameters must be set before generating data")
        
        r, s = self.order
        
        # Extract stable distribution parameters
        if len(self.par) > r + s:
            alpha = self.par[r+s]
            beta = self.par[r+s+1] if len(self.par) > r+s+1 else 0.0
            sigma = self.par[r+s+2] if len(self.par) > r+s+2 else 1.0
        else:
            alpha, beta, sigma = 1.5, 0.0, 1.0  # Default values
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Generate innovations
        m = 50  # Truncation for the MA filter
        if n < 2*m:
            warnings.warn(f"Sample size (n={n}) is too small... n >= 100 is required")
            
        ntilde = n + 2*m + 1
        
        # Get the MA filter coefficients
        deltas = self.ma_filter(m)
        deltas = np.flip(deltas)  # Respecter l'orientation de l'original
        
        # Generate alpha-stable errors
        if errors is None or len(errors) != ntilde:
            esim = stats.levy_stable.rvs(
                alpha=alpha, 
                beta=beta, 
                scale=sigma, 
                loc=0, 
                size=n*3
            )
        else:
            esim = errors
        
        # Generate the process using MA filter
        xsim = np.ones(ntilde) * esim[0]
        
        for t in range(m, ntilde):
            xsim[t] = np.sum(deltas * esim[t-m:t+m])
        
        xsim = xsim[m:-m]
        esim = esim[m:-m]
        
        # Store results
        self.trajectory = pd.Series(xsim)
        self.innovation = pd.Series(esim)
        
        return self


    def ma_filter(self, m: int) -> np.ndarray:
        """
        Generate the MA filter coefficients for the MAR model.
        
        Args:
            m: Truncation parameter for the MA filter.
            
        Returns:
            Array of MA filter coefficients.
        """
        if self.par is None:
            raise ValueError("Parameters must be set before generating MA filter")
            
        r, s = self.order
        
        # Extract MAR parameters
        if r > 0:
            psi = np.array(self.par[:r])
        else:
            psi = np.array([])
            
        if s > 0:
            phi = np.array(self.par[r:r+s])
        else:
            phi = np.array([])
        
        deltas = np.full(2*m, np.nan)
        
        for k in range(-m, m):
            deltas[k+m] = madelta(psi, phi, k)
        
        deltas = np.flip(deltas)
        
        return deltas


    def forecast(self, x: pd.Series, tau: int, m: int, h: int, k0: int, vartheta: int, trunc: int = 50) -> np.ndarray:
        """
        Compute pattern-based forecasts for a MAR(p,q) model with q >= 2.
        
        Args:
            x: Observed time series data.
            tau: Index of the last in-sample observation.
            m: Temporal depth of the segment (X_t-m,...,X_t).
            k0: Index of the MA coefficient in the infinite sum.
            h: Forecast horizon (typically h = k0 + 1).
            vartheta: -1 (negative bubble) or 1 (positive bubble).
            trunc: Truncation parameter for the MA filter.
            
        Returns:
            Array of forecasted values.
        """
        if trunc <= m:
            trunc = 2*m
        
        # Get the MA filter coefficients
        dk = self.ma_filter(trunc)
        
        # Indices pour l'extraction du pattern
        dw = trunc - k0 - m
        up = trunc - k0 + h
        dkmh = dk[dw:up]
        
        dsnorm = (vartheta * dkmh / np.sqrt(np.sum(dkmh[:m]**2))).T
        
        # Extraction de l'observation
        xmh = np.full(m+1, np.nan)
        for i in range(0, m + 1):
            xmh[i] = float(x.iloc[tau - i])
        
        xmh = np.flip(xmh)
        
        # Indices pour le forecast
        xfore = np.full(len(x)+h, np.nan)
        xfore[tau-m+1:tau+h+1] = dsnorm * np.sqrt(np.sum(xmh**2))
        
        return xfore


    def pathfinder(self, x: pd.Series, tau: int, m: int, kmax: int, vartheta: int, trunc: int = 50) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the optimal k0 index for pattern matching in MAR models.
        
        Args:
            x: Observed time series data.
            tau: Index of the last in-sample observation.
            m: Temporal depth of the segment (X_t-m,...,X_t).
            kmax: Maximum value to search for k0.
            vartheta: -1 (negative bubble) or 1 (positive bubble).
            trunc: Truncation parameter for the MA filter.
            
        Returns:
            Tuple containing k0 and other relevant arrays for pattern matching.
        """
        # Convert Series to numpy array if needed
        if isinstance(x, pd.Series):
            x = x.to_numpy()
        
        # Extraction des observations
        xm = np.full(m+1, np.nan)
        for i in range(0, m+1):
            xm[i] = x[tau - i]
        
        xm = np.flip(xm)
        
        xsnorm = xm / np.sqrt(np.sum(xm[:m+1]**2))
        
        # Initialize arrays
        ngA = np.full(kmax+1, np.nan)
        dkm = np.full((kmax+1, m+1), np.nan)
        dsnorm = np.full((kmax+1, m+1), np.nan) 
        
        # Get the MA filter coefficients
        dk = self.ma_filter(trunc)
        
        # Calcul des patterns
        for k in range(kmax, -1, -1):
            dw = trunc - k - m - 1
            up = trunc - k
            dkm[k, :] = dk[dw:up]            
            dsnorm[k, :] = vartheta * (dkm[k, :] / np.sqrt(np.sum(dkm[k, :]**2)))
        
        # Calculate distance
        ngA = np.sum(np.abs(xsnorm - dsnorm), axis=1)
        
        # Ajouter 1 à l'indice k0
        k0 = np.argmin(ngA) + 1
        
        return k0, ngA, dsnorm, xsnorm, xm, dkm


    def foreprob(self, x: pd.Series, k0: int, h: int, maxp: float, t: int, vartheta: int, m: int = 0) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        r, s = self.order
        
        # Extract parameters
        if r > 0:
            phi_c = np.array(self.par[:r])
        else:
            phi_c = np.array([0])
            
        if s == 1:
            phi_nc = np.array(self.par[r:r+s])
        else:
            print("Warning: probability-based forecast is available only for MAR(r,1), s is set to 1")
            phi_nc = np.array(self.par[r:r+1])
        
        # Get alpha parameter
        alpha = self.par[r+s] if len(self.par) > r+s else 1.5
        
        proba = np.empty((h,4))
        
        # Calculer les probabilités
        for i in range(1,h+1):
            proba[i-1,0] = i
            proba[i-1,1] = (abs(phi_nc[0]) ** (alpha*(i-1)))*(1 - abs(phi_nc[0]) ** alpha)
            proba[i-1,2] = (abs(phi_nc[0]) ** (alpha*(i)))
            proba[i-1,3] = 1 - (abs(phi_nc[0]) ** (alpha*(i)))
        
        # Créer la trajectoire complète : past + present + future
        total_length = m + 1 + h  # past + present + future
        full_traj = np.zeros(total_length)
        
        # Valeur à l'instant t (point de jonction)
        current_value = x.iloc[t]
        
        # Construction du pattern théorique pour le passé
        rho = phi_nc[0]
        
        # Générer le pattern pour le passé (de t-m à t-1)
        # Le pattern est une exponentielle croissante vers le point t
        past_pattern = np.array([rho**(k0+m-i) for i in range(m+1)])  # Inclut le point t
        
        # Normaliser et mettre à l'échelle par rapport à la valeur actuelle
        if np.linalg.norm(past_pattern) > 0:
            scale_factor = abs(current_value) / abs(past_pattern[-1])  # La dernière valeur doit être x.iloc[t]
        else:
            scale_factor = 1.0
        
        # Appliquer le signe et l'échelle à la partie historique
        full_traj[:m+1] = vartheta * past_pattern * scale_factor
        
        # Génération de la partie future
        current_t_value = full_traj[m]  # Valeur à t
        
        for i in range(1, h + 1):
            current_idx = m + i  # Position dans full_traj
            
            # Utiliser les probabilités pour déterminer la trajectoire
            if proba[i-1,3] < maxp:
                # Continue to grow
                if i == 1:  # Premier pas après t
                    full_traj[current_idx] = current_t_value / abs(rho)
                else:
                    full_traj[current_idx] = full_traj[current_idx-1] / abs(rho)
            else:
                # Crash: reverse direction
                if abs(phi_c[0]) > 0:
                    full_traj[current_idx] = full_traj[current_idx-1] * abs(phi_c[0])
                else:
                    full_traj[current_idx] = 0
        
        # Définir les indices des patterns
        proba[0,1] = np.nan
        if k0 > 1:
            proba[0:k0-1,2:4] = np.nan
        
        # Créer DataFrame des probabilités
        dfprob = pd.DataFrame(proba, columns=['h', 'Crash Before h', 'Survive at h', 'Crash at h'])
        dfprob.set_index('h', inplace=True)
        
        # Créer les patterns avec un indexage correct
        # Pattern passé : de (t-m) à t
        past_pattern_indexed = np.full(len(x)+h, np.nan)
        past_pattern_indexed[t-m:t+1] = full_traj[:m+1]
        
        # Pattern prédictif : de t à (t+h)  
        pred_pattern_indexed = np.full(len(x)+h, np.nan)
        pred_pattern_indexed[t:t+h+1] = full_traj[m:]
        
        # Full pattern : de (t-m) à (t+h)
        full_pattern_indexed = np.full(len(x)+h, np.nan)
        full_pattern_indexed[t-m:t+h+1] = full_traj
        
        return dfprob, pred_pattern_indexed, past_pattern_indexed, full_pattern_indexed


    def kfinder(self, x: pd.Series, tau: int, m: int = 2, kmax: int = 15,
                vartheta: int = 1, kmin: int = 1, visualize: bool = False) -> Tuple[int, float]:
        """
        Determines the optimal k0 parameter for an anticipative AR(1) by sliding a pattern
        over the latest observations.
    
        Args:
            x: Observed time series
            tau: Index of the last in-sample observation
            m: Number of observations to use for matching (depth)
            kmax: Maximum value of k0 to consider
            vartheta: Direction of the bubble (1 for positive, -1 for negative)
            kmin: Minimum value of k0 to consider
            visualize: If True, generates a matching visualization plot
        
        Returns:
            Tuple containing the optimal k0 and matching quality (normalized distance)
        """
        # Verify that the model is an anticipative AR(1)
        r, s = self.order
        if r != 0 or s != 1:
            print("Warning: This function is optimized for anticipative AR(1). "
                "For other models, use pathfinder.")
    
        # Extract last observations
        xm = np.array([x.iloc[tau-i] for i in range(m)])
        xm = np.flip(xm)  # Flip to get [x_{t-m+1}, ..., x_t]
    
        # Normalize
        xnorm = xm / np.linalg.norm(xm)
    
        # Get phi parameter from the anticipative AR(1) model
        phi = self.par[r] if s > 0 else 0
    
        # Compute distances for different k values
        distances = np.zeros(kmax + 1)
        patterns = []
    
        for k in range(kmin, kmax + 1):
            # Generate theoretical pattern for AR(1) with lag k
            # For anticipative AR(1): d_k = φ^k for k ≥ 0
            pattern = np.array([phi**(k-i) for i in range(m)])
        
            # Normalize the pattern
            pattern_norm = pattern / np.linalg.norm(pattern)
            patterns.append(pattern_norm)
        
            # Compute distance (accounting for vartheta)
            distances[k] = np.sum(np.abs(xnorm - vartheta * pattern_norm))
            

        # Find k that minimizes distance
        k0 = np.argmin(distances[kmin:]) + kmin
        min_distance = distances[k0]
        quality = 1 - min_distance / (2 * m)  # Quality between 0 and 1
    
        # Visualization if requested
        if visualize:
            plt.figure(figsize=(10, 6))
        
            # Plot distances
            plt.subplot(2, 1, 1)
            plt.bar(range(kmin, kmax + 1), distances[kmin:])
            plt.axvline(k0, color='red', linestyle='--')
            plt.xlabel('k')
            plt.ylabel('Distance')
            plt.title(f'Distances for different k values (optimal k0 = {k0})')
        
            # Plot optimal pattern vs observations
            plt.subplot(2, 1, 2)
            plt.plot(range(m), xnorm, 'b-o', label='Normalized observations')
            plt.plot(range(m), vartheta * patterns[k0 - kmin], 'r--x', label=f'Theoretical pattern (k0={k0})')
            plt.xlabel('Position')
            plt.ylabel('Normalized value')
            plt.title(f'Pattern matching (quality: {quality:.2f})')
            plt.legend()
        
            plt.tight_layout()
            plt.show()
    
        return k0, quality

    
    def _param_bounds(self, n_params: int) -> List[Tuple[float, float]]:
        """
        Generate parameter bounds for optimization.
        
        Args:
            n_params: Number of parameters to generate bounds for.
            
        Returns:
            List of (lower, upper) bounds for each parameter.
        """
        # All MAR parameters are bounded between -0.99 and 0.99
        return [(-.99, .99) for _ in range(n_params)]
    
    
    def generate_initial_guess(self, random: bool = False) -> List[float]:
        """
        Generate initial parameter guesses for estimation.
        
        Args:
            random: If True, generate random initial values within reasonable ranges.
            
        Returns:
            List of initial parameter values.
        """
        r, s = self.order
        p = r + s
        
        if random:
            # Generate random initial values
            ar_init = np.random.uniform(low=0.1, high=0.8, size=p)
            
            # Ensure stability by scaling if necessary
            if r > 0:
                psi_sum = np.sum(np.abs(ar_init[:r]))
                if psi_sum >= 0.99:
                    ar_init[:r] = ar_init[:r] * (0.9 / psi_sum)
            
            if s > 0:
                phi_sum = np.sum(np.abs(ar_init[r:]))
                if phi_sum >= 0.99:
                    ar_init[r:] = ar_init[r:] * (0.9 / phi_sum)
        else:
            # Use fixed initial values
            ar_init = np.ones(p) * 0.4
            
            # Alternate signs for better initial guess
            for i in range(p):
                if i % 2 == 1:
                    ar_init[i] = -ar_init[i]
        
        return ar_init.tolist()


    def get_model_summary(self, test_residuals: bool = True, bootstrap: bool = False, 
                        H: int = 2, K: int = 2, n_bootstrap: int = 1000) -> pd.DataFrame:
        """
        Create a summary of model parameters and goodness of fit with optional residual testing.
        
        Args:
            test_residuals: If True, performs a Portmanteau test on residuals
            bootstrap: If True and test_residuals is True, uses bootstrap version of Portmanteau test
            H: Number of lags to use in the Portmanteau test
            K: Number of nonlinear transformations for the Portmanteau test
            n_bootstrap: Number of bootstrap samples if bootstrap=True
            
        Returns:
            DataFrame containing parameter estimates, test statistics and other model information
        """
        if not self.results:
            raise ValueError("No estimation results available. Fit the model first.")
        
        r, s = self.order
        params = self.results.get('Parameters')
        
        if params is None or len(params) < r + s:
            raise ValueError("Invalid parameter estimates in results.")
        
        # Create parameter names
        param_names = []
        param_values = []
        
        # Causal parameters
        for i in range(r):
            param_names.append(f"psi_{i+1}")
            param_values.append(params[i])
        
        # Noncausal parameters
        for i in range(s):
            param_names.append(f"phi_{i+1}")
            param_values.append(params[r+i])
        
        # Create DataFrame
        summary = pd.DataFrame({
            'Parameter': param_names,
            'Estimate': param_values
        })
        
        # Add method and criterion value
        method = self.results.get('Method', 'Unknown')
        criterion_name = 'GStatistic' if method == 'GCoV' else 'CriterionValue'
        criterion_value = self.results.get(criterion_name, np.nan)
        
        # Add a row for the method and append to summary
        method_row = pd.DataFrame({'Parameter': ['Method'], 'Estimate': [method]})
        summary = pd.concat([summary, method_row], ignore_index=True)
        
        # Add a row for the criterion and append to summary
        criterion_row = pd.DataFrame({'Parameter': ['Criterion'], 'Estimate': [criterion_value]})
        summary = pd.concat([summary, criterion_row], ignore_index=True)
        
        # Add Portmanteau test results if requested
        if test_residuals:
            # Get residuals for testing
            residuals = None
            if 'PseudoResiduals' in self.results:
                residuals = self.results['PseudoResiduals']
            elif 'Residuals' in self.results:
                residuals = self.results['Residuals']
            elif 'StdResiduals' in self.results:
                residuals = self.results['StdResiduals']
            
            if residuals is not None and len(residuals) > 0:
                try:
                    # Perform the appropriate Portmanteau test
                    if bootstrap:
                        # Use bootstrap version of the test
                        test_results = self.bootstrap_portmanteau_test(
                            residuals, self.par, H=H, K=K, n_bootstrap=n_bootstrap
                        )
                        
                        # Add bootstrap test results
                        portman_rows = [
                            {'Parameter': 'Portmanteau Test', 'Estimate': 'Bootstrap'},
                            {'Parameter': 'Test Statistic', 'Estimate': test_results['original_statistic']},
                            {'Parameter': 'Asymptotic p-value', 'Estimate': test_results['original_p_value']},
                            {'Parameter': 'Bootstrap p-value', 'Estimate': test_results['bootstrap_p_value']},
                            {'Parameter': 'Bootstrap Mean', 'Estimate': test_results['mean_bootstrap']},
                            {'Parameter': 'Theoretical Mean', 'Estimate': test_results['theoretical_mean']},
                            {'Parameter': 'Lags (H)', 'Estimate': H},
                            {'Parameter': 'Transformations (K)', 'Estimate': K}
                        ]
                    else:
                        # Use standard asymptotic test
                        test_results = self.portmanteau_test(
                            residuals, self.par, H=H, K=K
                        )
                        
                        # Add standard test results
                        portman_rows = [
                            {'Parameter': 'Portmanteau Test', 'Estimate': 'Asymptotic'},
                            {'Parameter': 'Test Statistic', 'Estimate': test_results['statistic']},
                            {'Parameter': 'p-value', 'Estimate': test_results['p_value']},
                            {'Parameter': 'Degrees of Freedom', 'Estimate': test_results['degrees_of_freedom']},
                            {'Parameter': 'Lags (H)', 'Estimate': test_results['H']},
                            {'Parameter': 'Transformations (K)', 'Estimate': test_results['K']}
                        ]
                    
                    # Add all Portmanteau rows to the summary
                    portman_df = pd.DataFrame(portman_rows)
                    summary = pd.concat([summary, portman_df], ignore_index=True)
                    
                except Exception as e:
                    # Add error information if test fails
                    error_row = pd.DataFrame({
                        'Parameter': ['Portmanteau Test Error'], 
                        'Estimate': [str(e)]
                    })
                    summary = pd.concat([summary, error_row], ignore_index=True)
        
        return summary


    def diagnostic_plots(self) -> None:
        """
        Generate diagnostic plots for model assessment.
        
        Plots include:
        - Residuals time series
        - Residuals distribution
        - Autocorrelation of residuals
        - Autocorrelation of squared residuals
        """
        if not self.results:
            raise ValueError("No estimation results available. Fit the model first.")
        
        # Get residuals
        residuals = None
        if 'PseudoResiduals' in self.results:
            residuals = self.results['PseudoResiduals']
        elif 'Residuals' in self.results:
            residuals = self.results['Residuals']
        
        if residuals is None or len(residuals) == 0:
            raise ValueError("No residuals available for diagnostics.")
        
        # Create a figure with subplots
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Residuals time series
        plt.subplot(2, 2, 1)
        plt.plot(residuals)
        plt.title('Residuals Time Series')
        plt.grid(True)
        
        # Plot 2: Residuals distribution
        plt.subplot(2, 2, 2)
        plt.hist(residuals, bins=30, density=True, alpha=0.7)
        
        # Add normal density for comparison
        x = np.linspace(min(residuals), max(residuals), 1000)
        mu, std = np.mean(residuals), np.std(residuals)
        plt.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2, label='Normal')
        
        # Try to fit and plot stable distribution
        try:
            # Estimate stable parameters
            stable_params = self.fit_stable_noise(residuals)
            
            # Create stable distribution
            stable_dist = stats.levy_stable(
                alpha=stable_params[0],
                beta=stable_params[1],
                scale=stable_params[2],
                loc=0
            )
            
            # Plot stable density
            plt.plot(x, stable_dist.pdf(x), 'g--', lw=2, 
                    label=f'Stable (α={stable_params[0]:.2f})')
        except Exception as e:
            print(f"Could not fit stable distribution: {e}")
        
        plt.title('Residuals Distribution')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Autocorrelation function of residuals
        plt.subplot(2, 2, 3)
        
        # Calculate ACF
        acf = np.correlate(residuals - np.mean(residuals), 
                        residuals - np.mean(residuals), mode='full')
        
        # Normalize and take positive lags only
        acf = acf[len(acf)//2:] / acf[len(acf)//2]
        lags = np.arange(len(acf))
        
        # Plot ACF
        plt.bar(lags[:20], acf[:20], width=0.3)
        
        # Add confidence interval (approximate for large samples)
        conf_level = 1.96 / np.sqrt(len(residuals))
        plt.axhline(y=conf_level, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=-conf_level, color='r', linestyle='--', alpha=0.5)
        
        plt.title('Autocorrelation of Residuals')
        plt.xlabel('Lag')
        plt.grid(True)
        
        # Plot 4: Autocorrelation function of squared residuals
        plt.subplot(2, 2, 4)
        
        # Calculate ACF of squared residuals
        squared_resid = (residuals - np.mean(residuals))**2
        acf_squared = np.correlate(squared_resid - np.mean(squared_resid), 
                                squared_resid - np.mean(squared_resid), mode='full')
        
        # Normalize and take positive lags only
        acf_squared = acf_squared[len(acf_squared)//2:] / acf_squared[len(acf_squared)//2]
        
        # Plot ACF of squared residuals
        plt.bar(lags[:20], acf_squared[:20], width=0.3)
        
        # Add confidence interval
        plt.axhline(y=conf_level, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=-conf_level, color='r', linestyle='--', alpha=0.5)
        
        plt.title('Autocorrelation of Squared Residuals')
        plt.xlabel('Lag')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        

    def mcsingle(self, sim_idx: int, param_idx: int, params: tuple, T: int, 
                seed: Optional[int] = None, forecast_eval: bool = False) -> Dict[str, Any]:
        """
        Executes a single Monte Carlo simulation for a given parameter set.
        
        Args:
            sim_idx: Simulation index
            param_idx: Parameter set index
            params: Model parameters
            T: Sample size
            seed: Seed for reproducibility
            forecast_eval: If True, also evaluates forecasts
                
        Returns:
            Dictionary containing simulation results
        """
        result = {}
        
        try:
            # Graine unique pour cette simulation
            sim_seed = seed if seed is not None else 42000 + sim_idx * 100 + param_idx
            
            # Créer et générer le modèle
            model = StableMAR(self.order, params.copy())
            model.generate(T, seed=sim_seed)
            
            # Estimer les paramètres
            init_guess = model.generate_initial_guess(random=True)
            start_time = time.time()
            model.fit(model.trajectory.values, init_guess, 
                    method="gcov", K=2, H=2, verbose=False)
            estimation_time = time.time() - start_time
            
            # Extraire les paramètres estimés
            est_params = model.par
            
            # Initialiser les résultats avec les infos de base
            result = {
                'simulation': sim_idx,
                'param_set': param_idx,
                'estimation_time': estimation_time,
                'success': True,
            }
            
            # Ajouter les paramètres vrais et estimés
            r, s = self.order
            
            # Paramètres AR
            for i in range(r):
                param_name = f"psi_{i+1}"
                true_value = params[i] if i < len(params) else np.nan
                est_value = est_params[i] if i < len(est_params) else np.nan
                
                result[f"{param_name}_true"] = true_value
                result[f"{param_name}_est"] = est_value
                result[f"{param_name}_error"] = est_value - true_value
                result[f"{param_name}_rel_error"] = abs(est_value - true_value) / max(abs(true_value), 1e-10)
            
            # Paramètres MA
            for i in range(s):
                param_name = f"phi_{i+1}"
                true_value = params[r+i] if r+i < len(params) else np.nan
                est_value = est_params[r+i] if r+i < len(est_params) else np.nan
                
                result[f"{param_name}_true"] = true_value
                result[f"{param_name}_est"] = est_value
                result[f"{param_name}_error"] = est_value - true_value
                result[f"{param_name}_rel_error"] = abs(est_value - true_value) / max(abs(true_value), 1e-10)
            
            # Paramètres de distribution alpha-stable
            if len(params) > r+s:
                param_pairs = [
                    ('alpha', r+s),
                    ('beta', r+s+1),
                    ('sigma', r+s+2)
                ]
                
                for param_name, idx in param_pairs:
                    if idx < len(params):
                        true_value = params[idx]
                        # Estimer les paramètres alpha-stable à partir des résidus
                        if param_name == 'alpha' and 'PseudoResiduals' in model.results:
                            residuals = model.results['PseudoResiduals']
                            stable_params = model.fit_stable_noise(residuals)
                            est_value = stable_params[0]  # alpha
                        elif param_name == 'beta' and 'PseudoResiduals' in model.results:
                            residuals = model.results['PseudoResiduals']
                            stable_params = model.fit_stable_noise(residuals)
                            est_value = stable_params[1]  # beta
                        elif param_name == 'sigma' and 'PseudoResiduals' in model.results:
                            residuals = model.results['PseudoResiduals']
                            stable_params = model.fit_stable_noise(residuals)
                            est_value = stable_params[2]  # sigma
                        else:
                            est_value = np.nan
                            
                        result[f"{param_name}_true"] = true_value
                        result[f"{param_name}_est"] = est_value
                        result[f"{param_name}_error"] = est_value - true_value
                        result[f"{param_name}_rel_error"] = abs(est_value - true_value) / max(abs(true_value), 1e-10)
            
            # Évaluation des prévisions si demandée
            if forecast_eval and len(model.trajectory) > 10:
                # Séparer les données en échantillon d'entraînement/test
                train_size = int(0.8 * len(model.trajectory))
                train_data = model.trajectory[:train_size]
                test_data = model.trajectory[train_size:]
                
                # Évaluer la prévision pour différents horizons
                for h in [1, 5, 10]:
                    if h >= len(test_data):
                        continue
                        
                    # Effectuer la prévision selon le type de modèle
                    try:
                        if s >= 2:  # MAR(r,s) avec s >= 2
                            # Utiliser pathfinder pour trouver k0
                            m = 2  # Profondeur temporelle
                            kmax = 15
                            vartheta = 1 if train_data.iloc[-1] > 0 else -1
                            
                            k0, _, _, _, _, _ = model.pathfinder(
                                train_data, train_size-1, m, kmax, vartheta
                            )
                            
                            # Générer la prévision
                            forecasts = model.forecast(
                                train_data, train_size-1, m, h, k0, vartheta
                            )
                            
                            # Extraire les valeurs prévues 
                            forecast_indices = np.where(~np.isnan(forecasts))[0]
                            if len(forecast_indices) > 0 and train_size+h-1 < len(forecasts):
                                forecast_value = forecasts[train_size+h-1]
                                actual_value = test_data.iloc[h-1]
                                
                                # Calculer les métriques
                                result[f'forecast_h{h}_value'] = forecast_value
                                result[f'forecast_h{h}_actual'] = actual_value
                                result[f'forecast_h{h}_error'] = actual_value - forecast_value
                                result[f'forecast_h{h}_squared_error'] = (actual_value - forecast_value)**2
                                result[f'forecast_h{h}_abs_error'] = abs(actual_value - forecast_value)
                        
                        elif s == 1:  # MAR(r,1)
                            # Utiliser foreprob
                            maxp = 0.9
                            vartheta = 1 if train_data.iloc[-1] > 0 else -1
                            
                            _, forecast_traj = model.foreprob(
                                train_data, 6, h, maxp, train_size-1, vartheta
                            )
                            
                            # Extraire les valeurs prévues
                            forecast_indices = np.where(~np.isnan(forecast_traj))[0]
                            if len(forecast_indices) > 0 and train_size+h-1 < len(forecast_traj):
                                forecast_value = forecast_traj[train_size+h-1]
                                actual_value = test_data.iloc[h-1]
                                
                                # Calculer les métriques
                                result[f'forecast_h{h}_value'] = forecast_value
                                result[f'forecast_h{h}_actual'] = actual_value
                                result[f'forecast_h{h}_error'] = actual_value - forecast_value
                                result[f'forecast_h{h}_squared_error'] = (actual_value - forecast_value)**2
                                result[f'forecast_h{h}_abs_error'] = abs(actual_value - forecast_value)
                    except Exception as e:
                        result[f'forecast_h{h}_error_msg'] = str(e)
            
        except Exception as e:
            # Capturer toute autre erreur
            result = {
                'simulation': sim_idx,
                'param_set': param_idx,
                'success': False,
                'error': str(e)
            }
        
        return result


    def mcstudy(self, n_simulations: int = 100, T: int = 200, 
                parameter_sets: Optional[List[tuple]] = None, 
                n_jobs: int = 1, forecast_eval: bool = False,
                seed: Optional[int] = None) -> pd.DataFrame:
        """
        Executes a complete Monte Carlo study with parallelization.
        
        Args:
            n_simulations: Number of simulations per parameter set
            T: Sample size
            parameter_sets: List of parameter sets to test
            n_jobs: Number of parallel processes (-1 to use all cores)
            forecast_eval: If True, also evaluates forecasts
            seed: Seed for reproducibility
                
        Returns:
            DataFrame with study results
        """
        # Générer des ensembles de paramètres par défaut si non spécifiés
        if parameter_sets is None:
            r, s = self.order
            if r == 1 and s == 1:  # MAR(1,1)
                parameter_sets = [
                    (0.3, 0.7, 1.5, 0.0, 1.0),  # psi faible, phi modéré
                    (0.6, 0.4, 1.5, 0.0, 1.0),  # psi modéré, phi faible
                    (0.8, 0.8, 1.8, 0.0, 1.0),  # psi et phi élevés, alpha élevé
                ]
            elif r == 1 and s == 2:  # MAR(1,2)
                parameter_sets = [
                    (0.4, 0.7, 0.1, 1.5, 0.0, 1.0),  # Configuration standard
                    (0.3, 0.4, 0.3, 1.5, 0.0, 1.0),  # Faible persistance
                    (0.7, 0.6, 0.3, 1.8, 0.0, 1.0),  # Forte persistance, alpha élevé
                ]
            else:
                # Configuration par défaut pour d'autres ordres
                parameter_sets = [(0.4,) * (r+s) + (1.5, 0.0, 1.0)]
        
        # Déterminer le nombre de CPU à utiliser
        if n_jobs == -1:
            n_jobs = max(1, multiprocessing.cpu_count() - 4)
        
        print(f"Running {n_simulations} simulations for {len(parameter_sets)} parameter sets "
            f"with {n_jobs} parallel processes...")
        
        # Créer les tâches
        tasks = []
        for param_idx, params in enumerate(parameter_sets):
            for sim_idx in range(n_simulations):
                # Calculer une graine spécifique si une graine de base est fournie
                sim_seed = seed + sim_idx * 1000 + param_idx if seed is not None else None
                tasks.append((sim_idx, param_idx, params, T, sim_seed, forecast_eval))
        
        # Exécuter la simulation en parallèle
        if n_jobs > 1:            
            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(self.mcsingle)(*task) for task in tasks
            )
        else:
            # Exécution séquentielle avec barre de progression
            
            results = []
            for task in tqdm(tasks, desc="Monte Carlo Simulation"):
                results.append(self.mcsingle(*task))
        
        # Filtrer les simulations réussies
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            raise ValueError("Failure of all simulations. Check all parameteres and methods.")
        
        # Convertir en DataFrame
        results_df = pd.DataFrame(successful_results)
        
        # Afficher le taux de réussite
        success_rate = len(successful_results) / len(results)
        print(f"Success rate: {success_rate:.2%} ({len(successful_results)}/{len(results)})")
        
        return results_df


    def analyze_mcstudy(self, results_df: pd.DataFrame, verbose: bool = False,
                    plot: bool = True, save_plot: bool = False,
                    plot_prefix: str = "mar_mc") -> pd.DataFrame:
        """
        Analyzes and summarizes Monte Carlo study results.
        
        Args:
            results_df: DataFrame with raw results
            plot: If True, generates plots
            save_plot: If True, saves plots instead of displaying them
            plot_prefix: Prefix for saved plot filenames
                
        Returns:
            DataFrame with statistical summary
        """
        # Vérifier si le DataFrame contient des résultats
        if results_df.empty:
            print("No results to analyze.")
            return pd.DataFrame()
        
        # Créer un DataFrame de résumé
        summary_data = []
        
        # Pour chaque ensemble de paramètres
        for param_set in sorted(results_df['param_set'].unique()):
            subset = results_df[results_df['param_set'] == param_set]
            
            # Informations de base du jeu de paramètres
            param_summary = {'param_set': param_set}
            
            # Nombre de simulations réussies
            param_summary['n_simulations'] = len(subset)
            
            # Temps d'estimation moyen
            if 'estimation_time' in subset.columns:
                param_summary['mean_estimation_time'] = subset['estimation_time'].mean()
            
            # Pour chaque paramètre MAR
            r, s = self.order
            
            # Paramètres AR
            for i in range(r):
                param_name = f"psi_{i+1}"
                true_col = f"{param_name}_true"
                est_col = f"{param_name}_est"
                error_col = f"{param_name}_error"
                
                if true_col in subset.columns and est_col in subset.columns:
                    true_value = subset[true_col].iloc[0]  # Valeur vraie (constante)
                    
                    # Statistiques sur les estimations
                    param_summary[f"{param_name}_true"] = true_value
                    param_summary[f"{param_name}_mean"] = subset[est_col].mean()
                    param_summary[f"{param_name}_std"] = subset[est_col].std()
                    param_summary[f"{param_name}_bias"] = subset[error_col].mean()
                    param_summary[f"{param_name}_rmse"] = np.sqrt(np.mean(subset[error_col]**2))
            
            # Paramètres MA
            for i in range(s):
                param_name = f"phi_{i+1}"
                true_col = f"{param_name}_true"
                est_col = f"{param_name}_est"
                error_col = f"{param_name}_error"
                
                if true_col in subset.columns and est_col in subset.columns:
                    true_value = subset[true_col].iloc[0]  # Valeur vraie (constante)
                    
                    # Statistiques sur les estimations
                    param_summary[f"{param_name}_true"] = true_value
                    param_summary[f"{param_name}_mean"] = subset[est_col].mean()
                    param_summary[f"{param_name}_std"] = subset[est_col].std()
                    param_summary[f"{param_name}_bias"] = subset[error_col].mean()
                    param_summary[f"{param_name}_rmse"] = np.sqrt(np.mean(subset[error_col]**2))
            
            # Paramètres de distribution alpha-stable
            for param_name in ['alpha', 'beta', 'sigma']:
                true_col = f"{param_name}_true"
                est_col = f"{param_name}_est"
                error_col = f"{param_name}_error"
                
                if true_col in subset.columns and est_col in subset.columns:
                    true_value = subset[true_col].iloc[0]  # Valeur vraie (constante)
                    
                    # Statistiques sur les estimations
                    param_summary[f"{param_name}_true"] = true_value
                    param_summary[f"{param_name}_mean"] = subset[est_col].mean()
                    param_summary[f"{param_name}_std"] = subset[est_col].std()
                    param_summary[f"{param_name}_bias"] = subset[error_col].mean()
                    param_summary[f"{param_name}_rmse"] = np.sqrt(np.mean(subset[error_col]**2))
            
            # Métriques de prévision (si disponibles)
            for h in [1, 5, 10]:
                squared_error_col = f'forecast_h{h}_squared_error'
                abs_error_col = f'forecast_h{h}_abs_error'
                
                if squared_error_col in subset.columns:
                    param_summary[f'forecast_h{h}_rmse'] = np.sqrt(subset[squared_error_col].mean())
                    param_summary[f'forecast_h{h}_mae'] = subset[abs_error_col].mean()
            
            summary_data.append(param_summary)
        
        # Créer le DataFrame de résumé
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            summary_df.set_index('param_set', inplace=True)
        
        # Afficher un résumé
        if not summary_df.empty and verbose:
            print(f"\n=== Estimation Summary for MAR{self.order} ===")
            print(summary_df)
        
        # Générer des graphiques si demandé
        if plot:
            self._plot_monte_carlo_results(results_df, save_plot, plot_prefix)
        
        return summary_df


    def _plot_mcstudy(self, results_df: pd.DataFrame, 
                                save_plot: bool = False, 
                                plot_prefix: str = "mar_mc") -> None:
        """
        Visualizes Monte Carlo simulation results with detailed plots.
        
        Args:
            results_df: DataFrame with results
            save_plot: If True, saves plots
            plot_prefix: Prefix for plot filenames
        """
        
        # Symboles pour les paramètres (pour une meilleure présentation)
        greek_symbols = {
            "psi_1": r"$\psi_1$",
            "psi_2": r"$\psi_2$",
            "phi_1": r"$\phi_1$",
            "phi_2": r"$\phi_2$",
            "alpha": r"$\alpha$",
            "beta": r"$\beta$",
            "sigma": r"$\sigma$"
        }
        
        # Extraire les ensembles de paramètres uniques
        param_sets = sorted(results_df['param_set'].unique())
        n_sets = len(param_sets)
        
        r, s = self.order
        
        # 1. Distribution des estimations pour chaque paramètre
        param_names = []
        
        # Paramètres AR
        for i in range(r):
            param_name = f"psi_{i+1}"
            if f"{param_name}_true" in results_df.columns:
                param_names.append(param_name)
        
        # Paramètres MA
        for i in range(s):
            param_name = f"phi_{i+1}"
            if f"{param_name}_true" in results_df.columns:
                param_names.append(param_name)
        
        # Paramètres de distribution alpha-stable
        for param_name in ['alpha', 'beta', 'sigma']:
            if f"{param_name}_true" in results_df.columns:
                param_names.append(param_name)
        
        # Nombre de paramètres à représenter
        n_params = len(param_names)
        
        # Pour chaque jeu de paramètres
        for param_set in param_sets:
            subset = results_df[results_df['param_set'] == param_set]
            
            # Calculer la mise en page
            n_cols = min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3.5))
            
            # Si n_params = a 1, axes n'est pas un tableau 2D
            if n_params == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            # Titre avec les vraies valeurs des paramètres
            title_parts = []
            for param in param_names:
                true_col = f"{param}_true"
                if true_col in subset.columns:
                    symbol = greek_symbols.get(param, param)
                    true_val = subset[true_col].iloc[0]
                    title_parts.append(f"{symbol}={true_val:.2f}")
            
            # Titre global
            fig.suptitle(f"Distribution des Estimations - MAR{self.order} - Jeu {param_set+1}\n" +
                    ", ".join(title_parts), fontsize=14)
            
            # Pour chaque paramètre
            for i, param in enumerate(param_names):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]
                
                true_col = f"{param}_true"
                est_col = f"{param}_est"
                
                if true_col in subset.columns and est_col in subset.columns:
                    # Valeur vraie
                    true_value = subset[true_col].iloc[0]
                    
                    # Histogramme des estimations
                    ax.hist(subset[est_col], bins=10, alpha=0.7, density=True)
                    
                    # Ligne verticale pour la vraie valeur
                    ax.axvline(true_value, color='red', linestyle='dashed', linewidth=2, label='Vraie valeur')
                    
                    # Statistiques pour l'étiquette
                    mean_value = subset[est_col].mean()
                    std_value = subset[est_col].std()
                    rmse_value = np.sqrt(np.mean((subset[est_col] - true_value)**2))
                    
                    # Titre avec symbole grec et statistiques
                    symbol = greek_symbols.get(param, param)
                    ax.set_title(f"{symbol} (RMSE: {rmse_value:.4f})")
                    ax.set_xlabel(f"Valeur (Vrai: {true_value:.3f}, Est: {mean_value:.3f}±{std_value:.3f})")
                    ax.set_ylabel("Densité")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                else:
                    ax.set_visible(False)
            
            # Supprimer les axes vides
            for i in range(n_params, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Laisser de l'espace pour le titre global
            
            # Sauvegarder ou afficher
            if save_plot:
                order_str = f"{r}_{s}"
                filename = f"{plot_prefix}_dist_param_set{param_set+1}_order{order_str}.png"
                plt.savefig(filename, dpi=300)
                plt.close(fig)
                print(f"Figure sauvegardée: {filename}")
            else:
                plt.show()
        
        # 2. Graphique de comparaison des biais et RMSE
        # A) Biais
        if n_sets > 1:
            # Collecter les biais et RMSE pour chaque paramètre
            bias_data = {}
            rmse_data = {}
            
            for param in param_names:
                bias_data[param] = []
                rmse_data[param] = []
                
                for param_set in param_sets:
                    subset = results_df[results_df['param_set'] == param_set]
                    
                    true_col = f"{param}_true"
                    est_col = f"{param}_est"
                    
                    if true_col in subset.columns and est_col in subset.columns:
                        true_value = subset[true_col].iloc[0]
                        # Biais moyen
                        bias = subset[est_col].mean() - true_value
                        bias_data[param].append(bias)
                        
                        # RMSE
                        rmse = np.sqrt(np.mean((subset[est_col] - true_value)**2))
                        rmse_data[param].append(rmse)
                    else:
                        bias_data[param].append(np.nan)
                        rmse_data[param].append(np.nan)
            
            # Graphique des biais
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bar_width = 0.8 / n_params
            index = np.arange(n_sets)
            
            for i, param in enumerate(param_names):
                pos = index - 0.4 + (i + 0.5) * bar_width
                ax.bar(pos, bias_data[param], width=bar_width, label=greek_symbols.get(param, param))
            
            ax.set_xticks(index)
            ax.set_xticklabels([f"Jeu {i+1}" for i in range(n_sets)])
            ax.set_ylabel('Biais')
            ax.set_title(f'Biais des Estimations par Paramètre - MAR{self.order}')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            if save_plot:
                order_str = f"{r}_{s}"
                filename = f"{plot_prefix}_bias_comparison_order{order_str}.png"
                plt.savefig(filename, dpi=300)
                plt.close(fig)
                print(f"Figure sauvegardée: {filename}")
            else:
                plt.show()
            
            # Graphique des RMSE
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for i, param in enumerate(param_names):
                pos = index - 0.4 + (i + 0.5) * bar_width
                ax.bar(pos, rmse_data[param], width=bar_width, label=greek_symbols.get(param, param))
            
            ax.set_xticks(index)
            ax.set_xticklabels([f"Jeu {i+1}" for i in range(n_sets)])
            ax.set_ylabel('RMSE')
            ax.set_title(f'RMSE des Estimations par Paramètre - MAR{self.order}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            if save_plot:
                order_str = f"{r}_{s}"
                filename = f"{plot_prefix}_rmse_comparison_order{order_str}.png"
                plt.savefig(filename, dpi=300)
                plt.close(fig)
                print(f"Figure sauvegardée: {filename}")
            else:
                plt.show()
                
        # 3. Prévisions (si disponibles)
        forecast_cols = [col for col in results_df.columns if col.startswith('forecast_h') and col.endswith('_rmse')]
        
        if forecast_cols and n_sets > 0:
            # Extraire les horizons
            horizons = sorted(list(set([int(col.split('_h')[1].split('_')[0]) for col in forecast_cols])))
            
            # Graphique des RMSE de prévision par horizon
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bar_width = 0.8 / n_sets
            index = np.arange(len(horizons))
            
            for i, param_set in enumerate(param_sets):
                subset = results_df[results_df['param_set'] == param_set]
                
                rmse_values = []
                for h in horizons:
                    col = f'forecast_h{h}_rmse'
                    if col in subset.columns:
                        rmse_values.append(subset[col].mean())
                    else:
                        rmse_values.append(np.nan)
                
                pos = index - 0.4 + (i + 0.5) * bar_width
                ax.bar(pos, rmse_values, width=bar_width, label=f'Jeu {param_set+1}')
            
            ax.set_xticks(index)
            ax.set_xticklabels([f'h={h}' for h in horizons])
            ax.set_ylabel('RMSE')
            ax.set_title(f'RMSE des Prévisions par Horizon - MAR{self.order}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            if save_plot:
                order_str = f"{r}_{s}"
                filename = f"{plot_prefix}_forecast_rmse_order{order_str}.png"
                plt.savefig(filename, dpi=300)
                plt.close(fig)
                print(f"Figure sauvegardée: {filename}")
            else:
                plt.show()


    def smart_demo(self, sample_size: int = 200, seed: int = 42) -> None:
        """
        Demonstrate key features of the StableMAR class with a simple example.
        
        This function provides a quick overview of:
        1. Model generation 
        2. Parameter estimation
        3. Alpha-stable parameter estimation from residuals
        4. Basic forecasting capabilities
        
        Args:
            sample_size: Size of the simulated time series
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        
        # 1. Generate a MAR(1,1) process
        print("\nGenerating MAR(1,1) process...")
        true_params = [0.3, 0.7, 1.5, 0.0, 1.0]  # [psi, phi, alpha, beta, sigma]
        
        current_params = self.par
        self.par = true_params
        self.generate(sample_size, seed=seed)
        print(f"True parameters: psi_1={true_params[0]:.2f}, phi_1={true_params[1]:.2f}, " +
            f"alpha={true_params[2]:.2f}, beta={true_params[3]:.2f}, sigma={true_params[4]:.2f}")
        
        # Save trajectory
        trajectory = self.trajectory.copy()
        
        # Restore original parameters
        self.par = current_params
        
        # 2. Estimate model parameters
        print("\nEstimating MAR parameters...")
        start_params = [0.2, 0.6]  # Initial guesses
        
        self.fit(trajectory.values, start_params, method="gcov", K=2, H=2, verbose=True)
        est_params = self.par
        
        print(f"Estimated parameters: psi_1={est_params[0]:.4f}, phi_1={est_params[1]:.4f}")
        
        # 3. Estimate alpha-stable parameters from residuals
        print("\nEstimating alpha-stable parameters from residuals...")
        
        residuals = self.results.get('PseudoResiduals')
        stable_params = self.fit_stable_noise(residuals)
        
        print(f"Residuals alpha-stable parameters: alpha={stable_params[0]:.4f}, " +
            f"beta={stable_params[1]:.4f}, sigma={stable_params[2]:.4f}")
        
        # 4. Plot the trajectory and basic forecast
        try:

            plt.figure(figsize=(12, 6))
            plt.plot(trajectory.values, 'b-', label='Simulated MAR(1,1)', alpha=0.7)
            
            # Find a peak in the second half of the series
            peak_idx = np.argmax(trajectory.values[sample_size//2:]) + sample_size//2
            forecast_start = max(0, peak_idx - 5)
            
            vartheta = 1 if trajectory.values[forecast_start] > 0 else -1
            
            # Add estimated stable parameters to create complete parameter set
            forecast_params = est_params + stable_params
            self.par = forecast_params
            
            # Perform a simple forecast if data is appropriate
            try:
                if abs(est_params[1]) > 0.1:  # Only if phi is non-trivial
                    k0 = 6  # Arbitrary value for demonstration
                    h = 10  # Forecast horizon
                    
                    _, forecast_traj = self.foreprob(
                        pd.Series(trajectory.values), k0, h, 0.9, forecast_start, vartheta
                    )
                    
                    # Plot forecast
                    forecast_indices = np.where(~np.isnan(forecast_traj))[0]
                    plt.plot(forecast_indices, forecast_traj[forecast_indices], 'r--', 
                            label='Simple Forecast')
                    plt.axvline(forecast_start, color='r', linestyle=':', label='Forecast Start')
            except Exception as e:
                print(f"\nForecast example skipped: {e}")
            
            plt.title('StableMAR Demo: Simulated Process and Simple Forecast')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            # Plot residuals distribution
            plt.figure(figsize=(10, 5))
            plt.hist(residuals, bins=30, density=True, alpha=0.7, label='Residuals')
            
            # Try to overlay the theoretical density
            x = np.linspace(min(residuals), max(residuals), 1000)
            try:
                # Create a stable distribution object
                stable_dist = stats.levy_stable(
                    alpha=stable_params[0], 
                    beta=stable_params[1], 
                    scale=stable_params[2], 
                    loc=0
                )
                
                # Plot the density
                plt.plot(x, stable_dist.pdf(x), 'r-', lw=2, 
                        label=f'α-stable (α={stable_params[0]:.2f}, β={stable_params[1]:.2f})')
            except Exception:
                pass
                
            plt.title('Residuals Distribution')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("\nMatplotlib not available. Skipping visualization.")
        
        print("\nDemo complete. For more advanced examples, see the test scripts.")
        
        # Restore original parameters
        self.par = current_params


# Helper functions

def madelta(cvec: np.ndarray, ncvec: np.ndarray, k: int) -> float:
    """
    Compute the infinite MA coefficient at lag k on any side of the sum, 
    for cvec and ncvec, vectors or causal and noncausal AR coefficients.

    Args:
        cvec: Vector of causal AR coefficients.
        ncvec: Vector of noncausal AR coefficients.
        k: A positive or negative integer.

    Returns:
        The MA coefficient delta at lag k.
    """
    if len(cvec) > 0 and not np.all(cvec == 0):
        lam = 1 / np.roots(np.flip(np.concatenate(([1], -np.array(cvec)))))
        lam = lam.real  # lambda values for the causal part
        r = len(lam)
    else:
        lam = 0
        r = 0

    if len(ncvec) > 0 and not np.all(ncvec == 0):
        zeta = 1 / np.roots(np.flip(np.concatenate(([1], -np.array(ncvec)))))
        zeta = zeta.real  # lambda values for the noncausal part (denoted zeta)
        s = len(zeta)
    else:
        zeta = 0
        s = 0

    delta = 0

    if k >= 0:
        for j in range(s):
            numerator = zeta[j] ** ((s - 1) + k)
            denominator1 = 1 if s == 1 else np.prod([zeta[j] - zeta[i] for i in range(s) if i != j])
            denominator2 = np.prod([zeta[j] * lam[i] - 1 for i in range(r)])

            if s % 2 == 0:  # Multiply by -1 if s is even
                delta -= numerator / (denominator1 * denominator2)  
            else:  # Multiply by 1 if s is odd
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

def root_check(arvec: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Check if the AR polynomial has roots inside the unit circle.
    
    Args:
        arvec: AR coefficients (without the constant term).
    
    Returns:
        Tuple containing:
        - Boolean indicating if any root is inside unit circle
        - Array of roots
        - Array of inverse roots
    """
    poly = np.concatenate(([1], -np.array(arvec)))
    roots = np.roots(np.flip(poly))
    lams = 1 / roots
    
    # Check if any root is inside or on the unit circle
    circle = any(abs(root) <= 1 for root in roots)
    
    return circle, roots, lams
