import numpy as np
from scipy.linalg import eigh, qr
from scipy.signal import welch, csd

class ForeCA:
    """
    ForeCA (Forecasting Component Analysis) implementation for finding forecasting components
    in time series data using spectral entropy minimization.
    
    Parameters:
    -----------
    n_components : int, optional (default=1)
        Number of forecasting components to extract.
        
    nfft : int, optional (default=512)
        Number of FFT points for spectral estimation.
        
    window : int or str or tuple, optional (default=None)
        Window type and size for spectral estimation. If None, uses nfft.
        
    overlap : int, optional (default=None)
        Number of points to overlap between segments. If None, uses window/2.
        
    fs : float, optional (default=1)
        Sampling frequency of the input data.
        
    tol : float, optional (default=1e-10)
        Tolerance for convergence in the EM algorithm.
        
    max_iter : int, optional (default=1000)
        Maximum number of iterations for the EM algorithm.
    """
    
    def __init__(self, n_components=1, nfft=512, window=None, overlap=None, fs=1, tol=1e-10, max_iter=1000):
        self.n_components = n_components
        self.nfft = nfft
        self.window = window if window is not None else nfft
        self.overlap = overlap if overlap is not None else int(self.window / 2)
        self.fs = fs
        self.tol = tol
        self.max_iter = max_iter
        self.weights_ = None
        self.entropies_ = None
        
    def fit(self, X):
        """
        Fit the ForeCA model to the input data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input time series data.
            
        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input data must be 2-dimensional (n_samples, n_features)")
        
        if X.shape[0] < X.shape[1]:
            X = X.T
            print("Input data transposed to have n_samples >= n_features")
            
        if self.n_components > X.shape[1]:
            raise ValueError("n_components must be <= n_features")
            
        self.weights_, self.entropies_ = self._foreca(
            X, 
            k=self.n_components,
            Nfft=self.nfft,
            Window=self.window,
            Overlap=self.overlap,
            fs=self.fs,
            tol=self.tol,
            max_iter=self.max_iter
        )
        return self
    
    def transform(self, X):
        """
        Transform the input data using the fitted ForeCA model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input time series data.
            
        Returns:
        --------
        X_transformed : array-like of shape (n_samples, n_components)
            Transformed data.
        """
        if self.weights_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return X @ self.weights_
    
    def fit_transform(self, X):
        """
        Fit the model and transform the input data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input time series data.
            
        Returns:
        --------
        X_transformed : array-like of shape (n_samples, n_components)
            Transformed data.
        """
        return self.fit(X).transform(X)
    
    @staticmethod
    def _inverse_sqrt_matrix_eigh(cov):
        """Compute inverse square root of a matrix using eigenvalue decomposition."""
        eigvals, eigvecs = eigh(cov)
        inv_sqrt_eigvals = np.diag(1.0 / np.sqrt(eigvals))
        cov_inv_sqrt = eigvecs @ inv_sqrt_eigvals @ eigvecs.T
        return cov_inv_sqrt
    
    @staticmethod
    def _whiten(series):
        """Whiten the input data."""
        covariance_matrix = np.cov(series, rowvar=False)
        inv_sqrt_cov = ForeCA._inverse_sqrt_matrix_eigh(covariance_matrix)
        whitened_data = series @ inv_sqrt_cov
        return whitened_data
    
    @staticmethod
    def _is_positive_semidefinite(matrix):
        """Check if a matrix is positive semidefinite."""
        eigenvalues = np.linalg.eigvals(matrix)
        return np.all(eigenvalues >= -1e-10)
    
    @staticmethod
    def _spectrumcalc(U, window='hann', overlap=None, nfft=256, fs=1):
        """Compute the cross-spectral density matrix."""
        freqs, spectrum = welch(U.T, fs=fs, nperseg=256, noverlap=overlap, 
                               nfft=nfft, axis=-1, return_onesided=False)
        
        n = len(spectrum)
        C = np.empty((len(spectrum), len(spectrum)), dtype=object)
        
        for i in range(len(spectrum)):
            C[i,i] = spectrum[i]
            
        for i in range(len(spectrum)):
            for j in range(i+1, len(spectrum)):
                _, ij = csd(U[:,i], U[:,j], fs=fs, nperseg=256, noverlap=overlap,
                           nfft=nfft, axis=-1, return_onesided=False)
                C[i,j] = ij
                C[j,i] = ij.conjugate()
                
        C_array = np.empty((n, n, spectrum.shape[1]), dtype=complex)
        for i in range(n):
            for j in range(n):
                C_array[i, j, :] = C[i, j]
                
        return C_array
    
    @staticmethod
    def _spectral_entropy(spectrum, weights):
        """Compute spectral entropy."""
        #if there are more than one weights, return a list of entropies
        if weights.ndim > 1:
            entropies = []
            for w in weights.T:
                entropies.append(ForeCA._spectral_entropy(spectrum, w))
            return entropies
        likelihoods = np.einsum("i,ijv,j->v", weights, spectrum, weights)
        likelihoods /= np.sum(likelihoods)
        
        if not ForeCA._check_positive_semidefinite(spectrum):
            raise ValueError("The spectrum is not positive semidefinite.")
            
        log_likelihoods = np.log(likelihoods)
        if np.any(log_likelihoods >= 0):
            raise ValueError("Log likelihoods must be negative for spectral entropy calculation.")
            
        T = len(log_likelihoods)
        h = np.real_if_close((-1/T) * np.dot(likelihoods, log_likelihoods), tol=1e-10)
        return h
    
    @staticmethod
    def _forecastability(entropy, base = 2):
        return 1 - entropy * np.log(base)/ np.log(2*np.pi)

    @staticmethod
    def _preprocessed_data(data):
        """Center and whiten the input data."""
        data = data - np.mean(data, axis=0)
        whitened_data = ForeCA._whiten(data)
        return whitened_data
    
    @staticmethod
    def _check_positive_semidefinite(S):
        """Check if all matrices in the spectrum are positive semidefinite."""
        for i in range(S.shape[2]):
            if not ForeCA._is_positive_semidefinite(S[:,:,i]):
                print(f"Spectrum at index {i} is not positive semidefinite.")
                return False
        return True
    
    @staticmethod
    def _EM(data, max_iter=100, window=None, overlap=None, nfft=512, fs=1, tol=1e-10):
        """Expectation-Maximization algorithm for finding forecasting components."""
        weights = np.random.uniform(-1, 1, len(data.T))
        weights = weights / np.linalg.norm(weights)
        entropies = []
        spectrum = ForeCA._spectrumcalc(data, window=window, overlap=overlap, nfft=nfft, fs=fs)
        h_old = 100
        print(f"Spectrum: {spectrum[0]}")
        for i in range(max_iter):
            likelihoods = np.array([weights.T @ spectrum[:,:,j] @ weights for j in range(spectrum.shape[2])])
            likelihoods /= np.sum(likelihoods)
            likelihoods[likelihoods <= 0] = 1e-10
            
            if np.any(likelihoods <= 0):
                print("Warning: Some likelihoods are zero or negative, replacing with small value.")
                
            log_likelihoods = np.log(likelihoods)
            if np.any(log_likelihoods >= 0):
                raise ValueError("Log likelihoods must be negative for spectral entropy calculation.")
                
            T = spectrum.shape[2]
            S_U = np.zeros((len(spectrum), len(spectrum)), dtype=complex)
            for j in range(T):
                S_U -= spectrum[:,:,j] * log_likelihoods[j]
            S_U /= T
            
            eigenvalues, eigenvectors = eigh(S_U.real)
            newweights = eigenvectors[:, np.argmin(eigenvalues)]
            entropies.append(ForeCA._spectral_entropy(spectrum, newweights))
            
            h = -np.sum(likelihoods * np.log(likelihoods))
            if np.abs(h - h_old) < tol:
                print(f"Converged after {i} iterations")
                break
                
            h_old = h
            weights = newweights
            
            if i == max_iter - 1:
                print("Maximum iterations reached without convergence.")
                
        return weights, entropies
    
    @staticmethod
    def _project_to_nullspace(U, W, Wwhiten):
        """Project U onto the null space of W."""
        if W.shape[0] == 0:
            return U, None

        Wprev = W.T @ Wwhiten
        Q, _ = qr(Wprev.T, mode='full')
        Null = Q[:, Wprev.shape[0]:]
        return U @ Null, Null
    
    @staticmethod
    def _foreca(series, k, Nfft=512, Window=None, Overlap=None, fs=1, tol=1e-10, max_iter=1000):
        """Core ForeCA algorithm implementation."""
        if Window is None:
            Window = Nfft
        if Overlap is None:
            Overlap = int(Window / 2)
        series = series - series.mean(axis=0, keepdims=True)
        Sigma = np.cov(series, rowvar=False)
        D, E = eigh(Sigma)
        Wwhiten = E @ np.diag(1. / np.sqrt(D)) @ E.T
        
        # U = ForeCA._preprocessed_data(series)
        U = ForeCA._whiten(series)
        if not ForeCA._is_positive_semidefinite(U.T @ U):
            raise ValueError("Input data covariance matrix is not positive semidefinite.")
        Uoriginal = U.copy()
        print("Whitened data:\n", U[:3,:])
        weights, _ = ForeCA._EM(U, max_iter=max_iter, tol=tol)
        weights = weights.reshape(-1, 1)
        print('weights: ', weights)
        U, Null = ForeCA._project_to_nullspace(U, weights, Wwhiten)
        weights = Wwhiten @ weights
        # weights = weights/np.linalg.norm(weights)
        print(f"weights after: {weights}")
        for i in range(k-1):
            nextweight, _ = ForeCA._EM(U, max_iter=max_iter, tol=tol)
            temp = Wwhiten @ Null @ nextweight.reshape(-1, 1)
            temp = temp / np.linalg.norm(temp)
            weights = np.hstack((weights, temp))
            U, Null = ForeCA._project_to_nullspace(Uoriginal, weights, Wwhiten)
            
        return weights, None