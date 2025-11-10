# gmm.py

import numpy as np
from scipy.stats import multivariate_normal # 다변량 정규분포 PDF 계산용
# K-Means 결과를 초기화에 사용하려면 KMeans 임포트 필요 (선택적 고급 초기화)
# from kmeans import KMeans

class GMM:
    def __init__(self, k, max_iters=150, tol=1e-4, reg_covar=1e-6, init_params='random'):
        """
        가우시안 혼합 모델 (GMM) 초기화
        Args:
            k (int): 가우시안 요소(클러스터)의 개수
            max_iters (int): EM 알고리즘의 최대 반복 횟수
            tol (float): 로그 가능도(log-likelihood) 변화량에 대한 수렴 허용 오차
            reg_covar (float): 공분산 행렬의 대각선에 더해질 작은 값 (특이성 방지용)
            init_params (str): 초기화 방법. 'random' 또는 'kmeans' (kmeans는 직접 구현 필요)
        """
        if k <= 0:
            raise ValueError("k (number of components) must be a positive integer.")
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.reg_covar = reg_covar
        self.init_params = init_params # 초기화 방법 저장

        self.means_ = None
        self.covariances_ = None
        self.weights_ = None # 혼합 계수 (파이 값들)
        self.log_likelihood_history_ = []
        self.converged_ = False

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape

        if n_samples < self.k:
            raise ValueError(f"Number of samples ({n_samples}) must be at least k ({self.k}).")

        if self.init_params == 'random':
            # 1. 평균(means_) 초기화: 데이터에서 무작위로 k개의 점 선택
            random_indices = np.random.choice(np.arange(n_samples), size=self.k, replace=False)
            self.means_ = X[random_indices]

            # 2. 공분산(covariances_) 초기화: 모든 요소에 대해 동일한 초기 공분산 사용
            self.covariances_ = np.zeros((self.k, n_features, n_features))
            data_cov = np.cov(X, rowvar=False)
            if n_features == 1 and np.isscalar(data_cov): # 1D 데이터 처리
                data_cov = np.array([[data_cov]])
            
            # 모든 대각 요소가 reg_covar보다 크도록 보장 (data_cov가 0에 가까울 수 있음)
            # initial_covariance = data_cov + np.eye(n_features) * self.reg_covar
            # 좀 더 안정적인 방법: 단위행렬에 전체 분산의 평균을 곱하고 reg_covar 더하기
            diag_val = np.var(X, axis=0).mean() # 전체 분산의 평균, 또는 그냥 1.0 사용
            initial_covariance = np.eye(n_features) * diag_val + np.eye(n_features) * self.reg_covar


            for i in range(self.k):
                self.covariances_[i] = np.copy(initial_covariance)

            # 3. 혼합 계수(weights_) 초기화: 균등하게 1/k
            self.weights_ = np.full(self.k, 1/self.k)

        else:
            raise ValueError(f"Unsupported init_params: {self.init_params}. Choose 'random' or implement 'kmeans'.")


    def _e_step(self, X):
        n_samples = X.shape[0]
        weighted_probs = np.zeros((n_samples, self.k))

        for j in range(self.k):
            mean_j = self.means_[j]
            cov_j = self.covariances_[j]
            weight_j = self.weights_[j]
            
            try:
                # 안정성을 위해 allow_singular=True를 사용하거나, cov_j가 항상 양의 정부호인지 확인
                pdf_values_j = multivariate_normal.pdf(X, mean=mean_j, cov=cov_j, allow_singular=True)
            except np.linalg.LinAlgError: # 공분산 행렬이 여전히 문제 있을 경우
                # print(f"Warning: LinAlgError for component {j} in E-step. Using epsilon probabilities.")
                pdf_values_j = np.full(n_samples, np.finfo(float).eps) # 매우 작은 값
            except ValueError as ve: # mean, cov 차원 불일치 등
                # print(f"Warning: ValueError for component {j} in E-step: {ve}. Using epsilon probabilities.")
                pdf_values_j = np.full(n_samples, np.finfo(float).eps)

            weighted_probs[:, j] = weight_j * pdf_values_j

        epsilon = np.finfo(float).eps
        likelihood_per_point_sum = np.sum(weighted_probs, axis=1)
        
        current_log_likelihood = np.sum(np.log(likelihood_per_point_sum + epsilon))

        responsibilities = weighted_probs / (likelihood_per_point_sum[:, np.newaxis] + epsilon)
        
        return responsibilities, current_log_likelihood

    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        
        # N_k: 각 컴포넌트 k에 할당된 유효 데이터 포인트 수 (responsibilities의 합)
        N_k = np.sum(responsibilities, axis=0) # shape: (k,)
        epsilon_Nk = 1e-9 # N_k가 0이 되는 것을 방지하기 위한 작은 값

        # 1. 혼합 계수(weights_) 업데이트
        self.weights_ = (N_k + epsilon_Nk) / (n_samples + self.k * epsilon_Nk) # 합이 1이 되도록 정규화
        self.weights_ = self.weights_ / np.sum(self.weights_) # 확실하게 합이 1이 되도록

        # 2. 평균(means_) 업데이트
        # weighted_sum_X = np.dot(responsibilities.T, X) # (k, n_samples) @ (n_samples, n_features) -> (k, n_features)
        # self.means_ = weighted_sum_X / (N_k[:, np.newaxis] + epsilon_Nk)
        for j in range(self.k):
            if N_k[j] > epsilon_Nk: # 해당 컴포넌트에 충분한 responsibility가 할당된 경우
                 self.means_[j] = np.sum(responsibilities[:, j, np.newaxis] * X, axis=0) / N_k[j]
            # else: # N_k[j]가 매우 작으면 평균을 업데이트하지 않거나 (이전 값 유지) 재초기화. 여기서는 유지.
            #    print(f"Warning: N_k for component {j} is very small ({N_k[j]:.2e}). Mean not updated significantly.")


        # 3. 공분산(covariances_) 업데이트
        for j in range(self.k):
            if N_k[j] > epsilon_Nk: # 해당 컴포넌트에 충분한 responsibility가 할당된 경우
                diff = X - self.means_[j]  # (n_samples, n_features)
                
                # (n_samples, n_features, 1) * (n_samples, 1, n_features) -> (n_samples, n_features, n_features)
                # 각 샘플에 대한 외적 (outer product) 계산 후 가중합
                weighted_sum_diff_outer = np.zeros((n_features, n_features))
                for i in range(n_samples):
                    resp_ni = responsibilities[i, j]
                    diff_i = diff[i, :, np.newaxis] # (n_features, 1)
                    weighted_sum_diff_outer += resp_ni * (diff_i @ diff_i.T)
                
                self.covariances_[j] = weighted_sum_diff_outer / N_k[j]
                # 정규화(regularization) 추가: 공분산 행렬의 대각선에 작은 값을 더함
                self.covariances_[j] += np.eye(n_features) * self.reg_covar
            # else: # N_k[j]가 매우 작으면 공분산 업데이트하지 않거나 (이전 값 유지) 재초기화. 여기서는 유지.
            #    print(f"Warning: N_k for component {j} is very small ({N_k[j]:.2e}). Covariance not updated significantly.")


    def fit(self, X):
        if X.ndim != 2:
            raise ValueError("Input data X must be 2-dimensional.")
        if X.shape[0] < self.k: # 샘플 수가 컴포넌트 수보다 적으면 안됨
             raise ValueError(f"Number of samples ({X.shape[0]}) must be at least k ({self.k}).")

        self._initialize_parameters(X)
        self.log_likelihood_history_ = []
        self.converged_ = False

        for i in range(self.max_iters):
            try:
                responsibilities, current_log_likelihood = self._e_step(X)
            except np.linalg.LinAlgError as e:
                print(f"LinAlgError during E-step at iteration {i+1}: {e}. Stopping.")
                break # E-step에서 심각한 오류 발생 시 중단
            except ValueError as e:
                print(f"ValueError during E-step at iteration {i+1}: {e}. Stopping.")
                break

            self.log_likelihood_history_.append(current_log_likelihood)

            # M-step
            self._m_step(X, responsibilities)

            # 수렴 확인
            if i > 0:
                log_likelihood_change = self.log_likelihood_history_[i] - self.log_likelihood_history_[i-1]
                if log_likelihood_change < 0 and not np.isclose(self.log_likelihood_history_[i], self.log_likelihood_history_[i-1]):
                    print(f"Warning: Log-likelihood decreased at iteration {i+1} from {self.log_likelihood_history_[i-1]:.4f} to {self.log_likelihood_history_[i]:.4f}.")
                    # GMM의 로그 가능도는 이론적으로 감소하지 않아야 하지만, 수치적 문제나 구현 오류로 발생 가능
                
                if abs(log_likelihood_change) < self.tol:
                    print(f"Converged at iteration {i+1}. Log-likelihood change ({log_likelihood_change:.6f}) < tol ({self.tol}).")
                    self.converged_ = True
                    break
            
            if i == self.max_iters - 1:
                print(f"Reached maximum GMM iterations ({self.max_iters}) without convergence based on tol. Last log-likelihood change: {log_likelihood_change if i > 0 else 'N/A'}")
        
        # print("GMM fitting finished.")


    def predict_proba(self, X):
        if self.means_ is None or self.covariances_ is None or self.weights_ is None:
            raise ValueError("Model not fitted yet! Call fit() first.")
        if X.ndim != 2:
            raise ValueError("Input data X for predict_proba must be 2-dimensional.")
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError(f"Number of features in X ({X.shape[1]}) does not match model features ({self.means_.shape[1]}).")

        # E-step과 거의 동일한 로직이지만, 로그 가능도는 반환하지 않음
        n_samples = X.shape[0]
        weighted_probs = np.zeros((n_samples, self.k))

        for j in range(self.k):
            mean_j = self.means_[j]
            cov_j = self.covariances_[j]
            weight_j = self.weights_[j]
            try:
                pdf_values_j = multivariate_normal.pdf(X, mean=mean_j, cov=cov_j, allow_singular=True)
            except (np.linalg.LinAlgError, ValueError):
                pdf_values_j = np.full(n_samples, np.finfo(float).eps)
            weighted_probs[:, j] = weight_j * pdf_values_j
        
        epsilon = np.finfo(float).eps
        likelihood_per_point_sum = np.sum(weighted_probs, axis=1)
        responsibilities = weighted_probs / (likelihood_per_point_sum[:, np.newaxis] + epsilon)
        
        return responsibilities

    def predict(self, X):
        responsibilities = self.predict_proba(X)
        return np.argmax(responsibilities, axis=1)