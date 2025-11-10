# hw4.py

import pandas as pd
import numpy as np
from kmeans import KMeans
from gmm import GMM
import matplotlib.pyplot as plt


def plot_clustering_results(X, labels, centroids, title, ax=None, is_gmm=False):
    """ K-Means 및 GMM 결과 시각화를 위한 통합 함수 """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # 'tab10' 또는 다른 적절한 컬러맵 사용
    cmap = plt.get_cmap('tab10', n_clusters if n_clusters > 0 else 1)

    for i, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, 
                  color=cmap(i), label=f'Cluster {label}', alpha=0.7)

    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], s=250, marker='X', 
                  c='black', edgecolor='white', linewidth=1.5, label='Centroids/Means')

    ax.set_title(title)
    ax.set_xlabel('Weight (kg)')
    ax.set_ylabel('Length (cm)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # GMM의 경우 타원 그리기 (선택적 고급 기능)
    if is_gmm and hasattr(centroids, 'covariances_') and centroids.covariances_ is not None:
        # 이 부분은 GMM 객체(centroids 인자에 GMM model 전달)와 공분산 정보를 사용해 타원을 그리는 로직
        # 예시: from matplotlib.patches import Ellipse
        # for i in range(centroids.k):
        #    mean = centroids.means_[i]
        #    cov = centroids.covariances_[i]
        #    # ... (eigenvalue/vector로 타원 파라미터 계산 및 그리기) ...
        pass # 지금은 생략

    if not ax: # 단독 호출 시 plt.show()
        plt.show()


def main():
    # --- 1. 데이터셋 로드 ---
    try:
        df = pd.read_csv("hw4_dataset.csv", header=0)
        X = df.to_numpy()
        print(f"데이터셋 로드 완료. 형태: {X.shape}")
    except FileNotFoundError:
        print("오류: hw4_dataset.csv 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"데이터셋 로드 중 오류 발생: {e}")
        return

    # --- 2. 초기 데이터 시각화 ---
    plt.figure(figsize = (8, 6))
    scatter_init = plt.scatter(X[:, 0], X[:, 1], s = 50, c = X[:, 1], cmap = "viridis")
    plt.colorbar(scatter_init, label = "Length(cm)")
    plt.title("Raw data (Weight vs. Length)")
    plt.xlabel("Weight (kg)")
    plt.ylabel("Length (cm)")
    plt.show()

    # --- 3. K-Means 실행 (최종 k=5 사용) ---
    final_k_value_kmeans = 5 
    print(f"\n--- 최종 K-Means 클러스터링 시작 (k={final_k_value_kmeans}) ---")
    kmeans_model = KMeans(k=final_k_value_kmeans)
    try:
        kmeans_model.fit(X)
        print(f"K={final_k_value_kmeans}, K-Means 모델 학습 완료.")
        # K-Means 결과 시각화 (별도 창)
        plot_clustering_results(X, kmeans_model.labels, kmeans_model.centroids, 
                                title=f'K-Means Clustering (k={final_k_value_kmeans})')
        plt.show() # 개별 plot 함수가 plt.show()를 호출하지 않는 경우 필요
    except Exception as e:
        print(f"K={final_k_value_kmeans}, K-Means 모델 학습 중 오류: {e}.")


    # --- 4. GMM 실행 ---
    # GMM에 사용할 k 값 (K-Means와 같게 하거나 다르게 설정 가능)
    # 여기서는 K-Means에서 좋았던 k=5를 사용. 필요시 GMM에 대한 k값 탐색 별도 진행 가능
    final_k_value_gmm = 5 
    print(f"\n--- GMM 클러스터링 시작 (k={final_k_value_gmm}) ---")

    gmm_model = GMM(k=final_k_value_gmm, max_iters=150, tol=1e-4, reg_covar=1e-6, init_params='random')
    try:
        gmm_model.fit(X) # GMM 모델 학습
        if gmm_model.converged_:
            print(f"K={final_k_value_gmm}, GMM 모델 학습 완료 (수렴).")
            # 수렴 시에는 log_likelihood_history_가 비어있지 않다고 가정 가능
            print(f"  최종 Log-Likelihood: {gmm_model.log_likelihood_history_[-1]:.4f}")
        else: # 최대 반복 도달 시
            print(f"K={final_k_value_gmm}, GMM 모델 학습 완료 (최대 반복 도달).")
            if gmm_model.log_likelihood_history_: # 리스트가 비어있지 않은지 확인
                print(f"  최종 Log-Likelihood: {gmm_model.log_likelihood_history_[-1]:.4f}")
            else:
                print(f"  최종 Log-Likelihood: N/A (기록 없음)") # 리스트가 비어있으면 'N/A' 출력

        # GMM 결과 레이블 예측
        gmm_labels = gmm_model.predict(X)

        # GMM 결과 시각화 (별도 창, K-Means와 동일한 플로팅 함수 사용, GMM 평균을 centroid로 전달)
        plot_clustering_results(X, gmm_labels, gmm_model.means_, 
                                title=f'GMM Clustering (k={final_k_value_gmm})', is_gmm=True) # is_gmm=True로 타원 그리기 로직 활성화 가능
        plt.show() # 개별 plot 함수가 plt.show()를 호출하지 않는 경우 필요
        
        # (선택 사항) GMM 로그 가능도 변화 시각화
        plt.figure(figsize=(8, 5))
        plt.plot(gmm_model.log_likelihood_history_)
        plt.title(f'GMM Log-Likelihood vs. Iterations (k={final_k_value_gmm})')
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"K={final_k_value_gmm}, GMM 모델 실행 중 오류 발생: {e}.")
    # --- [추가 종료] ---

    print("\n--- 모든 메인 코드 실행 완료 ---")

if __name__ == "__main__":
  main()