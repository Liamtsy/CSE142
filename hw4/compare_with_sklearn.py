import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 직접 구현한 모델 임포트
from kmeans import KMeans
from gmm import GMM

# Scikit-learn 모델 임포트
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.mixture import GaussianMixture as SklearnGMM

def plot_simple_clusters(X, labels, centers_or_means, title, ax, model_name="Custom"):
    """
    타원 없이 클러스터링 결과를 간단히 시각화합니다.
    K-Means의 centroids 또는 GMM의 means_를 centers_or_means로 받습니다.
    """
    unique_labels = np.unique(labels)
    
    # 레이블 개수 또는 centers/means 개수를 기준으로 색상 수 결정
    k_for_color = centers_or_means.shape[0] if centers_or_means is not None else len(unique_labels)
    if k_for_color == 0 and len(unique_labels) > 0 : # labels만 있고 centers 정보가 없을 경우
        k_for_color = len(unique_labels)
    if k_for_color == 0: k_for_color = 1 # 최소 1개 색상

    cmap = plt.get_cmap('tab10', k_for_color)
    legend_handles = []

    # 데이터 포인트 그리기
    sorted_unique_labels = np.sort(unique_labels) # 일관된 색상 매핑을 위해 정렬
    for i, label_val in enumerate(sorted_unique_labels):
        color_idx = i % k_for_color # 색상 인덱스가 cmap 범위를 넘지 않도록
        current_color = cmap(color_idx)
        cluster_points = X[labels == label_val]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=30, 
                  color=current_color, alpha=0.6)
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        label=f'Cluster {label_val}',
                                        markerfacecolor=current_color, markersize=8))

    # 중심점 또는 평균점 그리기
    if centers_or_means is not None:
        marker_style = 'P' if "Sklearn" in model_name else 'X'
        center_color = 'red' if "Sklearn" in model_name else 'black'
        center_label_text = f'{model_name} Centers/Means'
        
        ax.scatter(centers_or_means[:, 0], centers_or_means[:, 1], 
                  s=250, marker=marker_style, c=center_color, 
                  edgecolor='white', linewidth=1.5)
        legend_handles.append(plt.Line2D([0], [0], marker=marker_style, color='w', 
                                  label=center_label_text,
                                  markerfacecolor=center_color, markersize=10))

    ax.set_title(title)
    ax.set_xlabel('Weight (kg)')
    ax.set_ylabel('Length (cm)')
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize='small', framealpha=0.7)
    ax.grid(True, alpha=0.3)

def run_comparisons():
    # --- 데이터셋 로드 ---
    try:
        df = pd.read_csv("hw4_dataset.csv", header=0)
        X = df.to_numpy()
        print(f"데이터셋 로드 완료 (비교용). 형태: {X.shape}")
    except FileNotFoundError:
        print("오류 (비교용): hw4_dataset.csv 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"데이터셋 로드 중 오류 발생 (비교용): {e}")
        return

    chosen_k = 5  # 비교에 사용할 k 값

    # --- K-Means 비교 ---
    print(f"\n--- K-Means 비교 (k={chosen_k}) ---")
    my_kmeans_labels_comp, my_kmeans_centers_comp = None, None
    try:
        my_kmeans_comp = KMeans(k=chosen_k)
        my_kmeans_comp.fit(X)
        my_kmeans_labels_comp = my_kmeans_comp.labels
        my_kmeans_centers_comp = my_kmeans_comp.centroids
        print("My K-Means (for comparison) 학습 완료.")
    except Exception as e:
        print(f"My K-Means (for comparison) 학습 중 오류: {e}")

    sklearn_kmeans_labels_comp, sklearn_kmeans_centers_comp = None, None
    try:
        sklearn_kmeans_comp = SklearnKMeans(n_clusters=chosen_k, n_init='auto', random_state=42)
        sklearn_kmeans_comp.fit(X)
        sklearn_kmeans_labels_comp = sklearn_kmeans_comp.labels_
        sklearn_kmeans_centers_comp = sklearn_kmeans_comp.cluster_centers_
        print("Sklearn K-Means (for comparison) 학습 완료.")
    except Exception as e:
        print(f"Sklearn K-Means (for comparison) 학습 중 오류: {e}")

    if my_kmeans_labels_comp is not None and sklearn_kmeans_labels_comp is not None:
        fig_kmeans_comp, axes_kmeans_comp = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        plot_simple_clusters(X, my_kmeans_labels_comp, my_kmeans_centers_comp,
                            title=f'My K-Means (k={chosen_k})', ax=axes_kmeans_comp[0], model_name="My K-Means")
        plot_simple_clusters(X, sklearn_kmeans_labels_comp, sklearn_kmeans_centers_comp,
                            title=f'Sklearn K-Means (k={chosen_k})', ax=axes_kmeans_comp[1], model_name="Sklearn K-Means")
        fig_kmeans_comp.suptitle(f"K-Means Implementation Comparison (k={chosen_k})", fontsize=16)
        plt.show()

    # --- GMM 비교 ---
    print(f"\n--- GMM 비교 (k={chosen_k}) ---")
    my_gmm_labels_comp, my_gmm_means_comp, my_gmm_loglik_comp = None, None, "N/A"
    try:
        my_gmm_comp = GMM(k=chosen_k, max_iters=150, tol=1e-4, reg_covar=1e-6, init_params='random')
        my_gmm_comp.fit(X)
        my_gmm_labels_comp = my_gmm_comp.predict(X)
        my_gmm_means_comp = my_gmm_comp.means_
        if my_gmm_comp.log_likelihood_history_:
            my_gmm_loglik_comp = f"{my_gmm_comp.log_likelihood_history_[-1]:.2f}"
        print(f"My GMM (for comparison) 학습 완료. Converged: {my_gmm_comp.converged_}, LogLik: {my_gmm_loglik_comp}")
    except Exception as e:
        print(f"My GMM (for comparison) 학습 중 오류: {e}")

    sklearn_gmm_labels_comp, sklearn_gmm_means_comp, sklearn_gmm_loglik_comp = None, None, "N/A"
    try:
        sklearn_gmm_comp = SklearnGMM(n_components=chosen_k, covariance_type='full',
                                      max_iter=150, tol=1e-4, reg_covar=1e-6,
                                      init_params='kmeans', n_init=1, random_state=42)
        sklearn_gmm_comp.fit(X)
        sklearn_gmm_labels_comp = sklearn_gmm_comp.predict(X)
        sklearn_gmm_means_comp = sklearn_gmm_comp.means_
        sklearn_total_loglik_val = sklearn_gmm_comp.score(X) * X.shape[0]
        sklearn_gmm_loglik_comp = f"{sklearn_total_loglik_val:.2f}"
        print(f"Sklearn GMM (for comparison) 학습 완료. Converged: {sklearn_gmm_comp.converged_}, LogLik (total): {sklearn_gmm_loglik_comp}")
    except Exception as e:
        print(f"Sklearn GMM (for comparison) 학습 중 오류: {e}")
        
    if my_gmm_labels_comp is not None and sklearn_gmm_labels_comp is not None:
        fig_gmm_comp, axes_gmm_comp = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        plot_simple_clusters(X, my_gmm_labels_comp, my_gmm_means_comp,
                            title=f'My GMM (k={chosen_k}), LogLik: {my_gmm_loglik_comp}', 
                            ax=axes_gmm_comp[0], model_name="My GMM")
        plot_simple_clusters(X, sklearn_gmm_labels_comp, sklearn_gmm_means_comp,
                            title=f'Sklearn GMM (k={chosen_k}), LogLik: {sklearn_gmm_loglik_comp}', 
                            ax=axes_gmm_comp[1], model_name="Sklearn GMM")
        fig_gmm_comp.suptitle(f"GMM Implementation Comparison (k={chosen_k})", fontsize=16)
        plt.show()

if __name__ == '__main__':
    # plot_simple_clusters 함수를 이 파일 위에 정의하거나 여기서 임포트해야 합니다.
    # 예시: 위에 정의했다고 가정
    run_comparisons()