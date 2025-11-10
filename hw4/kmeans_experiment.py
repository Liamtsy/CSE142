import pandas as pd
import numpy as np
from kmeans import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def main():
    # 1. Dataset Load
    df = pd.read_csv("hw4_dataset.csv", header = 0)
    X = df.to_numpy()

    # 2. Data Visualization
    plt.figure(figsize = (8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], s = 50, c = X[:, 1], cmap = "viridis")  # Mapping Length to a color "viridis"
    plt.colorbar(scatter, label = "Length(cm)")  # Adding a colorbar
    plt.title("Raw data (Weight vs. Length)")
    plt.xlabel("Weight (kg)")
    plt.ylabel("Length (cm)")
    plt.show()

    # 3. Setting up the parameters
    k_values_to_test = range(2, 9)
    wcss_scores = []
    silhouette_scores = []
    
    print("\n--- K-Means 클러스터 개수(k)에 따른 실험 시작 ---")

    # 4. 여러 k 값에 대해 K-Means 실행 및 평가
    for current_k in k_values_to_test: # k_value 대신 current_k 사용
        print(f"\n[ K = {current_k} ] 로 K-Means 실행 중...")
        
        kmeans_model = KMeans(k=current_k) # 현재 k 값으로 모델 생성
        try:
            kmeans_model.fit(X)
            print(f"  K={current_k}, K-means model is trained well.")

            # WCSS (Within-Cluster Sum of Squares) 계산
            current_wcss = 0
            # kmeans_model.labels와 kmeans_model.centroids가 None이 아닌지, 그리고 k가 0보다 큰지 확인
            if kmeans_model.labels is not None and kmeans_model.centroids is not None and kmeans_model.k > 0:
                for j in range(kmeans_model.k): # kmeans_model.k 사용
                    cluster_points = X[kmeans_model.labels == j]
                    if len(cluster_points) > 0:
                        centroid = kmeans_model.centroids[j]
                        current_wcss += np.sum((cluster_points - centroid)**2)
                wcss_scores.append(current_wcss)
                print(f"  K={current_k}, WCSS: {current_wcss:.2f}")
            else:
                wcss_scores.append(np.nan) # 학습 실패 또는 레이블/중심점 없음
                print(f"  K={current_k}, WCSS 계산 불가 (모델 학습 실패, 레이블/중심점 없음 또는 k=0)")


            # 실루엣 점수 계산
            # 실루엣 점수는 kmeans_model.labels가 존재하고, 고유 레이블이 2개 이상일 때만 계산 가능
            if kmeans_model.labels is not None and len(np.unique(kmeans_model.labels)) > 1:
                score = silhouette_score(X, kmeans_model.labels)
                silhouette_scores.append(score)
                print(f"  K={current_k}, Silhouette Score: {score:.4f}")
            else:
                silhouette_scores.append(np.nan) # 레이블 부족 또는 학습 실패 시
                print(f"  K={current_k}, Silhouette Score 계산 불가 (클러스터 수 < 2 또는 레이블 없음)")
            
            if kmeans_model.labels is not None and kmeans_model.centroids is not None:
                print(f"  K={current_k}, Visualizing the result of clustering...")
                kmeans_model.plot_clusters(X, title=f'K-Means Clustering (k={current_k}) on hw4_dataset')

        except Exception as e: # 예외 발생 시
            print(f"  K={current_k} 실행 중 오류 발생: {e}.")
            wcss_scores.append(np.nan) # 오류 시 WCSS 리스트에 NaN 추가
            silhouette_scores.append(np.nan) # 오류 시 실루엣 리스트에 NaN 추가

if __name__ == "__main__":
    main()