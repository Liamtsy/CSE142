# gmm_experiments.py

import pandas as pd
import numpy as np
from gmm import GMM # gmm.py 파일에서 GMM 클래스를 가져옵니다.
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score # 실루엣 점수 계산용

def run_gmm_k_experiments():
    # 1. 데이터셋 로드
    try:
        df = pd.read_csv('hw4_dataset.csv', header=0) # 헤더가 있는 CSV 기준
        X = df.to_numpy()
        print(f"데이터셋 로드 완료 (GMM k 실험용). 형태: {X.shape}")
    except FileNotFoundError:
        print("오류 (GMM k 실험용): hw4_dataset.csv 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"데이터셋 로드 중 오류 발생 (GMM k 실험용): {e}")
        return

    # 실험할 k 값의 범위 설정
    k_values_to_test = range(2, 9)  # 예: k=2부터 8까지 테스트 (GMM은 k=1도 가능하지만 보통 2부터)
    log_likelihoods = []
    silhouette_scores_gmm = []
    
    # GMM 파라미터 (필요시 조정)
    max_iters_gmm = 150 # GMM 수렴을 위해 반복 횟수를 늘려볼 수 있습니다.
    tol_gmm = 1e-4
    reg_covar_gmm = 1e-6

    print(f"\n--- GMM 컴포넌트 개수(k)에 따른 실험 시작 (max_iters={max_iters_gmm}) ---")
    for k_val in k_values_to_test:
        print(f"\n[ GMM K = {k_val} ] 실행 중...")
        
        # GMM 모델 생성 (init_params='random' 사용)
        gmm_model = GMM(k=k_val, max_iters=max_iters_gmm, tol=tol_gmm, reg_covar=reg_covar_gmm, init_params='random')
        
        try:
            gmm_model.fit(X) # 모델 학습
            
            # 최종 로그 가능도 저장
            if gmm_model.log_likelihood_history_: # 기록이 있는지 확인
                final_log_likelihood = gmm_model.log_likelihood_history_[-1]
                log_likelihoods.append(final_log_likelihood)
                print(f"  K={k_val}, 최종 Log-Likelihood: {final_log_likelihood:.4f}, 수렴: {gmm_model.converged_}")
            else:
                log_likelihoods.append(np.nan)
                print(f"  K={k_val}, Log-Likelihood 기록 없음.")

            # 실루엣 점수 계산 (GMM 예측 레이블 사용)
            if gmm_model.means_ is not None: # 모델이 최소한 초기화라도 되었는지 확인
                gmm_labels = gmm_model.predict(X)
                if len(np.unique(gmm_labels)) > 1: # 레이블이 최소 2종류 이상
                    score = silhouette_score(X, gmm_labels)
                    silhouette_scores_gmm.append(score)
                    print(f"  K={k_val}, Silhouette Score: {score:.4f}")
                else:
                    silhouette_scores_gmm.append(np.nan)
                    print(f"  K={k_val}, Silhouette Score 계산 불가 (클러스터 수 < 2)")
            else:
                silhouette_scores_gmm.append(np.nan)
                print(f"  K={k_val}, Silhouette Score 계산 불가 (모델 학습 실패)")
            
            # (선택 사항) 각 k에 대한 GMM 클러스터링 결과 시각화
            # 주석을 해제하면 각 k마다 플롯이 생성됩니다.
            # if gmm_model.means_ is not None:
            #     from hw4 import plot_clustering_results # hw4.py의 플로팅 함수 재활용 가정
            #     plot_clustering_results(X, gmm_labels, gmm_model.means_,
            #                             title=f'GMM Clustering (k={k_val}) - Experiment', is_gmm=True)
            #     plt.show()


        except Exception as e:
            print(f"  K={k_val} 실행 중 오류 발생: {e}")
            log_likelihoods.append(np.nan)
            silhouette_scores_gmm.append(np.nan)

    # --- 결과 플롯 (로그 가능도, 실루엣 점수) ---
    print("\n--- GMM k 실험 결과 플로팅 ---")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. 로그 가능도 vs. k 플롯
    valid_loglik_indices = ~np.isnan(log_likelihoods)
    if np.any(valid_loglik_indices):
        axes[0].plot(np.array(list(k_values_to_test))[valid_loglik_indices], np.array(log_likelihoods)[valid_loglik_indices], marker='o', linestyle='-')
    axes[0].set_title('Log-Likelihood vs. Number of Components (k)')
    axes[0].set_xlabel('Number of Components (k)')
    axes[0].set_ylabel('Log-Likelihood')
    axes[0].set_xticks(list(k_values_to_test))
    axes[0].grid(True, alpha=0.5)

    # 2. 실루엣 점수 vs. k 플롯
    valid_silhouette_indices = ~np.isnan(silhouette_scores_gmm)
    if np.any(valid_silhouette_indices):
        axes[1].plot(np.array(list(k_values_to_test))[valid_silhouette_indices], np.array(silhouette_scores_gmm)[valid_silhouette_indices], marker='o', linestyle='-')
    axes[1].set_title('Silhouette Score (GMM) vs. Number of Components (k)')
    axes[1].set_xlabel('Number of Components (k)')
    axes[1].set_ylabel('Average Silhouette Score')
    axes[1].set_xticks(list(k_values_to_test))
    axes[1].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()

    print("\nGMM k 값 실험 완료.")
    print("위 그래프와 (필요시) 개별 k에 대한 클러스터 플롯을 참고하여 GMM에 적합한 k값을 결정하세요.")

if __name__ == '__main__':
    run_gmm_k_experiments()