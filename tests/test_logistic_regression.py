import time

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from classification_models.linear_models.logistic_regression import \
    LogisticRegression as CustomLR

def compare_logistic_regressions():
    print("=== Logistic Regression Implementations ===\n")

    datasets = [
        ("Breast Cancer", load_breast_cancer(return_X_y=True)),
        ("Synthetic Balanced",
         make_classification(
             n_samples=1000, n_features=10, n_redundant=0,
             n_informative=8, n_clusters_per_class=1,
             weights=[0.5, 0.5], random_state=42
         )),
        ("Synthetic Imbalanced",
         make_classification(
             n_samples=1000, n_features=10, n_redundant=0,
             n_informative=8, n_clusters_per_class=1,
             weights=[0.9, 0.1], random_state=42
         ))
    ]

    for dataset_name, (X, y) in datasets:
        print(f"--- {dataset_name} Dataset ---")
        print(f"Shape: {X.shape}")
        print(
            f"Class distribution: {np.bincount(y)} (ratio: "
            f"{np.bincount(y)[1] / np.bincount(y)[0]:.2f})"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr_params = {
            'max_epochs': 1000,
            'tolerance': 1e-6,
            'verbose': False
        }

        print("\n🔧 Custom Logistic Regression:")
        start_time = time.time()

        custom_lr = CustomLR(**lr_params)
        custom_lr.fit(X_train_scaled, y_train)

        custom_train_time = time.time() - start_time

        start_time = time.time()
        custom_pred_proba = custom_lr.predict_proba(X_test_scaled)
        custom_pred = custom_lr.predict(X_test_scaled)
        custom_pred_time = time.time() - start_time

        custom_accuracy = accuracy_score(y_test, custom_pred)
        custom_precision = precision_score(y_test, custom_pred, zero_division=0)
        custom_recall = recall_score(y_test, custom_pred, zero_division=0)
        custom_f1 = f1_score(y_test, custom_pred, zero_division=0)
        custom_auc = roc_auc_score(y_test, custom_pred_proba)
        custom_logloss = log_loss(y_test, custom_pred_proba)

        print(f"   Training time: {custom_train_time:.3f}s")
        print(f"   Prediction time: {custom_pred_time:.4f}s")
        print(f"   Accuracy: {custom_accuracy:.4f}")
        print(f"   Precision: {custom_precision:.4f}")
        print(f"   Recall: {custom_recall:.4f}")
        print(f"   F1 Score: {custom_f1:.4f}")
        print(f"   ROC AUC: {custom_auc:.4f}")
        print(f"   Log Loss: {custom_logloss:.4f}")
        print(f"   Training epochs: {len(custom_lr.training_history)}")

        print("\n📚 Sklearn Logistic Regression:")
        start_time = time.time()

        sklearn_lr = SklearnLR(max_iter=1000, random_state=42, solver='lbfgs')
        sklearn_lr.fit(X_train_scaled, y_train)

        sklearn_train_time = time.time() - start_time

        start_time = time.time()
        sklearn_pred_proba = sklearn_lr.predict_proba(X_test_scaled)[:, 1]
        sklearn_pred = sklearn_lr.predict(X_test_scaled)
        sklearn_pred_time = time.time() - start_time

        sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
        sklearn_precision = precision_score(y_test, sklearn_pred, zero_division=0)
        sklearn_recall = recall_score(y_test, sklearn_pred, zero_division=0)
        sklearn_f1 = f1_score(y_test, sklearn_pred, zero_division=0)
        sklearn_auc = roc_auc_score(y_test, sklearn_pred_proba)
        sklearn_logloss = log_loss(y_test, sklearn_pred_proba)

        print(f"   Training time: {sklearn_train_time:.3f}s")
        print(f"   Prediction time: {sklearn_pred_time:.4f}s")
        print(f"   Accuracy: {sklearn_accuracy:.4f}")
        print(f"   Precision: {sklearn_precision:.4f}")
        print(f"   Recall: {sklearn_recall:.4f}")
        print(f"   F1 Score: {sklearn_f1:.4f}")
        print(f"   ROC AUC: {sklearn_auc:.4f}")
        print(f"   Log Loss: {sklearn_logloss:.4f}")
        print(f"   Training iterations: {sklearn_lr.n_iter_[0]}")

        print("\n📊 Comparison:")
        acc_diff = abs(custom_accuracy - sklearn_accuracy) / sklearn_accuracy * 100
        auc_diff = abs(custom_auc - sklearn_auc) / sklearn_auc * 100
        logloss_diff = abs(custom_logloss - sklearn_logloss) / sklearn_logloss * 100
        speed_ratio = sklearn_train_time / custom_train_time

        print(f"   Accuracy difference: {acc_diff:.1f}%")
        print(f"   AUC difference: {auc_diff:.1f}%")
        print(f"   Log Loss difference: {logloss_diff:.1f}%")
        print(f"   Speed ratio (sklearn/custom): {speed_ratio:.2f}x")

        # Сравнение коэффициентов
        if hasattr(custom_lr, 'get_training_metadata'):
            custom_metadata = custom_lr.get_training_metadata()
            custom_coeffs = custom_metadata['coefficients']
            custom_intercept = custom_metadata['intercept']

            sklearn_coeffs = sklearn_lr.coef_[0]
            sklearn_intercept = sklearn_lr.intercept_[0]

            print(f"\n🎯 Model Parameters:")
            print(
                f"   Intercept - Custom: {custom_intercept:.4f}, Sklearn: "
                f"{sklearn_intercept:.4f}"
            )

            coeff_correlation = np.corrcoef(custom_coeffs, sklearn_coeffs)[0, 1]
            print(f"   Coefficients correlation: {coeff_correlation:.4f}")

            # Топ-5 самых важных признаков
            custom_abs_coeffs = np.abs(custom_coeffs)
            sklearn_abs_coeffs = np.abs(sklearn_coeffs)

            custom_top5 = np.argsort(custom_abs_coeffs)[-5:][::-1]
            sklearn_top5 = np.argsort(sklearn_abs_coeffs)[-5:][::-1]

            print("   Top 5 Features by |coefficient|:")
            print("   Custom LR  | Sklearn LR")
            print("   Feat  Coef | Feat  Coef")
            for i in range(5):
                if i < len(custom_top5) and i < len(sklearn_top5):
                    custom_feat = custom_top5[i]
                    sklearn_feat = sklearn_top5[i]
                    custom_coef = custom_coeffs[custom_feat]
                    sklearn_coef = sklearn_coeffs[sklearn_feat]
                    print(
                        f"   {custom_feat:2d}  {custom_coef:+.3f} | {sklearn_feat:2d} "
                        f" {sklearn_coef:+.3f}"
                    )

        print("\n" + "=" * 60 + "\n")


def test_edge_cases():
    print("=== Edge Cases Testing ===\n")

    print("🔬 Small dataset (50 samples):")
    X_small, y_small = make_classification(
        n_samples=50, n_features=5, n_redundant=0, n_informative=4,
        n_clusters_per_class=1, random_state=42
    )

    custom_lr_small = CustomLR(max_epochs=500, verbose=False)
    sklearn_lr_small = SklearnLR(max_iter=500, random_state=42)

    custom_lr_small.fit(X_small, y_small)
    sklearn_lr_small.fit(X_small, y_small)

    custom_pred_small = custom_lr_small.predict(X_small)
    sklearn_pred_small = sklearn_lr_small.predict(X_small)

    print(f"   Custom Accuracy: {accuracy_score(y_small, custom_pred_small):.4f}")
    print(f"   Sklearn Accuracy: {accuracy_score(y_small, sklearn_pred_small):.4f}")

    print("\n📏 Single feature:")
    X_1d = X_small[:, [0]]

    custom_lr_1d = CustomLR(max_epochs=500, verbose=False)
    sklearn_lr_1d = SklearnLR(max_iter=500, random_state=42)

    custom_lr_1d.fit(X_1d, y_small)
    sklearn_lr_1d.fit(X_1d, y_small)

    custom_pred_1d = custom_lr_1d.predict(X_1d)
    sklearn_pred_1d = sklearn_lr_1d.predict(X_1d)

    print(f"   Custom Accuracy: {accuracy_score(y_small, custom_pred_1d):.4f}")
    print(f"   Sklearn Accuracy: {accuracy_score(y_small, sklearn_pred_1d):.4f}")

    print("\n🎯 Perfectly separable data:")
    X_perfect = np.array([[1, 1], [2, 2], [3, 3], [1, 2], [2, 1], [3, 1]])
    y_perfect = np.array([0, 0, 0, 1, 1, 1])

    custom_lr_perfect = CustomLR(max_epochs=1000, verbose=False)
    sklearn_lr_perfect = SklearnLR(max_iter=1000, random_state=42)

    try:
        custom_lr_perfect.fit(X_perfect, y_perfect)
        custom_pred_perfect = custom_lr_perfect.predict(X_perfect)
        custom_acc_perfect = accuracy_score(y_perfect, custom_pred_perfect)
        print(f"   Custom Accuracy: {custom_acc_perfect:.4f}")
    except Exception as e:
        print(f"   Custom LR failed: {e}")

    try:
        sklearn_lr_perfect.fit(X_perfect, y_perfect)
        sklearn_pred_perfect = sklearn_lr_perfect.predict(X_perfect)
        sklearn_acc_perfect = accuracy_score(y_perfect, sklearn_pred_perfect)
        print(f"   Sklearn Accuracy: {sklearn_acc_perfect:.4f}")
    except Exception as e:
        print(f"   Sklearn LR failed: {e}")

    print("\n⚖️ Highly imbalanced data (1:99 ratio):")
    X_imb, y_imb = make_classification(
        n_samples=1000, n_features=10, weights=[0.99, 0.01],
        n_clusters_per_class=1, random_state=42
    )

    X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
        X_imb, y_imb, test_size=0.2, random_state=42, stratify=y_imb
    )

    scaler_imb = StandardScaler()
    X_train_imb_scaled = scaler_imb.fit_transform(X_train_imb)
    X_test_imb_scaled = scaler_imb.transform(X_test_imb)

    custom_lr_imb = CustomLR(max_epochs=1000, verbose=False)
    sklearn_lr_imb = SklearnLR(max_iter=1000, random_state=42)

    custom_lr_imb.fit(X_train_imb_scaled, y_train_imb)
    sklearn_lr_imb.fit(X_train_imb_scaled, y_train_imb)

    custom_pred_imb = custom_lr_imb.predict(X_test_imb_scaled)
    sklearn_pred_imb = sklearn_lr_imb.predict(X_test_imb_scaled)

    custom_f1_imb = f1_score(y_test_imb, custom_pred_imb, zero_division=0)
    sklearn_f1_imb = f1_score(y_test_imb, sklearn_pred_imb, zero_division=0)

    print(f"   Test set class distribution: {np.bincount(y_test_imb)}")
    print(f"   Custom F1 Score: {custom_f1_imb:.4f}")
    print(f"   Sklearn F1 Score: {sklearn_f1_imb:.4f}")


def test_different_thresholds():
    print("\n=== Different Decision Thresholds Test ===")

    X, y = make_classification(
        n_samples=500, n_features=10, n_redundant=0, n_informative=8,
        n_clusters_per_class=1, weights=[0.7, 0.3], random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    custom_lr = CustomLR(max_epochs=1000, verbose=False)
    custom_lr.fit(X_train_scaled, y_train)

    thresholds = [0.3, 0.5, 0.7]

    print("Threshold | Precision | Recall | F1 Score")
    print("-" * 40)

    for threshold in thresholds:
        pred = custom_lr.predict(X_test_scaled, threshold=threshold)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)

        print(
            f"   {threshold:.1f}   |   {precision:.3f}   |  {recall:.3f}  |  {f1:.3f}"
        )


def test_convergence():
    print("\n=== Convergence Analysis ===")

    X, y = make_classification(n_samples=300, n_features=5, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    custom_lr = CustomLR(max_epochs=2000, tolerance=1e-8, verbose=False)
    custom_lr.fit(X_scaled, y)

    history = custom_lr.training_history

    print(f"Total epochs: {len(history)}")
    print(f"Initial loss: {history[0]:.6f}")
    print(f"Final loss: {history[-1]:.6f}")
    print(f"Loss reduction: {((history[0] - history[-1]) / history[0] * 100):.1f}%")

    # Проверка монотонности убывания
    non_decreasing_count = sum(
        1 for i in range(1, len(history))
        if history[i] > history[i - 1]
    )
    print(f"Non-decreasing steps: {non_decreasing_count}/{len(history) - 1}")

    if len(history) >= 100:
        print(f"Loss at epoch 100: {history[99]:.6f}")
        if len(history) >= 500:
            print(f"Loss at epoch 500: {history[499]:.6f}")


if __name__ == '__main__':
    np.random.seed(42)
    print("🚀 Starting Logistic Regression Comparison...")
    print("Note: Make sure to import your CustomLR implementation\n")

    try:
        compare_logistic_regressions()
        test_edge_cases()
        test_different_thresholds()
        test_convergence()

        print("✅ All tests completed successfully!")

    except NameError as e:
        print(f"❌ Please import your CustomLR implementation first! Error: {e}")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback


        traceback.print_exc()
