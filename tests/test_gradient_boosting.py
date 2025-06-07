import time

import numpy as np
from sklearn.datasets import load_diabetes, make_regression
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBDT
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from regression_models.ensembles.gradient_boosting import \
    GradientBoostingRegressor as CustomGBDT, MSELossGBDT, MAELossGBDT, HuberLossGBDT


def compare_gradient_boosting():
    print("=== Gradient Boosting Implementations ===\n")

    datasets = [
        ("Diabetes", load_diabetes(return_X_y=True)),
        ("Synthetic",
         make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42))
    ]

    for dataset_name, (X, y) in datasets:
        print(f"--- {dataset_name} Dataset ---")
        print(f"Shape: {X.shape}, Target range: [{y.min():.2f}, {y.max():.2f}]")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        gbdt_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 4,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'subsample': 0.8,
            'random_state': 42
        }

        print("\n🔧 Custom Gradient Boosting:")
        start_time = time.time()

        custom_gbdt = CustomGBDT(**gbdt_params, verbose=True)
        custom_gbdt.fit(X_train_scaled, y_train)

        custom_train_time = time.time() - start_time

        start_time = time.time()
        custom_pred = custom_gbdt.predict(X_test_scaled)
        custom_pred_time = time.time() - start_time

        custom_mse = mean_squared_error(y_test, custom_pred)
        custom_mae = mean_absolute_error(y_test, custom_pred)
        custom_r2 = r2_score(y_test, custom_pred)

        print(f"   Training time: {custom_train_time:.3f}s")
        print(f"   Prediction time: {custom_pred_time:.4f}s")
        print(f"   MSE: {custom_mse:.4f}")
        print(f"   MAE: {custom_mae:.4f}")
        print(f"   R² Score: {custom_r2:.4f}")
        print(f"   Number of trees used: {custom_gbdt._n_iter}")

        print("\n📚 Sklearn Gradient Boosting:")
        start_time = time.time()

        sklearn_gbdt = SklearnGBDT(**gbdt_params)
        sklearn_gbdt.fit(X_train_scaled, y_train)

        sklearn_train_time = time.time() - start_time

        start_time = time.time()
        sklearn_pred = sklearn_gbdt.predict(X_test_scaled)
        sklearn_pred_time = time.time() - start_time

        sklearn_mse = mean_squared_error(y_test, sklearn_pred)
        sklearn_mae = mean_absolute_error(y_test, sklearn_pred)
        sklearn_r2 = r2_score(y_test, sklearn_pred)

        print(f"   Training time: {sklearn_train_time:.3f}s")
        print(f"   Prediction time: {sklearn_pred_time:.4f}s")
        print(f"   MSE: {sklearn_mse:.4f}")
        print(f"   MAE: {sklearn_mae:.4f}")
        print(f"   R² Score: {sklearn_r2:.4f}")

        print("\n📊 Comparison:")
        mse_diff = abs(custom_mse - sklearn_mse) / sklearn_mse * 100
        r2_diff = abs(custom_r2 - sklearn_r2) / abs(
            sklearn_r2
        ) * 100 if sklearn_r2 != 0 else float('inf')
        speed_ratio = sklearn_train_time / custom_train_time

        print(f"   MSE difference: {mse_diff:.1f}%")
        print(f"   R² difference: {r2_diff:.1f}%")
        print(f"   Speed ratio (sklearn/custom): {speed_ratio:.2f}x")

        if hasattr(custom_gbdt, 'feature_importances_'):
            print(f"\n🎯 Feature Importances (Top 5):")
            custom_importances = custom_gbdt.feature_importances_
            sklearn_importances = sklearn_gbdt.feature_importances_

            custom_top5 = np.argsort(custom_importances)[-5:][::-1]
            sklearn_top5 = np.argsort(sklearn_importances)[-5:][::-1]

            print("   Custom GBDT | Sklearn GBDT")
            print("   Feature Imp | Feature Imp")
            for i in range(5):
                if i < len(custom_top5) and i < len(sklearn_top5):
                    custom_feat = custom_top5[i]
                    sklearn_feat = sklearn_top5[i]
                    custom_imp = custom_importances[custom_feat]
                    sklearn_imp = sklearn_importances[sklearn_feat]
                    print(
                        f"   {custom_feat:2d}  {custom_imp:.3f}  | {sklearn_feat:2d}  "
                        f"{sklearn_imp:.3f}"
                    )

        print("\n" + "=" * 60 + "\n")
        plot_learning_curves(custom_gbdt)
        plot_learning_curves(sklearn_gbdt)


def test_loss_functions():
    print("=== Loss Functions Test ===\n")

    X, y = make_regression(n_samples=500, n_features=8, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    loss_functions = [MSELossGBDT, MAELossGBDT, HuberLossGBDT]

    print("Testing different loss functions:")
    for loss in loss_functions:
        print(f"\n🔧 Loss function: {loss.__class__}\n")

        custom_gbdt = CustomGBDT(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4,
            loss=loss,
            random_state=42
        )

        start_time = time.time()
        custom_gbdt.fit(X_train, y_train)
        train_time = time.time() - start_time

        pred = custom_gbdt.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        print(f"   Training time: {train_time:.3f}s")
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R² Score: {r2:.4f}")


def test_early_stopping():
    print("\n=== Early Stopping Test ===\n")

    X, y = make_regression(n_samples=800, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🛑 Testing early stopping:")

    # Without early stopping
    gbdt_no_early = CustomGBDT(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        validation_fraction=0.0,  # Disable validation
        random_state=42,
        verbose=False
    )

    start_time = time.time()
    gbdt_no_early.fit(X_train, y_train)
    time_no_early = time.time() - start_time

    # With early stopping
    gbdt_early = CustomGBDT(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        validation_fraction=0.2,
        n_iter_no_change=10,
        tolerance=1e-4,
        random_state=42,
        verbose=False
    )

    start_time = time.time()
    gbdt_early.fit(X_train, y_train)
    time_early = time.time() - start_time

    pred_no_early = gbdt_no_early.predict(X_test)
    pred_early = gbdt_early.predict(X_test)

    print(f"   Without early stopping:")
    print(f"     Training time: {time_no_early:.3f}s")
    print(f"     Trees used: {gbdt_no_early._n_iter}")
    print(f"     Test R²: {r2_score(y_test, pred_no_early):.4f}")

    print(f"   With early stopping:")
    print(f"     Training time: {time_early:.3f}s")
    print(f"     Trees used: {gbdt_early._n_iter}")
    print(f"     Test R²: {r2_score(y_test, pred_early):.4f}")

    print(f"   Time saved: {((time_no_early - time_early) / time_no_early * 100):.1f}%")


def test_learning_curves():
    print("\n=== Learning Curves Test ===\n")

    X, y = make_regression(n_samples=600, n_features=8, noise=0.15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("📈 Generating learning curves:")

    gbdt = CustomGBDT(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        validation_fraction=0.2,
        random_state=42,
        verbose=False
    )

    gbdt.fit(X_train, y_train)

    print(f"   Final training score: {gbdt.train_score_[-1]:.6f}")
    if gbdt.validation_score_:
        print(f"   Final validation score: {gbdt.validation_score_[-1]:.6f}")

    # Test staged predictions
    if hasattr(gbdt, 'staged_predict'):
        print("\n🎭 Testing staged predictions:")
        staged_preds = list(gbdt.staged_predict(X_test[:5]))  # First 5 samples
        print(
            f"   Staged predictions shape: {len(staged_preds)} x {len(staged_preds[0])}"
        )

        # Calculate R² for each stage
        r2_scores = []
        for pred in staged_preds:
            full_pred = list(gbdt.staged_predict(X_test))[len(r2_scores)]
            r2_scores.append(r2_score(y_test, full_pred))

        print(f"   R² progression (first 10 stages): {r2_scores[:10]}")
        print(f"   Final staged R²: {r2_scores[-1]:.4f}")


def test_edge_cases():
    print("\n=== Edge Cases Testing ===\n")

    print("🔬 Small dataset (50 samples):")
    X_small, y_small = make_regression(
        n_samples=50,
        n_features=5,
        noise=0.1,
        random_state=42
    )

    custom_gbdt_small = CustomGBDT(n_estimators=20, learning_rate=0.2, random_state=42)
    sklearn_gbdt_small = SklearnGBDT(
        n_estimators=20,
        learning_rate=0.2,
        random_state=42
    )

    custom_gbdt_small.fit(X_small, y_small)
    sklearn_gbdt_small.fit(X_small, y_small)

    custom_pred_small = custom_gbdt_small.predict(X_small)
    sklearn_pred_small = sklearn_gbdt_small.predict(X_small)

    print(f"   Custom R²: {r2_score(y_small, custom_pred_small):.4f}")
    print(f"   Sklearn R²: {r2_score(y_small, sklearn_pred_small):.4f}")

    print("\n📏 Single feature:")
    X_1d = X_small[:, [0]]

    custom_gbdt_1d = CustomGBDT(n_estimators=20, learning_rate=0.2, random_state=42)
    sklearn_gbdt_1d = SklearnGBDT(n_estimators=20, learning_rate=0.2, random_state=42)

    custom_gbdt_1d.fit(X_1d, y_small)
    sklearn_gbdt_1d.fit(X_1d, y_small)

    custom_pred_1d = custom_gbdt_1d.predict(X_1d)
    sklearn_pred_1d = sklearn_gbdt_1d.predict(X_1d)

    print(f"   Custom R²: {r2_score(y_small, custom_pred_1d):.4f}")
    print(f"   Sklearn R²: {r2_score(y_small, sklearn_pred_1d):.4f}")

    print("\n🎯 Overfitting test (high learning rate):")
    X_train, X_test, y_train, y_test = train_test_split(
        X_small,
        y_small,
        test_size=0.3,
        random_state=42
    )

    custom_gbdt_overfit = CustomGBDT(
        n_estimators=50,
        learning_rate=0.5,
        max_depth=6,
        random_state=42
    )
    sklearn_gbdt_overfit = SklearnGBDT(
        n_estimators=50,
        learning_rate=0.5,
        max_depth=6,
        random_state=42
    )

    custom_gbdt_overfit.fit(X_train, y_train)
    sklearn_gbdt_overfit.fit(X_train, y_train)

    custom_train_r2 = r2_score(y_train, custom_gbdt_overfit.predict(X_train))
    custom_test_r2 = r2_score(y_test, custom_gbdt_overfit.predict(X_test))
    sklearn_train_r2 = r2_score(y_train, sklearn_gbdt_overfit.predict(X_train))
    sklearn_test_r2 = r2_score(y_test, sklearn_gbdt_overfit.predict(X_test))

    print(f"   Custom - Train R²: {custom_train_r2:.4f}, Test R²: {custom_test_r2:.4f}")
    print(
        f"   Sklearn - Train R²: {sklearn_train_r2:.4f}, Test R²: {sklearn_test_r2:.4f}"
    )

    custom_overfit = custom_train_r2 - custom_test_r2
    sklearn_overfit = sklearn_train_r2 - sklearn_test_r2
    print(f"   Custom overfitting gap: {custom_overfit:.4f}")
    print(f"   Sklearn overfitting gap: {sklearn_overfit:.4f}")


def test_hyperparameters():
    print("\n=== Hyperparameters Test ===\n")

    X, y = make_regression(n_samples=400, n_features=8, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print("🔧 Testing different hyperparameters:")

    # Learning rate test
    learning_rates = [0.01, 0.1, 0.3]
    print("\n   Learning Rate Impact:")
    for lr in learning_rates:
        gbdt = CustomGBDT(
            n_estimators=50,
            learning_rate=lr,
            max_depth=4,
            random_state=42
        )
        gbdt.fit(X_train, y_train)
        pred = gbdt.predict(X_test)
        r2 = r2_score(y_test, pred)
        print(f"     LR={lr:.2f}: R²={r2:.4f}")

    # Max depth test
    max_depths = [2, 4, 8]
    print("\n   Max Depth Impact:")
    for depth in max_depths:
        gbdt = CustomGBDT(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=depth,
            random_state=42
        )
        gbdt.fit(X_train, y_train)
        pred = gbdt.predict(X_test)
        r2 = r2_score(y_test, pred)
        print(f"     Depth={depth}: R²={r2:.4f}")

    # Subsample test
    subsamples = [0.5, 0.8, 1.0]
    print("\n   Subsample Impact:")
    for sub in subsamples:
        gbdt = CustomGBDT(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4,
            subsample=sub,
            random_state=42
        )
        gbdt.fit(X_train, y_train)
        pred = gbdt.predict(X_test)
        r2 = r2_score(y_test, pred)
        print(f"     Subsample={sub:.1f}: R²={r2:.4f}")


def plot_learning_curves(gbdt, title="Learning Curves"):
    """Plot training and validation curves if available"""
    try:
        import matplotlib.pyplot as plt

        if hasattr(gbdt, 'train_score_') and gbdt.train_score_:
            plt.figure(figsize=(10, 6))

            iterations = range(1, len(gbdt.train_score_) + 1)
            plt.plot(
                iterations,
                gbdt.train_score_,
                'b-',
                label='Training Loss',
                alpha=0.8
            )

            if hasattr(gbdt, 'validation_score_') and gbdt.validation_score_:
                plt.plot(
                    iterations,
                    gbdt.validation_score_,
                    'r-',
                    label='Validation Loss',
                    alpha=0.8
                )

            plt.xlabel('Boosting Iterations')
            plt.ylabel('Loss')
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

    except ImportError:
        print("   (Matplotlib not available for plotting)")


if __name__ == '__main__':
    print("🚀 Starting Gradient Boosting Comparison...")
    print(
        "Note: Uncomment the import line and replace CustomGBDT with your "
        "implementation\n"
    )

    try:
        compare_gradient_boosting()
        test_loss_functions()
        test_early_stopping()
        test_learning_curves()
        test_edge_cases()
        test_hyperparameters()

        print("\n✅ All tests completed successfully!")

    except NameError:
        print("❌ Please import your CustomGBDT implementation first!")
        print("Uncomment and adjust the import line at the top of the file:")
        print("# from your_module import GradientBoostingRegressor as CustomGBDT")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback


        traceback.print_exc()
