import time

import numpy as np
from sklearn.datasets import load_diabetes, make_regression
from sklearn.ensemble import RandomForestRegressor as SklearnRF
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from regression_models.ensembles.random_forest import RandomForestRegressor as CustomRF


def compare_random_forests():
    print("=== Random Forest Implementations ===\n")

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

        rf_params = {
            'n_estimators': 50,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }

        print("\n🔧 Custom Random Forest:")
        start_time = time.time()

        custom_rf = CustomRF(**rf_params, verbose=True)
        custom_rf.fit(X_train_scaled, y_train)

        custom_train_time = time.time() - start_time

        start_time = time.time()
        custom_pred = custom_rf.predict(X_test_scaled)
        custom_pred_time = time.time() - start_time

        custom_mse = mean_squared_error(y_test, custom_pred)
        custom_mae = mean_absolute_error(y_test, custom_pred)
        custom_r2 = r2_score(y_test, custom_pred)

        print(f"   Training time: {custom_train_time:.3f}s")
        print(f"   Prediction time: {custom_pred_time:.4f}s")
        print(f"   MSE: {custom_mse:.4f}")
        print(f"   MAE: {custom_mae:.4f}")
        print(f"   R² Score: {custom_r2:.4f}")

        if hasattr(custom_rf, 'oob_score_') and custom_rf.oob_score_ is not None:
            print(f"   OOB Score: {custom_rf.oob_score_:.4f}")

        print("\n📚 Sklearn Random Forest:")
        start_time = time.time()

        sklearn_rf = SklearnRF(**rf_params)
        sklearn_rf.fit(X_train_scaled, y_train)

        sklearn_train_time = time.time() - start_time

        start_time = time.time()
        sklearn_pred = sklearn_rf.predict(X_test_scaled)
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

        if hasattr(custom_rf, 'feature_importances_'):
            print(f"\n🎯 Feature Importances (Top 5):")
            custom_importances = custom_rf.feature_importances_
            sklearn_importances = sklearn_rf.feature_importances_

            custom_top5 = np.argsort(custom_importances)[-5:][::-1]
            sklearn_top5 = np.argsort(sklearn_importances)[-5:][::-1]

            print("   Custom RF  | Sklearn RF")
            print("   Feature Imp| Feature Imp")
            for i in range(5):
                if i < len(custom_top5) and i < len(sklearn_top5):
                    custom_feat = custom_top5[i]
                    sklearn_feat = sklearn_top5[i]
                    custom_imp = custom_importances[custom_feat]
                    sklearn_imp = sklearn_importances[sklearn_feat]
                    print(
                        f"   {custom_feat:2d}  {custom_imp:.3f} | {sklearn_feat:2d}  "
                        f"{sklearn_imp:.3f}"
                    )

        print("\n" + "=" * 60 + "\n")


def test_edge_cases():
    print("=== Edge Cases Testing ===\n")

    print("🔬 Small dataset (50 samples):")
    X_small, y_small = make_regression(
        n_samples=50,
        n_features=5,
        noise=0.1,
        random_state=42
    )

    custom_rf_small = CustomRF(n_estimators=10, random_state=42)
    sklearn_rf_small = SklearnRF(n_estimators=10, random_state=42)

    custom_rf_small.fit(X_small, y_small)
    sklearn_rf_small.fit(X_small, y_small)

    custom_pred_small = custom_rf_small.predict(X_small)
    sklearn_pred_small = sklearn_rf_small.predict(X_small)

    print(f"   Custom R²: {r2_score(y_small, custom_pred_small):.4f}")
    print(f"   Sklearn R²: {r2_score(y_small, sklearn_pred_small):.4f}")

    print("\n📏 Single feature:")
    X_1d = X_small[:, [0]]

    custom_rf_1d = CustomRF(n_estimators=10, random_state=42)
    sklearn_rf_1d = SklearnRF(n_estimators=10, random_state=42)

    custom_rf_1d.fit(X_1d, y_small)
    sklearn_rf_1d.fit(X_1d, y_small)

    custom_pred_1d = custom_rf_1d.predict(X_1d)
    sklearn_pred_1d = sklearn_rf_1d.predict(X_1d)

    print(f"   Custom R²: {r2_score(y_small, custom_pred_1d):.4f}")
    print(f"   Sklearn R²: {r2_score(y_small, sklearn_pred_1d):.4f}")

    print("\n🎯 Overfitting test (high depth):")
    custom_rf_deep = CustomRF(n_estimators=20, max_depth=20, random_state=42)
    sklearn_rf_deep = SklearnRF(n_estimators=20, max_depth=20, random_state=42)

    custom_rf_deep.fit(X_small, y_small)
    sklearn_rf_deep.fit(X_small, y_small)

    X_train, X_test, y_train, y_test = train_test_split(
        X_small,
        y_small,
        test_size=0.3,
        random_state=42
    )

    custom_rf_deep.fit(X_train, y_train)
    sklearn_rf_deep.fit(X_train, y_train)

    custom_train_r2 = r2_score(y_train, custom_rf_deep.predict(X_train))
    custom_test_r2 = r2_score(y_test, custom_rf_deep.predict(X_test))
    sklearn_train_r2 = r2_score(y_train, sklearn_rf_deep.predict(X_train))
    sklearn_test_r2 = r2_score(y_test, sklearn_rf_deep.predict(X_test))

    print(f"   Custom - Train R²: {custom_train_r2:.4f}, Test R²: {custom_test_r2:.4f}")
    print(
        f"   Sklearn - Train R²: {sklearn_train_r2:.4f}, Test R²: {sklearn_test_r2:.4f}"
    )


def test_prediction_intervals():
    print("\n=== Prediction Intervals Test ===")

    X, y = make_regression(n_samples=200, n_features=5, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    custom_rf = CustomRF(n_estimators=50, random_state=42)
    custom_rf.fit(X_train, y_train)

    if hasattr(custom_rf, 'predict_std'):
        pred_mean = custom_rf.predict(X_test)
        pred_std = custom_rf.predict_std(X_test)

        print(f"Mean prediction std: {np.mean(pred_std):.4f}")
        print(f"Max prediction std: {np.max(pred_std):.4f}")
        print(f"Min prediction std: {np.min(pred_std):.4f}")

        lower_bound = pred_mean - 2 * pred_std
        upper_bound = pred_mean + 2 * pred_std

        coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
        print(f"95% confidence interval coverage: {coverage:.1%}")


if __name__ == '__main__':

    print("🚀 Starting Random Forest Comparison...")
    print(
        "Note: Uncomment the import line and replace CustomRF with your "
        "implementation\n"
    )

    try:
        compare_random_forests()
        test_edge_cases()
        test_prediction_intervals()

        print("✅ All tests completed successfully!")

    except NameError:
        print("❌ Please import your CustomRF implementation first!")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback


        traceback.print_exc()
