import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import ssl

DATA_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv"
CRITICAL_THRESHOLD = 95.0  # Temperature above this is considered an 'incident'
TARGET_RECALL = 0.80       # Business requirement: Catch 80% of incidents

def setup_ssl():
    """Handle SSL certificate verification issues common on macOS."""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

def load_data(url):
    """Load dataset and perform initial preprocessing."""
    df = pd.read_csv(url)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['is_incident'] = (df['value'] >= CRITICAL_THRESHOLD).astype(int)
    return df

def prepare_features(df, window_size, horizon):
    """Apply sliding window formulation for features and target."""
    df_copy = df.copy()
    
    # Target: Will there be an incident in the NEXT H steps?
    df_copy['target_H'] = df_copy['is_incident'].rolling(window=horizon, min_periods=1).max().shift(-horizon)
    
    # Feature Engineering
    df_copy['feature_mean_W'] = df_copy['value'].rolling(window=window_size).mean()
    df_copy['feature_std_W'] = df_copy['value'].rolling(window=window_size).std()
    df_copy['feature_max_W'] = df_copy['value'].rolling(window=window_size).max()
    df_copy['feature_min_W'] = df_copy['value'].rolling(window=window_size).min()
    df_copy['feature_current'] = df_copy['value']
    
    return df_copy.dropna().reset_index(drop=True)

def evaluate_walk_forward(df_ready, n_estimators, max_depth, n_splits=5):
    """
    Perform Walk-Forward Validation using TimeSeriesSplit to ensure 
    temporal robustness across different time periods.
    """
    features = ['feature_mean_W', 'feature_std_W', 'feature_max_W', 'feature_min_W', 'feature_current']
    X = df_ready[features]
    y = df_ready['target_H']
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    all_precisions = []
    all_recalls = []
    all_thresholds = []
    
    # Variables to hold the final fold's data for lead-time and plotting
    last_model = None
    last_X_test = None
    last_y_test = None
    last_test_df = None
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        test_df = df_ready.iloc[test_index]
        
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42, 
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_probs = model.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
        
        # Find metrics closest to the target recall for this fold
        idx = (np.abs(recalls - TARGET_RECALL)).argmin()
        all_precisions.append(precisions[idx])
        all_recalls.append(recalls[idx])
        # thresholds array has length len(recalls) - 1, handle indexing safely
        thresh_idx = min(idx, len(thresholds) - 1)
        all_thresholds.append(thresholds[thresh_idx])
        
        # Save the latest fold's data
        last_model = model
        last_X_test = X_test
        last_y_test = y_test
        last_test_df = test_df

    return (
        np.mean(all_precisions), 
        np.mean(all_recalls), 
        np.mean(all_thresholds),
        last_model, last_X_test, last_y_test, last_test_df
    )

def objective(trial, df_raw):
    """Optuna objective function to maximize precision at 80% recall."""
    w = trial.suggest_int('W', 5, 48)   # 25 min to 4 hours
    h = trial.suggest_int('H', 2, 12)   # 10 min to 1 hour
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    
    df_ready = prepare_features(df_raw, w, h)
    
    # Optimize based on the mean precision across all temporal folds
    precision, _, _, _, _, _, _ = evaluate_walk_forward(df_ready, n_estimators, max_depth)
    
    return precision

def calculate_final_metrics(model, X_test, y_test, test_df, optimal_threshold, horizon):
    """Calculate False Positive Rate and Improved Detection Lead Time."""
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= optimal_threshold).astype(int)

    # 1. False Positive Rate
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # 2. Detection Lead Time
    lead_times = []
    is_incident_array = test_df['is_incident'].values

    for i in range(len(y_pred)):
        # If the model raises an alert
        if y_pred[i] == 1: 
            # Look ahead up to H steps to find when the actual incident starts
            look_ahead = is_incident_array[i + 1 : i + horizon + 1]
            incident_indices = np.where(look_ahead == 1)[0]
            
            # If there is an actual incident in the predicted horizon
            if len(incident_indices) > 0:
                # Capture the exact distance to the FIRST incident step
                lead_steps = incident_indices[0] + 1
                lead_times.append(lead_steps)

    avg_lead_steps = np.mean(lead_times) if lead_times else 0
    avg_lead_minutes = avg_lead_steps * 5  # NAB data is recorded in 5-minute intervals

    return fpr, avg_lead_minutes

def save_final_plots(model, X_test, y_test, target_recall, optimal_threshold):
    """Generate and save the final evaluation plots."""
    y_probs = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    # PR Curve
    plt.figure(figsize=(10, 6))
    plt.plot(recalls[:-1], precisions[:-1], label='PR Curve', color='blue')
    plt.axvline(x=target_recall, color='red', linestyle='--', label=f'Target Recall ({target_recall:.0%})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Optimized Precision-Recall Trade-off (Final Fold)')
    plt.legend()
    plt.grid(True)
    plt.savefig('pr_curve.png')
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    y_pred = (y_probs >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Incident', 'Incident'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Optimized Confusion Matrix (Threshold={optimal_threshold:.3f})')
    plt.savefig('confusion_matrix.png')

def main():
    setup_ssl()
    df_raw = load_data(DATA_URL)
    
    print("Starting Bayesian Optimization with Walk-Forward Validation (100 Trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, df_raw), n_trials=100, 
                    gc_after_trial=True, n_jobs=-1, show_progress_bar=True)
    
    best = study.best_params
    print("\n--- Optimization Complete ---")
    
    # Extract final metrics using the best parameters
    df_final = prepare_features(df_raw, best['W'], best['H'])
    precision, recall, threshold, model, X_test, y_test, test_df = evaluate_walk_forward(
        df_final, best['n_estimators'], best['max_depth']
    )
    
    # Calculate operational metrics based on the final fold
    fpr, lead_time_mins = calculate_final_metrics(model, X_test, y_test, test_df, threshold, best['H'])
    
    print(f"\nFinal Cross-Validated Results (Averages across folds):")
    print(f"  Window (W): {best['W']} steps ({best['W']*5} mins)")
    print(f"  Horizon (H): {best['H']} steps ({best['H']*5} mins)")
    print(f"  RF depth: {best['max_depth']}, trees: {best['n_estimators']}")
    print(f"  Avg Threshold: {threshold:.3f}")
    print(f"  --------------------------")
    print(f"  Mean Precision: {precision:.2%}")
    print(f"  Mean Recall: {recall:.2%}")
    print(f"  False Positive Rate (Final Fold): {fpr:.2%}")
    print(f"  Avg Detection Lead Time: {lead_time_mins:.1f} minutes")
    
    save_final_plots(model, X_test, y_test, TARGET_RECALL, threshold)
    print("\nSuccess: Optimized plots saved for the final temporal fold.")

if __name__ == "__main__":
    main()