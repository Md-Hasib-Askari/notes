# Hyperparameter Tuning

## Learning Objectives
- Understand hyperparameter optimization concepts and strategies
- Implement various tuning algorithms (Grid Search, Random Search, Bayesian Optimization)
- Use advanced optimization libraries (Optuna, Hyperopt, Ray Tune)
- Apply early stopping and pruning techniques
- Optimize hyperparameters for different ML algorithms
- Implement distributed and parallel tuning

## Introduction

Hyperparameter tuning is the process of finding optimal hyperparameter values that maximize model performance. Unlike model parameters learned during training, hyperparameters are set before training and control the learning process.

### Key Benefits
- **Performance**: Significantly improves model accuracy
- **Efficiency**: Reduces training time and computational costs
- **Automation**: Systematically explores parameter space
- **Reproducibility**: Ensures consistent optimization process
- **Scalability**: Handles complex parameter spaces

## Core Concepts

### 1. Hyperparameter Types
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Example hyperparameter spaces for different algorithms
HYPERPARAMETER_SPACES = {
    'random_forest': {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    },
    
    'svm': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'degree': [2, 3, 4],  # Only for poly kernel
        'class_weight': [None, 'balanced']
    },
    
    'neural_network': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'lbfgs', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [200, 500, 1000]
    }
}

def get_model_and_params(algorithm):
    """Get model instance and parameter space"""
    if algorithm == 'random_forest':
        return RandomForestClassifier(random_state=42), HYPERPARAMETER_SPACES['random_forest']
    elif algorithm == 'svm':
        return SVC(random_state=42), HYPERPARAMETER_SPACES['svm']
    elif algorithm == 'neural_network':
        return MLPClassifier(random_state=42), HYPERPARAMETER_SPACES['neural_network']
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
```

### 2. Search Strategies
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
import time

def grid_search_tuning(X_train, X_test, y_train, y_test, algorithm='random_forest'):
    """Exhaustive grid search hyperparameter tuning"""
    print(f"=== Grid Search for {algorithm} ===")
    
    model, param_grid = get_model_and_params(algorithm)
    
    # Reduce parameter space for demo
    if algorithm == 'random_forest':
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }
    
    start_time = time.time()
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    end_time = time.time()
    
    # Evaluate best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Total fits: {len(grid_search.cv_results_['params'])}")
    
    return grid_search

def random_search_tuning(X_train, X_test, y_train, y_test, algorithm='random_forest', n_iter=20):
    """Random search hyperparameter tuning"""
    print(f"=== Random Search for {algorithm} ===")
    
    model, param_distributions = get_model_and_params(algorithm)
    
    start_time = time.time()
    
    # Random search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    end_time = time.time()
    
    # Evaluate best model
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Total fits: {n_iter}")
    
    return random_search

# Example usage
def compare_search_strategies():
    """Compare different search strategies"""
    # Generate sample data
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=5, 
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Grid search
    grid_result = grid_search_tuning(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*50 + "\n")
    
    # Random search
    random_result = random_search_tuning(X_train, X_test, y_train, y_test, n_iter=20)
    
    return grid_result, random_result

# Run comparison
# grid_res, random_res = compare_search_strategies()
```

## Advanced Optimization Libraries

### 1. Optuna - Efficient Hyperparameter Optimization
```python
import optuna
from optuna.integration import SklearnIntegration
from sklearn.model_selection import cross_val_score

def optuna_optimization(X_train, y_train, X_test, y_test, n_trials=100):
    """Hyperparameter optimization using Optuna"""
    
    def objective(trial):
        """Objective function for Optuna optimization"""
        
        # Suggest hyperparameters
        algorithm = trial.suggest_categorical('algorithm', ['random_forest', 'svm'])
        
        if algorithm == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }
            model = RandomForestClassifier(**params, random_state=42)
            
        elif algorithm == 'svm':
            params = {
                'C': trial.suggest_float('C', 0.1, 100, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
            }
            if params['kernel'] == 'rbf':
                params['gamma'] = trial.suggest_float('gamma', 1e-4, 1e-1, log=True)
            
            model = SVC(**params, random_state=42)
        
        # Cross-validation score
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        return scores.mean()
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner()
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Results
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    # Train final model
    best_params = study.best_params.copy()
    algorithm = best_params.pop('algorithm')
    
    if algorithm == 'random_forest':
        final_model = RandomForestClassifier(**best_params, random_state=42)
    elif algorithm == 'svm':
        final_model = SVC(**best_params, random_state=42)
    
    final_model.fit(X_train, y_train)
    test_accuracy = final_model.score(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    return study, final_model

# Advanced Optuna features
def optuna_with_pruning(X_train, y_train, n_trials=50):
    """Optuna optimization with early stopping/pruning"""
    
    def objective_with_pruning(trial):
        """Objective with intermediate value reporting for pruning"""
        
        # Suggest parameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
        }
        
        model = RandomForestClassifier(**params, random_state=42)
        
        # Evaluate incrementally for pruning
        from sklearn.model_selection import StratifiedKFold
        
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            score = model.score(X_fold_val, y_fold_val)
            scores.append(score)
            
            # Report intermediate value
            trial.report(np.mean(scores), fold)
            
            # Prune if necessary
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(scores)
    
    # Study with aggressive pruning
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
            interval_steps=1
        )
    )
    
    study.optimize(objective_with_pruning, n_trials=n_trials)
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Best value: {study.best_value:.4f}")
    
    return study

def visualize_optuna_results(study):
    """Visualize Optuna optimization results"""
    try:
        # Optimization history
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.show()
        
        # Parameter importance
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.show()
        
        # Parallel coordinate plot
        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.show()
        
        # Contour plot (for 2D parameter relationships)
        if len(study.best_params) >= 2:
            param_names = list(study.best_params.keys())[:2]
            fig4 = optuna.visualization.plot_contour(study, params=param_names)
            fig4.show()
            
    except ImportError:
        print("Install plotly for visualizations: pip install plotly")
```

### 2. Hyperopt - Bayesian Optimization
```python
try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    from hyperopt.early_stop import no_progress_loss
    
    def hyperopt_optimization(X_train, y_train, X_test, y_test, max_evals=100):
        """Hyperparameter optimization using Hyperopt"""
        
        # Define search space
        space = {
            'algorithm': hp.choice('algorithm', [
                {
                    'type': 'random_forest',
                    'n_estimators': hp.choice('rf_n_estimators', [50, 100, 200, 500]),
                    'max_depth': hp.choice('rf_max_depth', [10, 20, 30, None]),
                    'min_samples_split': hp.choice('rf_min_samples_split', [2, 5, 10]),
                    'min_samples_leaf': hp.choice('rf_min_samples_leaf', [1, 2, 4])
                },
                {
                    'type': 'svm',
                    'C': hp.lognormal('svm_C', 0, 1),
                    'kernel': hp.choice('svm_kernel', ['linear', 'rbf']),
                    'gamma': hp.lognormal('svm_gamma', -3, 1)
                }
            ])
        }
        
        def objective(params):
            """Objective function for Hyperopt"""
            try:
                algorithm = params['algorithm']
                
                if algorithm['type'] == 'random_forest':
                    model = RandomForestClassifier(
                        n_estimators=algorithm['n_estimators'],
                        max_depth=algorithm['max_depth'],
                        min_samples_split=algorithm['min_samples_split'],
                        min_samples_leaf=algorithm['min_samples_leaf'],
                        random_state=42
                    )
                elif algorithm['type'] == 'svm':
                    model = SVC(
                        C=algorithm['C'],
                        kernel=algorithm['kernel'],
                        gamma=algorithm['gamma'] if algorithm['kernel'] == 'rbf' else 'scale',
                        random_state=42
                    )
                
                # Cross-validation
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                
                # Hyperopt minimizes, so return negative score
                return {'loss': -scores.mean(), 'status': STATUS_OK}
                
            except Exception as e:
                return {'loss': float('inf'), 'status': STATUS_OK}
        
        # Optimize
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            early_stop_fn=no_progress_loss(20)  # Early stopping
        )
        
        # Get best parameters
        best_params = space_eval(space, best)
        print(f"Best parameters: {best_params}")
        
        # Train final model
        algorithm = best_params['algorithm']
        if algorithm['type'] == 'random_forest':
            final_model = RandomForestClassifier(
                n_estimators=algorithm['n_estimators'],
                max_depth=algorithm['max_depth'],
                min_samples_split=algorithm['min_samples_split'],
                min_samples_leaf=algorithm['min_samples_leaf'],
                random_state=42
            )
        elif algorithm['type'] == 'svm':
            final_model = SVC(
                C=algorithm['C'],
                kernel=algorithm['kernel'],
                gamma=algorithm['gamma'] if algorithm['kernel'] == 'rbf' else 'scale',
                random_state=42
            )
        
        final_model.fit(X_train, y_train)
        test_accuracy = final_model.score(X_test, y_test)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        return trials, final_model, best_params
        
except ImportError:
    print("Install hyperopt: pip install hyperopt")
    
    def hyperopt_optimization(*args, **kwargs):
        print("Hyperopt not available. Install with: pip install hyperopt")
        return None, None, None
```

### 3. Ray Tune - Distributed Hyperparameter Tuning
```python
try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
    
    def ray_tune_optimization(X_train, y_train, X_test, y_test, num_samples=50):
        """Distributed hyperparameter tuning with Ray Tune"""
        
        def train_model(config):
            """Training function for Ray Tune"""
            from sklearn.model_selection import cross_val_score
            
            if config['algorithm'] == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=config['n_estimators'],
                    max_depth=config['max_depth'],
                    min_samples_split=config['min_samples_split'],
                    random_state=42
                )
            elif config['algorithm'] == 'svm':
                model = SVC(
                    C=config['C'],
                    kernel=config['kernel'],
                    gamma=config.get('gamma', 'scale'),
                    random_state=42
                )
            
            # Cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            
            # Report result
            tune.report(accuracy=scores.mean())
        
        # Define search space
        search_space = {
            'algorithm': tune.choice(['random_forest', 'svm']),
            'n_estimators': tune.choice([50, 100, 200, 500]),
            'max_depth': tune.choice([10, 20, 30]),
            'min_samples_split': tune.choice([2, 5, 10]),
            'C': tune.loguniform(0.1, 100),
            'kernel': tune.choice(['linear', 'rbf']),
            'gamma': tune.loguniform(1e-4, 1e-1)
        }
        
        # Scheduler for early stopping
        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=10,
            grace_period=1,
            reduction_factor=2
        )
        
        # Search algorithm
        search_alg = OptunaSearch(metric="accuracy", mode="max")
        
        # Reporter
        reporter = CLIReporter(
            metric_columns=["accuracy", "training_iteration"]
        )
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(num_cpus=4)  # Adjust based on your system
        
        # Run tuning
        result = tune.run(
            train_model,
            config=search_space,
            metric="accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            progress_reporter=reporter,
            verbose=1
        )
        
        # Get best result
        best_trial = result.get_best_trial("accuracy", "max", "last")
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial accuracy: {best_trial.last_result['accuracy']:.4f}")
        
        # Shutdown Ray
        ray.shutdown()
        
        return result, best_trial
        
except ImportError:
    print("Install Ray Tune: pip install ray[tune] optuna")
    
    def ray_tune_optimization(*args, **kwargs):
        print("Ray Tune not available. Install with: pip install ray[tune] optuna")
        return None, None
```

## Advanced Techniques

### 1. Multi-Objective Optimization
```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time

def multi_objective_optimization(X_train, y_train, X_test, y_test):
    """Multi-objective hyperparameter optimization"""
    
    def objective(trial):
        """Multi-objective function optimizing accuracy and training time"""
        
        # Suggest parameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
        }
        
        model = RandomForestClassifier(**params, random_state=42)
        
        # Measure training time
        start_time = time.time()
        
        # Cross-validation for accuracy
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        accuracy = scores.mean()
        
        training_time = time.time() - start_time
        
        # Return multiple objectives (Optuna supports multi-objective)
        return accuracy, -training_time  # Maximize accuracy, minimize time
    
    # Multi-objective study
    study = optuna.create_study(
        directions=['maximize', 'maximize'],  # Both objectives to maximize
        sampler=optuna.samplers.NSGAIISampler()
    )
    
    study.optimize(objective, n_trials=50)
    
    # Get Pareto front
    pareto_front = []
    for trial in study.best_trials:
        pareto_front.append({
            'params': trial.params,
            'accuracy': trial.values[0],
            'training_time': -trial.values[1]
        })
    
    print(f"Number of Pareto optimal solutions: {len(pareto_front)}")
    
    # Show trade-offs
    for i, solution in enumerate(pareto_front[:5]):  # Show top 5
        print(f"Solution {i+1}:")
        print(f"  Accuracy: {solution['accuracy']:.4f}")
        print(f"  Training time: {solution['training_time']:.2f}s")
        print(f"  Params: {solution['params']}")
        print()
    
    return study, pareto_front

def weighted_objective_optimization(X_train, y_train, weights={'accuracy': 0.7, 'speed': 0.3}):
    """Single objective with weighted combination of metrics"""
    
    def weighted_objective(trial):
        """Weighted combination of accuracy and speed"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 30)
        }
        
        model = RandomForestClassifier(**params, random_state=42)
        
        # Measure both accuracy and speed
        start_time = time.time()
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        training_time = time.time() - start_time
        
        accuracy = scores.mean()
        # Normalize speed (inverse of time, normalized to [0,1])
        speed_score = 1 / (1 + training_time)  # Higher is better
        
        # Weighted combination
        combined_score = (weights['accuracy'] * accuracy + 
                         weights['speed'] * speed_score)
        
        return combined_score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(weighted_objective, n_trials=50)
    
    return study
```

### 2. Neural Architecture Search (NAS)
```python
from sklearn.neural_network import MLPClassifier

def neural_architecture_search(X_train, y_train, n_trials=100):
    """Neural Architecture Search for MLPClassifier"""
    
    def nas_objective(trial):
        """Optimize neural network architecture"""
        
        # Number of hidden layers
        n_layers = trial.suggest_int('n_layers', 1, 4)
        
        # Hidden layer sizes
        hidden_layers = []
        for i in range(n_layers):
            layer_size = trial.suggest_int(f'layer_{i}_size', 10, 200)
            hidden_layers.append(layer_size)
        
        # Other hyperparameters
        params = {
            'hidden_layer_sizes': tuple(hidden_layers),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
            'max_iter': 500,
            'random_state': 42
        }
        
        model = MLPClassifier(**params)
        
        try:
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            return scores.mean()
        except:
            return 0.0  # Return low score for failed configurations
    
    study = optuna.create_study(direction='maximize')
    study.optimize(nas_objective, n_trials=n_trials)
    
    print(f"Best architecture: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    return study

def advanced_nas_with_regularization(X_train, y_train):
    """Advanced NAS with regularization techniques"""
    
    def advanced_nas_objective(trial):
        """NAS with dropout, batch normalization consideration"""
        
        # Architecture parameters
        n_layers = trial.suggest_int('n_layers', 2, 5)
        
        hidden_layers = []
        for i in range(n_layers):
            if i == 0:  # First layer
                layer_size = trial.suggest_int(f'layer_{i}_size', 50, 300)
            else:  # Subsequent layers (typically decreasing)
                prev_size = hidden_layers[-1]
                layer_size = trial.suggest_int(f'layer_{i}_size', 10, prev_size)
            hidden_layers.append(layer_size)
        
        # Regularization parameters
        alpha = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
        
        # Learning parameters
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True)
        
        # Batch size (affects learning dynamics)
        batch_size = trial.suggest_categorical('batch_size', ['auto', 32, 64, 128])
        
        params = {
            'hidden_layer_sizes': tuple(hidden_layers),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'solver': 'adam',
            'alpha': alpha,
            'batch_size': batch_size,
            'learning_rate_init': learning_rate_init,
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'random_state': 42
        }
        
        model = MLPClassifier(**params)
        
        try:
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            return scores.mean()
        except:
            return 0.0
    
    study = optuna.create_study(direction='maximize')
    study.optimize(advanced_nas_objective, n_trials=100)
    
    return study
```

### 3. Ensemble Hyperparameter Optimization
```python
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

def optimize_ensemble_hyperparameters(X_train, y_train, X_test, y_test):
    """Optimize hyperparameters for ensemble methods"""
    
    def ensemble_objective(trial):
        """Optimize ensemble configuration"""
        
        # Base estimators configuration
        estimators = []
        
        # Random Forest
        if trial.suggest_categorical('include_rf', [True, False]):
            rf_params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10)
            }
            rf = RandomForestClassifier(**rf_params, random_state=42)
            estimators.append(('rf', rf))
        
        # SVM
        if trial.suggest_categorical('include_svm', [True, False]):
            svm_params = {
                'C': trial.suggest_float('svm_C', 0.1, 100, log=True),
                'kernel': trial.suggest_categorical('svm_kernel', ['linear', 'rbf'])
            }
            if svm_params['kernel'] == 'rbf':
                svm_params['gamma'] = trial.suggest_float('svm_gamma', 1e-4, 1e-1, log=True)
            
            svm = SVC(**svm_params, probability=True, random_state=42)
            estimators.append(('svm', svm))
        
        # Neural Network
        if trial.suggest_categorical('include_nn', [True, False]):
            nn_params = {
                'hidden_layer_sizes': trial.suggest_categorical('nn_hidden_layers', 
                                                               [(50,), (100,), (50, 50)]),
                'alpha': trial.suggest_float('nn_alpha', 1e-5, 1e-1, log=True)
            }
            nn = MLPClassifier(**nn_params, max_iter=500, random_state=42)
            estimators.append(('nn', nn))
        
        if len(estimators) < 2:
            return 0.0  # Need at least 2 estimators for ensemble
        
        # Ensemble method
        ensemble_type = trial.suggest_categorical('ensemble_type', ['voting', 'stacking'])
        
        if ensemble_type == 'voting':
            voting_type = trial.suggest_categorical('voting_type', ['soft', 'hard'])
            ensemble = VotingClassifier(estimators=estimators, voting=voting_type)
        else:  # stacking
            meta_estimator = LogisticRegression(random_state=42)
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_estimator,
                cv=3
            )
        
        try:
            scores = cross_val_score(ensemble, X_train, y_train, cv=3, scoring='accuracy')
            return scores.mean()
        except:
            return 0.0
    
    study = optuna.create_study(direction='maximize')
    study.optimize(ensemble_objective, n_trials=50)
    
    print(f"Best ensemble configuration: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    return study
```

## Performance Monitoring and Analysis

### 1. Hyperparameter Importance Analysis
```python
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_hyperparameter_importance(study):
    """Analyze hyperparameter importance from Optuna study"""
    
    # Parameter importance
    importance = optuna.importance.get_param_importances(study)
    
    # Plot importance
    plt.figure(figsize=(10, 6))
    params = list(importance.keys())
    values = list(importance.values())
    
    plt.barh(params, values)
    plt.xlabel('Importance')
    plt.title('Hyperparameter Importance')
    plt.tight_layout()
    plt.show()
    
    # Print importance values
    print("Hyperparameter Importance:")
    for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{param}: {imp:.4f}")
    
    return importance

def analyze_parameter_correlations(study):
    """Analyze correlations between hyperparameters and performance"""
    
    # Extract trial data
    trials_df = study.trials_dataframe()
    
    # Correlation with objective value
    numeric_columns = trials_df.select_dtypes(include=[np.number]).columns
    correlations = trials_df[numeric_columns].corr()['value'].sort_values(ascending=False)
    
    # Plot correlations
    plt.figure(figsize=(10, 8))
    correlations.drop('value').plot(kind='barh')
    plt.xlabel('Correlation with Objective')
    plt.title('Parameter Correlations with Performance')
    plt.tight_layout()
    plt.show()
    
    return correlations

def convergence_analysis(study):
    """Analyze optimization convergence"""
    
    # Extract objective values over trials
    trial_numbers = [trial.number for trial in study.trials]
    objective_values = [trial.value for trial in study.trials if trial.value is not None]
    
    # Plot convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(trial_numbers[:len(objective_values)], objective_values)
    plt.xlabel('Trial Number')
    plt.ylabel('Objective Value')
    plt.title('Optimization Convergence')
    plt.grid(True)
    
    # Best value over time
    best_values = []
    current_best = float('-inf')
    for value in objective_values:
        if value > current_best:
            current_best = value
        best_values.append(current_best)
    
    plt.subplot(1, 2, 2)
    plt.plot(trial_numbers[:len(best_values)], best_values)
    plt.xlabel('Trial Number')
    plt.ylabel('Best Value So Far')
    plt.title('Best Value Over Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return best_values
```

### 2. Resource Usage Monitoring
```python
import psutil
import threading
import time
from collections import defaultdict

class ResourceMonitor:
    """Monitor resource usage during hyperparameter tuning"""
    
    def __init__(self):
        self.monitoring = False
        self.data = defaultdict(list)
        self.start_time = None
        
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
            
    def _monitor_resources(self):
        """Monitor system resources"""
        while self.monitoring:
            current_time = time.time() - self.start_time
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.data['cpu_percent'].append((current_time, cpu_percent))
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.data['memory_percent'].append((current_time, memory.percent))
            self.data['memory_used_gb'].append((current_time, memory.used / (1024**3)))
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.data['disk_read_mb'].append((current_time, disk_io.read_bytes / (1024**2)))
                self.data['disk_write_mb'].append((current_time, disk_io.write_bytes / (1024**2)))
            
            time.sleep(1)  # Monitor every second
            
    def plot_usage(self):
        """Plot resource usage"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU usage
        if self.data['cpu_percent']:
            times, values = zip(*self.data['cpu_percent'])
            axes[0, 0].plot(times, values)
            axes[0, 0].set_title('CPU Usage %')
            axes[0, 0].set_xlabel('Time (seconds)')
            axes[0, 0].set_ylabel('CPU %')
            axes[0, 0].grid(True)
        
        # Memory usage
        if self.data['memory_used_gb']:
            times, values = zip(*self.data['memory_used_gb'])
            axes[0, 1].plot(times, values)
            axes[0, 1].set_title('Memory Usage')
            axes[0, 1].set_xlabel('Time (seconds)')
            axes[0, 1].set_ylabel('Memory (GB)')
            axes[0, 1].grid(True)
        
        # Disk read
        if self.data['disk_read_mb']:
            times, values = zip(*self.data['disk_read_mb'])
            axes[1, 0].plot(times, values)
            axes[1, 0].set_title('Disk Read')
            axes[1, 0].set_xlabel('Time (seconds)')
            axes[1, 0].set_ylabel('Read (MB)')
            axes[1, 0].grid(True)
        
        # Disk write
        if self.data['disk_write_mb']:
            times, values = zip(*self.data['disk_write_mb'])
            axes[1, 1].plot(times, values)
            axes[1, 1].set_title('Disk Write')
            axes[1, 1].set_xlabel('Time (seconds)')
            axes[1, 1].set_ylabel('Write (MB)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def get_summary(self):
        """Get resource usage summary"""
        summary = {}
        
        for metric, data in self.data.items():
            if data:
                values = [value for _, value in data]
                summary[metric] = {
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'std': np.std(values)
                }
        
        return summary

def monitored_optimization_example():
    """Example of hyperparameter optimization with resource monitoring"""
    
    # Generate sample data
    X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize monitor
    monitor = ResourceMonitor()
    
    def monitored_objective(trial):
        """Objective function with monitoring"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
        }
        
        model = RandomForestClassifier(**params, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        return scores.mean()
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(monitored_objective, n_trials=50)
        
        print(f"Best value: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
    
    # Analyze resource usage
    summary = monitor.get_summary()
    print("\nResource Usage Summary:")
    for metric, stats in summary.items():
        print(f"{metric}:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Max: {stats['max']:.2f}")
        print(f"  Min: {stats['min']:.2f}")
    
    # Plot usage
    monitor.plot_usage()
    
    return study, monitor
```

## Best Practices and Common Pitfalls

### 1. Search Space Design
```python
def design_effective_search_space():
    """Guidelines for designing effective search spaces"""
    
    # Good search space design principles
    search_space_examples = {
        'random_forest': {
            # Use appropriate ranges based on data size
            'n_estimators': [10, 50, 100, 200, 500],  # Start small, go bigger
            'max_depth': [3, 5, 10, 20, None],  # Include None for unlimited
            'min_samples_split': [2, 5, 10, 20],  # Relative to dataset size
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.5, 1.0],  # Different strategies
            'bootstrap': [True, False]
        },
        
        'svm': {
            # Use log scale for C and gamma
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['linear', 'rbf', 'poly'],
            'degree': [2, 3, 4, 5],  # Only relevant for poly kernel
        },
        
        'neural_network': {
            # Architecture parameters
            'hidden_layer_sizes': [
                (50,), (100,), (200,),  # Single layer
                (50, 50), (100, 50), (100, 100),  # Two layers
                (100, 50, 25)  # Three layers
            ],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'lbfgs', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],  # L2 regularization
            'learning_rate_init': [0.001, 0.01, 0.1],
            'max_iter': [200, 500, 1000]
        }
    }
    
    # Bad practices to avoid
    bad_practices = {
        'too_granular': {
            # Don't be too granular - computational waste
            'n_estimators': list(range(50, 501, 10)),  # 46 values!
            'max_depth': list(range(1, 51)),  # 50 values!
        },
        
        'inappropriate_ranges': {
            # Ranges that don't make sense
            'n_estimators': [1, 2, 3],  # Too small
            'C': [1000, 10000, 100000],  # Likely too large
        },
        
        'missing_important_values': {
            # Missing important boundary values
            'max_depth': [5, 10, 15],  # Missing None (unlimited)
            'gamma': [0.01, 0.1, 1],  # Missing 'scale' and 'auto'
        }
    }
    
    return search_space_examples, bad_practices

def adaptive_search_space(trial_number, best_params=None):
    """Adapt search space based on optimization progress"""
    
    if trial_number < 20:  # Initial exploration
        space = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10, 20]
        }
    elif trial_number < 50:  # Focused search
        if best_params:
            # Narrow search around best found parameters
            best_n_est = best_params.get('n_estimators', 100)
            space = {
                'n_estimators': [max(10, best_n_est-50), best_n_est, best_n_est+50],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        else:
            space = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
    else:  # Fine-tuning
        if best_params:
            best_n_est = best_params.get('n_estimators', 100)
            best_depth = best_params.get('max_depth', 20)
            space = {
                'n_estimators': [best_n_est-20, best_n_est, best_n_est+20],
                'max_depth': [best_depth-5, best_depth, best_depth+5] if best_depth else [15, 20, 25, None],
                'min_samples_split': [2, 5, 10]
            }
        else:
            space = {
                'n_estimators': [80, 100, 120],
                'max_depth': [15, 20, 25],
                'min_samples_split': [2, 5]
            }
    
    return space
```

### 2. Validation Strategy
```python
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

def robust_validation_strategy(X, y, validation_type='standard'):
    """Implement robust validation for hyperparameter tuning"""
    
    if validation_type == 'standard':
        # Standard k-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
    elif validation_type == 'time_series':
        # Time series cross-validation
        cv = TimeSeriesSplit(n_splits=5)
        
    elif validation_type == 'nested':
        # Nested cross-validation for unbiased performance estimation
        outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=43)
        return outer_cv, inner_cv
    
    return cv

def nested_cross_validation_example(X, y, algorithm='random_forest'):
    """Example of nested cross-validation for unbiased evaluation"""
    
    outer_cv, inner_cv = robust_validation_strategy(X, y, validation_type='nested')
    
    model, param_grid = get_model_and_params(algorithm)
    
    # Reduce parameter grid for demo
    if algorithm == 'random_forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None]
        }
    
    outer_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"Outer fold {fold + 1}/3")
        
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        
        # Inner cross-validation for hyperparameter optimization
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_outer, y_train_outer)
        
        # Evaluate on outer test set
        best_model = grid_search.best_estimator_
        outer_score = best_model.score(X_test_outer, y_test_outer)
        outer_scores.append(outer_score)
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Outer score: {outer_score:.4f}")
    
    print(f"\nNested CV Results:")
    print(f"Mean score: {np.mean(outer_scores):.4f} (+/- {np.std(outer_scores) * 2:.4f})")
    
    return outer_scores
```

### 3. Early Stopping and Efficiency
```python
def efficient_hyperparameter_tuning_pipeline(X_train, y_train, max_time_minutes=30):
    """Efficient hyperparameter tuning with time constraints"""
    
    start_time = time.time()
    max_time_seconds = max_time_minutes * 60
    
    def time_constrained_objective(trial):
        """Objective function with time constraints"""
        
        # Check if we're running out of time
        if time.time() - start_time > max_time_seconds:
            raise optuna.TrialPruned()
        
        # Quick parameter validation
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 3, 30)
        
        # Early exit for clearly bad parameters
        if n_estimators < 10 or max_depth < 3:
            return 0.0
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            random_state=42
        )
        
        # Use smaller CV for speed
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        return scores.mean()
    
    # Study with time-based stopping
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=10,
            reduction_factor=3
        )
    )
    
    # Optimize with timeout
    study.optimize(
        time_constrained_objective,
        timeout=max_time_seconds,
        n_jobs=1  # Single job to avoid resource conflicts
    )
    
    elapsed_time = time.time() - start_time
    print(f"Optimization completed in {elapsed_time:.2f} seconds")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best value: {study.best_value:.4f}")
    
    return study

def progressive_hyperparameter_tuning(X_train, y_train, stages=['coarse', 'fine', 'ultra_fine']):
    """Progressive hyperparameter tuning with increasing precision"""
    
    results = {}
    
    for stage in stages:
        print(f"\n=== {stage.upper()} TUNING ===")
        
        if stage == 'coarse':
            # Fast, broad search
            n_trials = 20
            cv_folds = 3
            param_space = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 10]
            }
            
        elif stage == 'fine':
            # More focused search around best coarse result
            n_trials = 50
            cv_folds = 5
            if 'coarse' in results:
                best_coarse = results['coarse'].best_params
                base_n_est = best_coarse.get('n_estimators', 100)
                base_depth = best_coarse.get('max_depth', 20)
                
                param_space = {
                    'n_estimators': [max(10, base_n_est-50), base_n_est, base_n_est+50],
                    'max_depth': [base_depth-5, base_depth, base_depth+5] if base_depth else [15, 20, 25, None],
                    'min_samples_split': [2, 5, 10, 15]
                }
            else:
                param_space = {
                    'n_estimators': [80, 100, 120, 150],
                    'max_depth': [15, 20, 25, None],
                    'min_samples_split': [2, 5, 10]
                }
                
        elif stage == 'ultra_fine':
            # Very precise search
            n_trials = 100
            cv_folds = 5
            if 'fine' in results:
                best_fine = results['fine'].best_params
                base_n_est = best_fine.get('n_estimators', 100)
                base_depth = best_fine.get('max_depth', 20)
                base_split = best_fine.get('min_samples_split', 5)
                
                param_space = {
                    'n_estimators': [base_n_est-10, base_n_est, base_n_est+10],
                    'max_depth': [base_depth-2, base_depth, base_depth+2] if base_depth else [18, 20, 22, None],
                    'min_samples_split': [max(2, base_split-2), base_split, base_split+2],
                    'min_samples_leaf': [1, 2, 3, 4]
                }
            else:
                param_space = {
                    'n_estimators': [90, 100, 110],
                    'max_depth': [18, 20, 22],
                    'min_samples_split': [4, 5, 6],
                    'min_samples_leaf': [1, 2, 3]
                }
        
        # Run optimization for this stage
        def stage_objective(trial):
            params = {}
            for param, values in param_space.items():
                if isinstance(values[0], int):
                    params[param] = trial.suggest_categorical(param, values)
                else:
                    params[param] = trial.suggest_categorical(param, values)
            
            model = RandomForestClassifier(**params, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(stage_objective, n_trials=n_trials)
        
        results[stage] = study
        
        print(f"Best {stage} params: {study.best_params}")
        print(f"Best {stage} score: {study.best_value:.4f}")
    
    return results
```

## Summary

Hyperparameter tuning is crucial for maximizing ML model performance through:

1. **Search Strategies**: Grid Search, Random Search, Bayesian Optimization
2. **Advanced Libraries**: Optuna, Hyperopt, Ray Tune for efficient optimization
3. **Multi-Objective**: Balancing accuracy, speed, and resource usage
4. **Architecture Search**: Optimizing neural network architectures
5. **Ensemble Optimization**: Tuning complex ensemble configurations

### Key Takeaways
- Choose appropriate search strategy based on time/resource constraints
- Design effective search spaces with proper ranges and scales
- Use cross-validation for robust performance estimation
- Implement early stopping and pruning for efficiency
- Monitor resource usage during optimization
- Apply progressive tuning for complex problems

### Next Steps
- Experiment with different optimization algorithms
- Integrate with automated ML pipelines
- Explore domain-specific hyperparameter strategies
- Implement distributed tuning for large-scale problems