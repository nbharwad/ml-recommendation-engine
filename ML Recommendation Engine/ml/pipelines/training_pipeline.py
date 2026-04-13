"""
Model Training Pipeline — Airflow DAG
========================================
Orchestrates end-to-end ML training workflow.

Pipeline:
1. Data validation
2. Feature engineering (offline)
3. Training data preparation
4. Model training (Two-Tower + DLRM)
5. Offline evaluation with quality gates
6. Model registration
7. Canary deployment to staging
8. A/B test setup

Schedule: Daily (2 AM UTC) with manual trigger option
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

# In production: from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.operators.bash import BashOperator
# from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
# from airflow.utils.dates import days_ago

# ---------------------------------------------------------------------------
# DAG Configuration
# ---------------------------------------------------------------------------

DAG_CONFIG = {
    "dag_id": "recommendation_model_training",
    "description": "End-to-end recommendation model training pipeline",
    "schedule_interval": "0 2 * * *",  # daily at 2 AM UTC
    "start_date": datetime(2026, 1, 1),
    "max_active_runs": 1,
    "catchup": False,
    "tags": ["ml", "recommendation", "training"],
    "default_args": {
        "owner": "ml-platform",
        "depends_on_past": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
        "execution_timeout": timedelta(hours=4),
        "email": ["ml-platform-alerts@company.com"],
        "email_on_failure": True,
    },
}

# ---------------------------------------------------------------------------
# Task Definitions
# ---------------------------------------------------------------------------

def validate_training_data(**context):
    """
    Step 1: Run data validation suite.
    
    Validates:
    - Schema integrity
    - Volume checks (minimum rows)
    - Distribution stability (PSI)
    - Label rate consistency
    
    If validation fails → entire DAG stops.
    """
    from ml.features.data_validation import DataValidationSuite
    import numpy as np
    
    # In production: load from Delta Lake / S3
    # data = spark.read.parquet(f"s3://rec-data/training/{context['ds']}/")
    
    suite = DataValidationSuite().add_standard_rules()
    
    # Placeholder data
    np.random.seed(42)
    n = 1_000_000
    data = {
        "user_id": np.arange(n, dtype=np.float64),
        "item_id": np.arange(n, dtype=np.float64),
        "label": np.random.binomial(1, 0.03, n).astype(np.float64),
        "price": np.random.uniform(1, 200, n),
        "ctr_7d": np.random.uniform(0, 0.1, n),
    }
    
    results = suite.validate(data)
    
    if not results["passed"]:
        raise ValueError(
            f"Data validation FAILED: {results['critical_failures']} critical failures. "
            f"Check validation report for details."
        )
    
    # Push validation results to XCom
    # context["ti"].xcom_push(key="validation_results", value=results)
    return results


def prepare_features(**context):
    """
    Step 2: Offline feature engineering.
    
    Computes features that are too expensive for online:
    - User embeddings from interaction history
    - Item embeddings from content + behavior
    - User-item pair features (historical interactions)
    - Time-decay weighted features
    """
    print(f"Preparing features for training date: {context.get('ds', 'today')}")
    
    # In production (Spark job):
    # feature_config = FeatureConfig.load("s3://rec-config/feature_config.yaml")
    #
    # user_features = spark.sql("""
    #     SELECT user_id,
    #            COUNT(*) as purchase_count_30d,
    #            AVG(order_value) as avg_order_value,
    #            DATEDIFF(CURRENT_DATE, MAX(purchase_date)) as last_purchase_days_ago,
    #            ...
    #     FROM events
    #     WHERE event_date >= DATE_SUB(CURRENT_DATE, 30)
    #     GROUP BY user_id
    # """)
    #
    # item_features = spark.sql("""
    #     SELECT item_id,
    #            SUM(CASE WHEN event_type='CLICK' THEN 1 ELSE 0 END) /
    #            NULLIF(SUM(CASE WHEN event_type='VIEW' THEN 1 ELSE 0 END), 0) as ctr_7d,
    #            ...
    #     FROM events
    #     WHERE event_date >= DATE_SUB(CURRENT_DATE, 7)
    #     GROUP BY item_id
    # """)
    #
    # # Write to S3 for training
    # user_features.write.parquet(f"s3://rec-data/features/user/{context['ds']}/")
    # item_features.write.parquet(f"s3://rec-data/features/item/{context['ds']}/")
    
    return {"user_features_path": f"s3://rec-data/features/user/", "item_features_path": f"s3://rec-data/features/item/"}


def prepare_training_data(**context):
    """
    Step 3: Join features with labels for training.
    
    Creates:
    - Training set (T-30 to T-7)
    - Validation set (T-7 to T-1)
    - Test set (T-1)
    
    Each sample: (user_features, item_features, label)
    """
    print("Preparing training data splits")
    
    # In production:
    # interactions = spark.read.parquet("s3://rec-data/interactions/")
    # user_features = spark.read.parquet(features_path)
    # item_features = spark.read.parquet(features_path)
    #
    # joined = interactions.join(user_features, "user_id").join(item_features, "item_id")
    #
    # train = joined.filter("event_date < DATE_SUB(CURRENT_DATE, 7)")
    # val = joined.filter("event_date BETWEEN DATE_SUB(CURRENT_DATE, 7) AND DATE_SUB(CURRENT_DATE, 1)")
    # test = joined.filter("event_date = DATE_SUB(CURRENT_DATE, 1)")
    
    return {
        "train_path": "s3://rec-data/training/train/",
        "val_path": "s3://rec-data/training/val/",
        "test_path": "s3://rec-data/training/test/",
    }


def train_two_tower_model(**context):
    """
    Step 4a: Train Two-Tower model for candidate generation.
    
    Runs on GPU node (g5.2xlarge).
    Training time: ~2 hours for 100M samples.
    """
    print("Training Two-Tower model")
    
    # In production (KubernetesPodOperator):
    # - Image: rec-training:latest
    # - Node selector: gpu
    # - GPU: 1x A10G
    # - Command: python ml/models/two_tower/model.py --train-data ... --val-data ...
    
    return {"model_path": "s3://rec-models/two_tower/latest/", "model_version": "v2.3.1"}


def train_dlrm_model(**context):
    """
    Step 4b: Train DLRM ranking model.
    
    Runs on GPU node.
    Training time: ~3 hours for 100M samples.
    """
    print("Training DLRM ranking model")
    
    return {"model_path": "s3://rec-models/dlrm/latest/", "model_version": "v2.3.1"}


def train_xgboost_baseline(**context):
    """
    Step 4c: Train XGBoost baseline (for fallback + comparison).
    
    Runs on CPU node.
    Training time: ~30 minutes.
    """
    print("Training XGBoost baseline model")
    
    return {"model_path": "s3://rec-models/xgboost/latest/"}


def evaluate_models(**context):
    """
    Step 5: Offline evaluation with quality gates.
    
    Evaluates all models against test set.
    Blocks deployment if quality gates fail.
    """
    from ml.evaluation.evaluator import ModelEvaluator, EvaluationConfig
    import numpy as np

    config = EvaluationConfig()
    evaluator = ModelEvaluator(config)
    
    # In production: load predictions from model artifacts
    np.random.seed(42)
    n = 100_000
    labels = np.random.binomial(1, 0.03, n)
    candidate_scores = np.random.uniform(0, 0.1, n) + labels * 0.05
    baseline_scores = np.random.uniform(0, 0.1, n) + labels * 0.04
    
    results = evaluator.evaluate(labels, candidate_scores, baseline_scores)
    
    if not results["passed"]:
        raise ValueError(
            "Model evaluation FAILED quality gates. "
            "Deployment blocked. Check evaluation report."
        )
    
    return results


def register_model(**context):
    """
    Step 6: Register model in model registry.
    
    Uses MLflow for model versioning and tracking.
    """
    print("Registering model in MLflow registry")
    
    # In production:
    # import mlflow
    # mlflow.set_tracking_uri("http://mlflow.mlops.svc:5000")
    #
    # with mlflow.start_run(run_name=f"dlrm-{context['ds']}"):
    #     mlflow.log_metrics(evaluation_results["metrics"])
    #     mlflow.pytorch.log_model(model, "dlrm_ranking")
    #     mlflow.register_model("runs:/{run_id}/dlrm_ranking", "recommendation-ranking")
    
    return {"model_version": "v2.3.1", "registered": True}


def deploy_to_staging(**context):
    """
    Step 7: Deploy model to staging for canary validation.
    
    Updates Triton model repository in staging.
    Runs canary validation (recall@100 check).
    """
    print("Deploying to staging Triton server")
    
    # In production:
    # 1. Copy model to staging Triton model repo (S3/EFS)
    # 2. Triton auto-loads new model version
    # 3. Run canary validation queries
    # 4. If recall@100 < 80%, rollback
    
    return {"staging_deployed": True}


def setup_ab_test(**context):
    """
    Step 8: Configure A/B test for the new model.
    
    Sets up 90/10 traffic split:
    - 90% control (current production model)
    - 10% treatment (new model)
    """
    print("Setting up A/B test for new model")
    
    # In production:
    # Update experiment config in etcd/S3
    # experiment = {
    #     "experiment_id": f"exp-dlrm-{context['ds']}",
    #     "variants": {
    #         "control": 0.90,
    #         "treatment": 0.10,
    #     },
    #     "variant_parameters": {
    #         "control": {"model_version": "current"},
    #         "treatment": {"model_version": "v2.3.1"},
    #     },
    # }
    
    return {"experiment_id": f"exp-dlrm-{context.get('ds', 'latest')}"}


def reindex_embeddings(**context):
    """
    Step 9: Reindex item embeddings in Milvus.
    
    After Two-Tower training, all 10M item embeddings need recomputing
    and loading into the ANN index.
    
    Strategy:
    1. Compute all embeddings on GPU cluster (batch)
    2. Build new collection with HNSW index
    3. Canary validation (recall@100 check)
    4. Swap alias to new collection (zero-downtime)
    """
    print("Reindexing item embeddings in Milvus")
    
    # In production (Spark + Milvus):
    # 1. Load item features
    # 2. Run item tower inference (batch)
    # 3. Create new Milvus collection with timestamp
    # 4. Insert embeddings
    # 5. Build HNSW index
    # 6. Validate recall
    # 7. Switch alias
    
    return {"collection": f"item_embeddings_{context.get('ds', 'latest')}"}


def update_feature_store(**context):
    """
    Step 10: Update offline features in Redis.
    
    Batch-computed features (user embeddings, segments, etc.)
    are written to Redis for online serving.
    """
    print("Updating feature store with latest offline features")
    
    # In production:
    # 1. Load computed features from Spark output
    # 2. Pipeline write to Redis
    # 3. Verify write success rate > 99.9%
    
    return {"features_updated": True}


# ---------------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------------

# In production Airflow:
# 
# with DAG(**DAG_CONFIG) as dag:
#     
#     validate = PythonOperator(task_id="validate_data", python_callable=validate_training_data)
#     features = PythonOperator(task_id="prepare_features", python_callable=prepare_features)
#     prepare = PythonOperator(task_id="prepare_training_data", python_callable=prepare_training_data)
#     
#     train_tt = KubernetesPodOperator(
#         task_id="train_two_tower",
#         image="rec-training:latest",
#         arguments=["python", "ml/models/two_tower/model.py", "--train-data", "..."],
#         node_selector={"nvidia.com/gpu.present": "true"},
#         resources={"limits": {"nvidia.com/gpu": "1"}},
#     )
#     
#     train_dlrm_task = KubernetesPodOperator(
#         task_id="train_dlrm",
#         image="rec-training:latest",
#         arguments=["python", "ml/models/dlrm/model.py", "--train-data", "..."],
#         node_selector={"nvidia.com/gpu.present": "true"},
#         resources={"limits": {"nvidia.com/gpu": "1"}},
#     )
#     
#     train_xgb = PythonOperator(task_id="train_xgboost", python_callable=train_xgboost_baseline)
#     evaluate = PythonOperator(task_id="evaluate_models", python_callable=evaluate_models)
#     register = PythonOperator(task_id="register_model", python_callable=register_model)
#     deploy = PythonOperator(task_id="deploy_staging", python_callable=deploy_to_staging)
#     ab_test = PythonOperator(task_id="setup_ab_test", python_callable=setup_ab_test)
#     reindex = PythonOperator(task_id="reindex_embeddings", python_callable=reindex_embeddings)
#     update_fs = PythonOperator(task_id="update_feature_store", python_callable=update_feature_store)
#     
#     # DAG dependency graph:
#     validate >> features >> prepare
#     prepare >> [train_tt, train_dlrm_task, train_xgb]
#     [train_tt, train_dlrm_task, train_xgb] >> evaluate
#     evaluate >> register >> deploy >> ab_test
#     train_tt >> reindex
#     features >> update_fs

"""
DAG Execution Flow:

    validate_data
        │
    prepare_features ─────────────────┐
        │                             │
    prepare_training_data         update_feature_store
        │
    ┌───┼───────────┐
    │   │           │
train_tt  train_dlrm  train_xgb
    │   │           │
    │   └─────┬─────┘
    │         │
    │     evaluate
    │         │
    │     register
    │         │
    │     deploy_staging
    │         │
    │     setup_ab_test
    │
reindex_embeddings
"""

if __name__ == "__main__":
    print("Recommendation Model Training Pipeline")
    print("="*50)
    print("This is an Airflow DAG — run via Airflow scheduler")
    print(f"Schedule: {DAG_CONFIG['schedule_interval']}")
    print(f"Owner: {DAG_CONFIG['default_args']['owner']}")
