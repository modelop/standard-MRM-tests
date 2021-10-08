# Standard Risk Tests
This ModelOp Center monitor computes **stability** metrics, including Population Stability Index (**PSI**) and Characteristic Stability Indices (**CSI**), and their breakdown by buckets. It also computes **disparity** metrics (with respect to reference groups) and **group** metrics on **protected classes**, such as **race** or **gender**, as well as classification metrics such as **AUC**, **Accuracy**, **Precision**, **Recall**, and **F1_score**.

## Input Assets

| Type          | Number | Description                                           |
| ------------- | ------ | ----------------------------------------------------- |
| Baseline Data | **1**  | A dataset corresponding to training/reference data    |
| Sample Data   | **1**  | A dataset corresponding to a slice of production data |

## Assumptions & Requirements
 - Underlying `BUSINESS_MODEL` being monitored has an **extended input schema** asset.
 - `BUSINESS_MODEL` is a **classification** model.
 - Protected classes under consideration are **categorical** features.
 - Input data must contain:
     - 1 column with **role=label** (ground truth) 
     - 1 column with **role=score** (model output) 
     - At least 1 column with **protected_class=true** (protected attribute).

## Execution
1. `init` function extracts **extended** input schema (corresponding to the `BUSINESS_MODEL` being monitored) from job JSON.
2. **Monitoring parameters** are set based on the schema above. `predictors`, `feature_dataclass`, `special_values`, `score_column`, `label_column`, `protected_classes`, and `weight_column` are determined accordingly.
3. `metrics` function runs a **stability analysis** test.
   - For each `categorical` feature, the number of groups (`n_groups`) to break the data into is set by default to be equal to the number of unique values of this feature.
   - For each `numerical` feature, `n_groups` is set to **2** if this feature has more than 1 unique value. Otherwise, `n_groups` is set to **1**.
4. `metrics` function runs an **Aequitas Bias** test and an **Aequitas Group** test for each protected class in the list of protected classes. A reference group for each protected class is chosen by default (first occurence).
5. `metrics` function runs a **classification performance** test.
6. Each test result is appended to the corresponding list of tests to be returned by the model.

## Monitor Output

```JSON
{
    "accuracy": <accuracy>,
    "auc": <auc>,
    "f1_score": <f1_score>,
    "precision": <precision>,
    "recall": <recall>,
    "confusion_matrix": <confusion_matrix>,
    "stability": [
        {
            "test_name": "Stability Analysis",
            "test_category": "stability",
            "test_type": "stability_analysis",
            "test_id": "stability_stability_analysis",
            "values": {
                <predictive_feature_1>: {
                    "stability_analysis_table": <stability_analysis_table>,
                    "stability_index": <stability_index>,
                    "stability_chisq": <stability_chisq>,
                    "stability_ks": <stability_ks>
                },
                ...:...,
                <predictive_feature_n>: {
                    "stability_analysis_table": <stability_analysis_table>,
                    "stability_index": <stability_index>,
                    "stability_chisq": <stability_chisq>,
                    "stability_ks": <stability_ks>
                },
                <score_column>: {
                    "stability_analysis_table": <stability_analysis_table>,
                    "stability_index": <stability_index>,
                    "stability_chisq": <stability_chisq>,
                    "stability_ks": <stability_ks>
                }
            }
        }
    ],
    "performance": [
        {
            "test_category": "performance",
            "test_name": "Classification Metrics",
            "test_type": "classification_metrics",
            "test_id": "performance_classification_metrics",
            "values": {
                "accuracy": <accuracy>,
                "auc": <auc>,
                "f1_score": <f1_score>,
                "precision": <precision>,
                "recall": <recall>,
                "confusion_matrix": <confusion_matrix>
            }
        }
    ],
    "bias":[
        <aequitas_bias_test_result>, <aequitas_group_test_result> for protected_class in protected_classes
    ]
}
```