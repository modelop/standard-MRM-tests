import modelop.monitors.stability as stability
import modelop.monitors.performance as performance
import modelop.monitors.bias as bias
import modelop.schema.infer as infer
import modelop.utils as utils

logger = utils.configure_logger()

MONITORING_PARAMETERS = {}

# modelop.init
def init(job_json):
    """A function to extract input schema from job JSON.

    Args:
        job_json (str): job JSON in a string format.
    """

    # Extract input schema from job JSON
    input_schema_definition = infer.extract_input_schema(job_json)

    logger.info("Input schema definition: %s", input_schema_definition)

    # Get monitoring parameters from schema
    global MONITORING_PARAMETERS
    MONITORING_PARAMETERS = infer.set_monitoring_parameters(
        schema_json=input_schema_definition, check_schema=True
    )

    logger.info("predictors: %s", MONITORING_PARAMETERS["predictors"])
    logger.info("feature_dataclass: %s", MONITORING_PARAMETERS["feature_dataclass"])
    logger.info("special_values: %s", MONITORING_PARAMETERS["special_values"])
    logger.info("score_column: %s", MONITORING_PARAMETERS["score_column"])
    logger.info("label_column: %s", MONITORING_PARAMETERS["label_column"])
    logger.info("weight_column: %s", MONITORING_PARAMETERS["weight_column"])
    logger.info(
        "protected_classes: %s", str(MONITORING_PARAMETERS["protected_classes"])
    )


# modelop.metrics
def metrics(df_baseline, df_sample):

    score_column = MONITORING_PARAMETERS["score_column"]
    predictors = MONITORING_PARAMETERS["predictors"]

    # Initialize StabilityMonitor
    stability_monitor = stability.StabilityMonitor(
        df_baseline=df_baseline,
        df_sample=df_sample,
        predictors=predictors,
        feature_dataclass=MONITORING_PARAMETERS["feature_dataclass"],
        special_values=MONITORING_PARAMETERS["special_values"],
        score_column=score_column,
        label_column=MONITORING_PARAMETERS["label_column"],
        weight_column=MONITORING_PARAMETERS["weight_column"],
    )

    # Set default n_groups for each predictor and score
    n_groups = {}
    for feature in MONITORING_PARAMETERS["numerical_columns"] + [
        MONITORING_PARAMETERS["score_column"]
    ]:
        # If a feature has more than 1 unique value, set n_groups to 2; else set to 1
        feature_has_distinct_values = int(
            (min(df_baseline[feature]) != max(df_baseline[feature]))
        )
        n_groups[feature] = 1 + feature_has_distinct_values

    # Compute stability metrics
    stability_metrics = stability_monitor.compute_stability_indices(
        n_groups=n_groups, group_cuts={}
    )

    # Initialize ModelEvaluator
    model_evaluator = performance.ModelEvaluator(
        dataframe=df_sample,
        score_column=MONITORING_PARAMETERS["score_column"],
        label_column=MONITORING_PARAMETERS["label_column"],
    )

    # Compute classification metrics
    classification_metrics = model_evaluator.evaluate_performance(
        pre_defined_metrics="classification_metrics"
    )

    result = {
        # Top-level metrics
        str(score_column + "_PSI"): stability_metrics["values"][score_column][
            "stability_index"
        ]
    }

    result.update(
        # Top-level metrics
        {
            str(predictor + "_CSI"): stability_metrics["values"][predictor][
                "stability_index"
            ]
            for predictor in predictors
        }
    )

    result.update(
        # Top-level metrics
        {
            "accuracy": classification_metrics["values"]["accuracy"],
            "precision": classification_metrics["values"]["precision"],
            "recall": classification_metrics["values"]["recall"],
            "auc": classification_metrics["values"]["auc"],
            "f1_score": classification_metrics["values"]["f1_score"],
            "confusion_matrix": classification_metrics["values"]["confusion_matrix"]
        }
    )

    # Add Vanilla output
    result["stability"] = [stability_metrics]
    result["performance"] = [classification_metrics]

    if MONITORING_PARAMETERS["protected_classes"] == []:
        raise ValueError("Input Schema contains no Protected Classes!")

    result["bias"] = []
    for protected_class in MONITORING_PARAMETERS["protected_classes"]:
        # Initialize BiasMonitor
        bias_monitor = bias.BiasMonitor(
            dataframe=df_sample,
            score_column=MONITORING_PARAMETERS["score_column"],
            label_column=MONITORING_PARAMETERS["label_column"],
            protected_class=protected_class,
            reference_group=None,
        )

        # Compute aequitas_bias (disparity) metrics
        bias_metrics = bias_monitor.compute_bias_metrics(
            pre_defined_test="aequitas_bias", thresholds={"min": 0.8, "max": 1.25}
        )
        # Add BiasMonitor Vanilla output
        result["bias"].append(bias_metrics)

        # Top-level metrics
        for group_dict in bias_metrics["values"]:
            result.update(
                {
                    str(
                        protected_class
                        + "_"
                        + group_dict["attribute_value"]
                        + "_statistical_parity"
                    ): group_dict["ppr_disparity"],
                    str(
                        protected_class
                        + "_"
                        + group_dict["attribute_value"]
                        + "_impact_parity"
                    ): group_dict["pprev_disparity"],
                }
            )

        # Compute aequitas_group (Group) metrics
        group_metrics = bias_monitor.compute_group_metrics(
            pre_defined_test="aequitas_group",
        )

        # Add BiasMonitor Vanilla output
        result["bias"].append(group_metrics)

    yield result
