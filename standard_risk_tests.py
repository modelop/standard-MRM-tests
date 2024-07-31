import json

import modelop.monitors.bias as bias
import modelop.monitors.drift as drift
import modelop.monitors.performance as performance
import modelop.monitors.stability as stability
import modelop.schema.infer as infer
import modelop.stats.diagnostics as diagnostics
import modelop.utils as utils
from modelop_sdk.utils import dashboard_utils as dashboard_utils

DEPLOYABLE_MODEL = {}
JOB = {}
MODEL_METHODOLOGY = ""


# modelop.init
def init(job_json):
    global DEPLOYABLE_MODEL
    global JOB
    global MODEL_METHODOLOGY

    job = json.loads(job_json["rawJson"])
    DEPLOYABLE_MODEL = job["referenceModel"]
    MODEL_METHODOLOGY = DEPLOYABLE_MODEL.get("storedModel", {}).get("modelMetaData", {}).get("modelMethodology", "")

    JOB = job_json
    infer.validate_schema(job_json)


# modelop.metrics
def metrics(baseline, comparator) -> dict:

    execution_errors_array = []

    result = utils.merge(
        extract_model_fields(execution_errors_array),
        calculate_performance(comparator, execution_errors_array),
        calculate_bias(comparator, execution_errors_array),
        calculate_ks_drift(baseline, comparator, execution_errors_array),
        calculate_ks_concept_drift(baseline, comparator, execution_errors_array),
        calculate_stability(baseline, comparator, execution_errors_array),
        calculate_breusch_pagan(comparator, execution_errors_array),
        calculate_linearity_metrics(comparator, execution_errors_array),
        calculate_ljung_box_q_test(comparator, execution_errors_array),
        calculate_variance_inflation_factor(comparator, execution_errors_array),
        calculate_durbin_watson(comparator, execution_errors_array),
        calculate_engle_lagrange_multiplier_test(comparator, execution_errors_array),
        calculate_anderson_darling_test(comparator, execution_errors_array),
        calculate_cramer_von_mises_test(comparator, execution_errors_array),
        calculate_kolmogorov_smirnov_test(comparator, execution_errors_array),
    )

    result.update({"executionErrors": execution_errors_array})
    result.update({"executionErrorsCount": len(execution_errors_array)})

    yield result


def extract_model_fields(execution_errors_array):
    try:
        return {
            "modelUseCategory": DEPLOYABLE_MODEL.get("storedModel", {})
                .get("modelMetaData", {})
                .get("modelUseCategory", ""),
            "modelOrganization": DEPLOYABLE_MODEL.get("storedModel", {})
                .get("modelMetaData", {})
                .get("modelOrganization", ""),
            "modelRisk": DEPLOYABLE_MODEL.get("storedModel", {})
                .get("modelMetaData", {})
                .get("modelRisk", ""),
            "modelMethodology": MODEL_METHODOLOGY
        }
    except Exception as ex:
        error_message = f"Something went wrong when extracting modelop default fields: {str(ex)}"
        execution_errors_array.append(error_message)
        print(error_message)
        return {}


def calculate_performance(comparator, execution_errors_array):
    try:
        dashboard_utils.assert_df_not_none_and_not_empty(
            comparator, "Required comparator"
        )
        print("Dropping any nulls in comparator dataframe for performance monitors.")
        comparator_cleaned = check_and_drop_nulls(comparator)
        model_evaluator = performance.ModelEvaluator(dataframe=comparator_cleaned, job_json=JOB)
        if "regression" in MODEL_METHODOLOGY.casefold():
            return model_evaluator.evaluate_performance(
                pre_defined_metrics="regression_metrics"
            )
        else:
            return model_evaluator.evaluate_performance(
                pre_defined_metrics="classification_metrics"
            )
    except Exception as ex:
        error_message = f"Error occurred calculating performance metrics: {str(ex)}"
        print(error_message)
        execution_errors_array.append(error_message)
        return {"auc": -99, "r2_score": 99}


def calculate_bias(comparator, execution_errors_array):
    try:
        dashboard_utils.assert_df_not_none_and_not_empty(
            comparator, "Required comparator"
        )
        print("Dropping any nulls in comparator dataframe for bias monitors.")
        comparator_cleaned = check_and_drop_nulls(comparator)
        bias_monitor = bias.BiasMonitor(dataframe=comparator_cleaned, job_json=JOB)
        if "regression" in MODEL_METHODOLOGY.casefold():
            raise Exception("Bias metrics can not be run for regression models.")
        else:
            return bias_monitor.compute_bias_metrics(pre_defined_test="aequitas_bias")
    except Exception as ex:
        error_message = f"Error occurred calculating bias metrics: {str(ex)}"
        print(error_message)
        execution_errors_array.append(error_message)
        return {"Bias_maxPPRDisparityValue": -99, "Bias_minPPRDisparityValue": -99}


def calculate_ks_drift(baseline, sample, execution_errors_array):
    try:
        dashboard_utils.assert_df_not_none_and_not_empty(baseline, "Required baseline")
        dashboard_utils.assert_df_not_none_and_not_empty(sample, "Required comparator")
        drift_test = drift.DriftDetector(
            df_baseline=baseline, df_sample=sample, job_json=JOB
        )
        return drift_test.calculate_drift(pre_defined_test="Kolmogorov-Smirnov")
    except Exception as ex:
        error_message = f"Error occurred while calculating drift: {str(ex)}"
        print(error_message)
        execution_errors_array.append(error_message)
        return {"DataDrift_maxKolmogorov-SmirnovPValue": -99}


def calculate_ks_concept_drift(baseline, sample, execution_errors_array):
    try:
        dashboard_utils.assert_df_not_none_and_not_empty(baseline, "Required baseline")
        dashboard_utils.assert_df_not_none_and_not_empty(sample, "Required comparator")
        concept_drift_test = drift.ConceptDriftDetector(
            df_baseline=baseline, df_sample=sample, job_json=JOB
        )
        return concept_drift_test.calculate_concept_drift(
            pre_defined_test="Kolmogorov-Smirnov"
        )
    except Exception as ex:
        error_message = f"Error occurred while calculating concept drift: {str(ex)}"
        print(error_message)
        execution_errors_array.append(error_message)
        return {"ConceptDrift_maxKolmogorov-SmirnovPValueValue": -99}


def calculate_stability(df_baseline, df_comparator, execution_errors_array):
    try:
        dashboard_utils.assert_df_not_none_and_not_empty(
            df_baseline, "Required baseline"
        )
        dashboard_utils.assert_df_not_none_and_not_empty(
            df_comparator, "Required comparator"
        )
        stability_test = stability.StabilityMonitor(
            df_baseline=df_baseline, df_sample=df_comparator, job_json=JOB
        )
        return stability_test.compute_stability_indices()
    except Exception as ex:
        error_message = f"Error occurred while calculating stability: {str(ex)}"
        print(error_message)
        execution_errors_array.append(error_message)
        return {"CSI_maxCSIValue": -99}


def calculate_breusch_pagan(dataframe, execution_errors_array):
    """A function to run the Breauch-Pagan test on sample data
    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs),
        labels (ground truths) and numerical_columns (predictors)
        execution_errors_array (array): Array for collecting execution errors
    Returns:
        (dict): Breusch-Pagan test results
    """
    try:
        if "regression" in MODEL_METHODOLOGY.casefold():
            dashboard_utils.assert_df_not_none_and_not_empty(
                dataframe, "Required comparator"
            )
            homoscedasticity_metrics = diagnostics.HomoscedasticityMetrics(
                dataframe=dataframe, job_json=JOB
            )
            return homoscedasticity_metrics.breusch_pagan_test()
        else:
            raise Exception(
                "Breusch-Pagan metrics can only be run for regression models."
            )
    except Exception as ex:
        error_message = f"Error occurred while calculating breusch_pagan: {str(ex)}"
        print(error_message)
        execution_errors_array.append(error_message)
        return {"breusch_pagan_f_p_value": -99}


def calculate_variance_inflation_factor(dataframe, execution_errors_array):
    """A function to compute Variance Inflation Factors on sample data
    Args:
        dataframe (pandas.DataFrame): Sample prod data containing numerical_columns (predictors)
        execution_errors_array (array): Array for collecting execution errors
    Returns:
        (dict): Pearson Correlation results
    """
    try:
        dashboard_utils.assert_df_not_none_and_not_empty(
            dataframe, "Required comparator"
        )
        # dataframe=dataframe.astype('float')
        multicollinearity_metrics = diagnostics.MulticollinearityMetrics(
            dataframe=dataframe, job_json=JOB
        )
        return multicollinearity_metrics.variance_inflation_factor()
    except Exception as ex:
        error_message = (
            f"Error occurred while calculating variance_inflation_factor: {str(ex)}"
        )
        print(error_message)
        execution_errors_array.append(error_message)
        return {"Multicollinearity_maxVIFValue": -99}


def calculate_linearity_metrics(dataframe, execution_errors_array):
    """A function to compute Pearson Correlations on sample data
    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs)
        and numerical_columns (predictors)
        execution_errors_array (array): Array for collecting execution errors
    Returns:
        (dict): Pearson Correlation results
    """
    try:
        dashboard_utils.assert_df_not_none_and_not_empty(
            dataframe, "Required comparator"
        )
        linearity_metrics = diagnostics.LinearityMetrics(
            dataframe=dataframe, job_json=JOB
        )
        return linearity_metrics.pearson_correlation()
    except Exception as ex:
        error_message = (
            f"Error occurred while calculating calculate_linearity_metrics: {str(ex)}"
        )
        print(error_message)
        execution_errors_array.append(error_message)
        return {"Linearity_minPearsonCorrelationValue": -99}


def calculate_ljung_box_q_test(dataframe, execution_errors_array):
    """A function to run the Ljung-Box Q test on sample data
    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs),
        labels (ground truths) and numerical_columns (predictors)
        execution_errors_array (array): Array for collecting execution errors
    Returns:
        (dict): Ljung-Box Q test results
    """
    try:
        if "regression" in MODEL_METHODOLOGY.casefold():
            dashboard_utils.assert_df_not_none_and_not_empty(
                dataframe, "Required comparator"
            )
            homoscedasticity_metrics = diagnostics.HomoscedasticityMetrics(
                dataframe=dataframe, job_json=JOB
            )
            return homoscedasticity_metrics.ljung_box_q_test()
        else:
            raise Exception(
                "Ljung-Box Q metrics can only be run for regression models."
            )
    except Exception as ex:
        error_message = (
            f"Error occurred while calculating calculate_ljung_box_q_test: {str(ex)}"
        )
        print(error_message)
        execution_errors_array.append(error_message)
        return {"Homoscedasticity_minLjungBoxQPValue": -99}


def calculate_durbin_watson(dataframe, execution_errors_array):
    """A function to run the Durban Watson test on sample data

    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs) and
        labels (ground truths)
        execution_errors_array (array): Array for collecting execution errors
    Returns:
        (dict): Durbin-Watson test results
    """
    try:
        dashboard_utils.assert_df_not_none_and_not_empty(
            dataframe, "Required comparator"
        )
        autocorrelation_metrics = diagnostics.AutocorrelationMetrics(
            dataframe=dataframe, job_json=JOB
        )
        return autocorrelation_metrics.durbin_watson_test()
    except Exception as ex:
        error_message = (
            f"Error occurred while calculating durban_watson test: {str(ex)}"
        )
        print(error_message)
        execution_errors_array.append(error_message)
        return {"dw_statistic": -99}


def calculate_engle_lagrange_multiplier_test(dataframe, execution_errors_array):
    """A function to run the engle_lagrange_multiplier_test on sample data

    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs),
        labels (ground truths) and numerical_columns (predictors)
        execution_errors_array (array): Array for collecting execution errors
    Returns:
        (dict): Engle's Langrange Multiplier test results
    """
    try:
        if "regression" in MODEL_METHODOLOGY.casefold():
            dashboard_utils.assert_df_not_none_and_not_empty(
                dataframe, "Required comparator"
            )
            homoscedasticity_metrics = diagnostics.HomoscedasticityMetrics(
                dataframe=dataframe, job_json=JOB
            )
            return homoscedasticity_metrics.engle_lagrange_multiplier_test()
        else:
            raise Exception(
                "Engle's Langrange Multiplier metrics can only be run for regression models."
            )
    except Exception as ex:
        error_message = f"Error occurred while calculating engle_lagrange_multiplier test: {str(ex)}"
        print(error_message)
        execution_errors_array.append(error_message)
        return {"engle_lm_p_value": -99}


def calculate_anderson_darling_test(dataframe, execution_errors_array):
    """A function to run the calculate_anderson_darling_test on sample data

    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs) and
        labels (ground truths)
        execution_errors_array (array): Array for collecting execution errors
    Returns:
        (dict): Anderson-Darling test results
    """
    try:
        dashboard_utils.assert_df_not_none_and_not_empty(
            dataframe, "Required comparator"
        )
        normality_metrics = diagnostics.NormalityMetrics(
            dataframe=dataframe, job_json=JOB
        )
        return normality_metrics.anderson_darling_test()
    except Exception as ex:
        error_message = (
            f"Error occurred while calculating anderson_darling test: {str(ex)}"
        )
        print(error_message)
        execution_errors_array.append(error_message)
        return {"ad_p_value": -99}


def calculate_cramer_von_mises_test(dataframe, execution_errors_array):
    """A function to run the cramer_von_mises_test on sample data

    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs) and
        labels (ground truths)
        execution_errors_array (array): Array for collecting execution errors
    Returns:
        (dict): Cramer-von Mises test results
    """
    try:
        dashboard_utils.assert_df_not_none_and_not_empty(
            dataframe, "Required comparator"
        )
        normality_metrics = diagnostics.NormalityMetrics(
            dataframe=dataframe, job_json=JOB
        )
        return normality_metrics.cramer_von_mises_test()
    except Exception as ex:
        error_message = (
            f"Error occurred while calculating cramer_von_mises test: {str(ex)}"
        )
        print(error_message)
        execution_errors_array.append(error_message)
        return {"cvm_p_value": -99}


def calculate_kolmogorov_smirnov_test(dataframe, execution_errors_array):
    """A function to run the kolmogorov_smirnov_test on sample data

    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs) and
        labels (ground truths)
        execution_errors_array (array): Array for collecting execution errors
    Returns:
        (dict): Kolmogorov-Smirnov test results
    """
    try:
        dashboard_utils.assert_df_not_none_and_not_empty(
            dataframe, "Required comparator"
        )
        normality_metrics = diagnostics.NormalityMetrics(
            dataframe=dataframe, job_json=JOB
        )
        return normality_metrics.kolmogorov_smirnov_test()
    except Exception as ex:
        error_message = (
            f"Error occurred while calculating kolmogorov_smirnov test: {str(ex)}"
        )
        print(error_message)
        execution_errors_array.append(error_message)
        return {"ks_p_value": -99}

def check_and_drop_nulls(dataframe):
    """
    A function to check for and drop NULLs in given columns.

    Args:
        dataframe (pandas.DataFrame): Pandas DataFrame of data, most likely scored and labeled.

    Returns:
        pandas.DataFrame: Copy of the dataframe with dropped nulls.
    """
    # Checking for NULLs in columns
    for column in dataframe.columns:
        null_count = dataframe[column].isna().sum()
        if null_count > 0:
            dataframe = dataframe.dropna(subset=[column])

    return dataframe