import json
import traceback
import modelop.schema.infer as infer
import modelop.monitors.performance as performance
import modelop.monitors.drift as drift
import modelop.monitors.stability as stability
import modelop.stats.diagnostics as diagnostics

DEPLOYABLE_MODEL = {}
JOB = {}


# modelop.init
def init(job_json):
    global DEPLOYABLE_MODEL
    global JOB
    
    job = json.loads(job_json['rawJson'])
    DEPLOYABLE_MODEL = job['referenceModel']

    JOB = job_json
    infer.validate_schema(job_json)


# modelop.metrics
def metrics(baseline, comparator) -> dict:
    global DEPLOYABLE_MODEL
    
    result = {}
    
    result.update(
        {
            'modelUseCategory': DEPLOYABLE_MODEL.get('storedModel', {}).get('modelMetaData', {}).get('modelUseCategory', ''),
            'modelOrganization': DEPLOYABLE_MODEL.get('storedModel', {}).get('modelMetaData', {}).get('modelOrganization', ''),
            'modelRisk': DEPLOYABLE_MODEL.get('storedModel', {}).get('modelMetaData', {}).get('modelRisk', ''),
            'modelMethodology': DEPLOYABLE_MODEL.get('storedModel', {}).get('modelMetaData', {}).get('modelMethodology', '')
        }
    )

    result.update(calculate_performance(comparator))

    result.update(calculate_ks_drift(baseline, comparator))
    
    result.update(calculate_stability(baseline, comparator))
    
    result.update(calculate_breusch_pagan(comparator))

    result.update(calculate_linearity_metrics(comparator))

    result.update(calculate_ljung_box_q_test(comparator))

    result.update(calculate_variance_inflation_factor(comparator))

    result.update(calculate_durbin_watson(comparator))

    result.update(calculate_engle_lagrange_multiplier_test(comparator))

    result.update(calculate_anderson_darling_test(comparator))

    result.update(calculate_cramer_von_mises_test(comparator))

    result.update(calculate_kolmogorov_smirnov_test(comparator))
    
    yield result


def calculate_performance(comparator):
    try:
        model_evaluator = performance.ModelEvaluator(dataframe=comparator, job_json=JOB)
        if DEPLOYABLE_MODEL.get('storedModel', {}).get('modelMetaData', {}).get('modelMethodology', '').casefold() == 'regression'.casefold():
            return model_evaluator.evaluate_performance(pre_defined_metrics='regression_metrics')
        else:
            return model_evaluator.evaluate_performance(pre_defined_metrics ='classification_metrics')
    except Exception as ex:
        print('Error occurred calculating performance metrics')
        print(ex)
        print(traceback.format_exc())
        return {}


def calculate_ks_drift(baseline, sample):
    try:
        drift_test = drift.DriftDetector(df_baseline=baseline, df_sample=sample, job_json=JOB)
        return drift_test.calculate_drift(pre_defined_test='Kolmogorov-Smirnov', result_wrapper_key='data_drift')
    except Exception as ex:
        print("Error occurred while calculating drift")
        print(ex)
        print(traceback.format_exc())
        return {}

def calculate_stability(df_baseline, df_comparator):
    try:
        stability_test = stability.StabilityMonitor(
            df_baseline=df_baseline, 
            df_sample=df_comparator,
            job_json=JOB
        )
        return stability_test.compute_stability_indices()
    except Exception as ex:
        print("Error occurred while calculating stability")
        print(ex)
        print(traceback.format_exc())
        return {}


def calculate_breusch_pagan(dataframe):
    """A function to run the Breauch-Pagan test on sample data
    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs),
        labels (ground truths) and numerical_columns (predictors)
    Returns:
        (dict): Breusch-Pagan test results
    """
    try:
        homoscedasticity_metrics = diagnostics.HomoscedasticityMetrics(
            dataframe=dataframe,
            job_json=JOB
        )
        return homoscedasticity_metrics.breusch_pagan_test()
    except Exception as ex:
        print("Error occurred while calculating breusch_pagan")
        print(ex)
        print(traceback.format_exc())
        return {}


def calculate_variance_inflation_factor(dataframe):
    """A function to compute Variance Inflation Factors on sample data
    Args:
        dataframe (pandas.DataFrame): Sample prod data containing numerical_columns (predictors)
    Returns:
        (dict): Pearson Correlation results
    """
    try:
        dataframe=dataframe.astype('float')
        multicollinearity_metrics = diagnostics.MulticollinearityMetrics(
            dataframe=dataframe,
            job_json=JOB
        )
        return  multicollinearity_metrics.variance_inflation_factor()
    except Exception as ex:
        print("Error occurred while calculating variance_inflation_factor")
        print(ex)
        print(traceback.format_exc())
        return {}        


def calculate_linearity_metrics(dataframe):
    """A function to compute Pearson Correlations on sample data
    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs)
        and numerical_columns (predictors)
    Returns:
        (dict): Pearson Correlation results
    """
    try:
        linearity_metrics = diagnostics.LinearityMetrics(
            dataframe=dataframe,
            job_json=JOB
        )
        return linearity_metrics.pearson_correlation()
    except Exception as ex:
        print("Error occurred while calculating calculate_linearity_metrics")
        print(ex)
        print(traceback.format_exc())
        return {}


def calculate_ljung_box_q_test(dataframe):
    """A function to run the Ljung-Box Q test on sample data
    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs),
        labels (ground truths) and numerical_columns (predictors)
    Returns:
        (dict): Ljung-Box Q test results
    """
    try:
        homoscedasticity_metrics = diagnostics.HomoscedasticityMetrics(
            dataframe=dataframe,
            job_json=JOB
        )
        return homoscedasticity_metrics.ljung_box_q_test()
    except Exception as ex:
        print("Error occurred while calculating ljung_box_q_test")
        print(ex)
        print(traceback.format_exc())
        return {}


def calculate_durbin_watson(dataframe):
    """A function to run the Durban Watson test on sample data

    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs) and
        labels (ground truths)
    Returns:
        (dict): Durbin-Watson test results
    """
    try:
        autocorrelation_metrics = diagnostics.AutocorrelationMetrics(
            dataframe=dataframe,
            job_json=JOB
        )
        return autocorrelation_metrics.durbin_watson_test()
    except Exception as ex:
        print("Error occurred while calculating durban_watson test")
        print(ex)
        print(traceback.format_exc())
        return {}


def calculate_engle_lagrange_multiplier_test(dataframe):
    """A function to run the engle_lagrange_multiplier_test on sample data

    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs),
        labels (ground truths) and numerical_columns (predictors)
    Returns:
        (dict): Engle's Langrange Multiplier test results
    """
    try:
        homoscedasticity_metrics = diagnostics.HomoscedasticityMetrics(
            dataframe=dataframe,
            job_json=JOB
        )
        return homoscedasticity_metrics.engle_lagrange_multiplier_test()
    except Exception as ex:
        print("Error occurred while calculating durban_watson test")
        print(ex)
        print(traceback.format_exc())
        return {}


def calculate_anderson_darling_test(dataframe):
    """A function to run the calculate_anderson_darling_test on sample data

    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs) and
        labels (ground truths)
    Returns:
        (dict): Anderson-Darling test results
    """
    try:
        normality_metrics = diagnostics.NormalityMetrics(
            dataframe=dataframe,
            job_json=JOB
        )
        return normality_metrics.anderson_darling_test()
    except Exception as ex:
        print("Error occurred while calculating durban_watson test")
        print(ex)
        print(traceback.format_exc())
        return {}


def calculate_cramer_von_mises_test(dataframe):
    """A function to run the cramer_von_mises_test on sample data

    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs) and
        labels (ground truths)
    Returns:
        (dict): Cramer-von Mises test results
    """
    try:
        normality_metrics = diagnostics.NormalityMetrics(
            dataframe=dataframe,
            job_json=JOB
        )
        return normality_metrics.cramer_von_mises_test()
    except Exception as ex:
        print("Error occurred while calculating durban_watson test")
        print(ex)
        print(traceback.format_exc())
        return {}


def calculate_kolmogorov_smirnov_test(dataframe):
    """A function to run the kolmogorov_smirnov_test on sample data

    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs) and
        labels (ground truths)
    Returns:
        (dict): Kolmogorov-Smirnov test results
    """
    try:
        normality_metrics = diagnostics.NormalityMetrics(
            dataframe=dataframe,
            job_json=JOB
        )
        return normality_metrics.kolmogorov_smirnov_test()
    except Exception as ex:
        print("Error occurred while calculating durban_watson test")
        print(ex)
        print(traceback.format_exc())
        return {}
    
