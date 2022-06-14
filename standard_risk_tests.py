import json
import traceback
import modelop.schema.infer as infer
import modelop.monitors.performance as performance
import modelop.monitors.drift as drift
import modelop.monitors.stability as stability
import modelop.stats.diagnostics as diagnostics
#import numpy as np

DEPLOPYABLE_MODEL = {}
INPUT_SCHEMA = {}

def init(init_param):
    global DEPLOYABLE_MODEL
    global INPUT_SCHEMA
    
    job = json.loads(init_param['rawJson'])
    DEPLOYABLE_MODEL = job['referenceModel']
    INPUT_SCHEMA = infer.extract_input_schema(init_param)
    
def metrics(baseline, comparator) -> dict:
    global DEPLOYABLE_MODEL
    global INPUT_SCHEMA
    
    result = {}
    
    result.update(
        {'modelUseCategory': DEPLOYABLE_MODEL.get('storedModel', {}).get('modelMetaData', {}).get('modelUseCategory', ''),
        'modelOrganization': DEPLOYABLE_MODEL.get('storedModel', {}).get('modelMetaData', {}).get('modelOrganization', ''),
        'modelRisk': DEPLOYABLE_MODEL.get('storedModel', {}).get('modelMetaData', {}).get('modelRisk', ''),
        'modelMethodology': DEPLOYABLE_MODEL.get('storedModel', {}).get('modelMetaData', {}).get('modelMethodology', '')}
                 )
    result.update(calculate_performance(comparator, INPUT_SCHEMA))
    
    result.update(calculate_ks_drift(baseline, comparator))
    
    result.update(calculate_stability(baseline, comparator, INPUT_SCHEMA))
    
    result.update(calculate_breusch_pagan(comparator, INPUT_SCHEMA))
    
    result.update(calculate_linearity_metrics(comparator, INPUT_SCHEMA))
    
    result.update(calculate_ljung_box_q_test(comparator, INPUT_SCHEMA))
    
    result.update(calculate_variance_inflation_factor(comparator, INPUT_SCHEMA))
    
    
    yield result
    
def calculate_performance(comparator, input_schema) -> dict:
    try:
        monitoring_parameters = infer.set_monitoring_parameters(input_schema, check_schema=True)
        performance_test = performance.ModelEvaluator(dataframe=comparator,
                                                        score_column=monitoring_parameters.get('score_column', None),
                                                        label_column=monitoring_parameters.get('label_column', None))
        if DEPLOYABLE_MODEL.get('storedModel', {}).get('modelMetaData', {}).get('modelMethodology', '').casefold() == 'regression'.casefold():
            performance_result = performance_test.evaluate_performance(pre_defined_metrics='regression_metrics')
            return {
                'performance': [ performance_result ],
                'mae': performance_result.get('values', {}).get('mae', None),
                'r2_score': performance_result.get('values', {}).get('r2_score', None),
                'rmse': performance_result.get('values', {}).get('rmse', None)
            }
        else:
            performance_result = performance_test.evaluate_performance(pre_defined_metrics='classification_metrics')
            return {
                'performance': [ performance_result ],
                'accuracy': performance_result.get('values', {}).get('accuracy', None),
                'precision': performance_result.get('values', {}).get('precision', None),
                'recall': performance_result.get('values', {}).get('recall', None),
                'f1_score': performance_result.get('values', {}).get('f1_score', None),
                'auc': performance_result.get('values', {}).get('auc', None),
            }
    except Exception as ex:
        print('Error occurred calculating performance metrics')
        print(ex)
        print(traceback.format_exc())
        return {}
    
def calculate_ks_drift(baseline, sample) -> dict:
    try:
        drift_test = drift.DriftDetector(df_baseline=baseline, df_sample=sample)
        drift_result = drift_test.calculate_drift(pre_defined_test='Kolmogorov-Smirnov')
        return {"data_drift": [ drift_result ]}
    except:
        print("Error occurred while calculating drift")
        print(ex)
        print(traceback.format_exc())
        return {}
                   
def calculate_stability(df_baseline, df_comparator, input_schema) -> dict:
    try:
        monitoring_parameters = infer.set_monitoring_parameters(input_schema, check_schema=True)
       
    
        score_column = monitoring_parameters.get('score_column', None)
        predictors = monitoring_parameters.get('predictors', None)
    
        # Initialize StabilityMonitor
        stability_test = stability.StabilityMonitor(
            df_baseline=df_baseline, 
            df_sample=df_comparator,
            predictors=predictors,
            feature_dataclass=monitoring_parameters.get('feature_dataclass', None),
            special_values=monitoring_parameters.get('special_values', None),
            score_column=score_column,
            weight_column=monitoring_parameters.get('weight_column', None))
    
        # Set default n_groups for each predictor and score
        n_groups = {}
        for feature in monitoring_parameters.get('numerical_columns', None) + [monitoring_parameters.get('score_column', None)]:
        # If a feature has more than 1 unique value, set n_groups to 2; else set to 1
            feature_has_distinct_values = int(
                (min(df_baseline[feature]) != max(df_baseline[feature]))
            )
            n_groups[feature] = 1 + feature_has_distinct_values

        # Compute stability metrics
        stability_metrics = stability_test.compute_stability_indices(
            n_groups=n_groups, group_cuts={}
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
        # Add Vanilla output
        result["stability"] = [stability_metrics]
        
        return result
    except:
        print("Error occurred while calculating stability")
        print(ex)
        print(traceback.format_exc())
        return {}

def calculate_breusch_pagan(dataframe, input_schema) -> dict:
    """A function to run the Breauch-Pagan test on sample data
    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs),
        labels (ground truths) and numerical_columns (predictors)
    Returns:
        (dict): Breusch-Pagan test results
    """
    try:
        monitoring_parameters = infer.set_monitoring_parameters(input_schema, check_schema=True)

        # Initialize metrics class
        homoscedasticity_metrics = diagnostics.HomoscedasticityMetrics(
            dataframe=dataframe,
            score_column=monitoring_parameters.get('score_column', None),
            label_column=monitoring_parameters.get('label_column', None),
            numerical_predictors=monitoring_parameters.get('numerical_columns')
        )

        # Run test
        test_results = homoscedasticity_metrics.breusch_pagan_test()

        result = {
            # Top-level metrics
            "breusch_pagan_lm_statistic": test_results["values"]["lm_statistic"],
            "breusch_pagan_lm_p_value": test_results["values"]["lm_p_value"],
            "breusch_pagan_f_statistic": test_results["values"]["f_statistic"],
            "breusch_pagan_f_p_value": test_results["values"]["f_p_value"],
            # Vanilla HomoscedasticityMetrics output
            "homoscedasticity_breusch_pagan": [test_results],
        }
        return result
    except:
        print("Error occurred while calculating breusch_pagan")
        print(ex)
        print(traceback.format_exc())
        return {}
    
def calculate_variance_inflation_factor(dataframe, input_schema) -> dict:
    """A function to compute Variance Inflation Factors on sample data
    Args:
        dataframe (pandas.DataFrame): Sample prod data containing numerical_columns (predictors)
    Returns:
        (dict): Pearson Correlation results
    """
    try:
        monitoring_parameters = infer.set_monitoring_parameters(input_schema, check_schema=True)

        dataframe=dataframe.astype('float')
        multicollinearity_metrics = diagnostics.MulticollinearityMetrics(
            dataframe=dataframe,
            numerical_predictors= monitoring_parameters.get('numerical_columns')
        )

        # Run test
        test_results = multicollinearity_metrics.variance_inflation_factor()

        result = {
            # Vanilla MulticollinearityMetrics output
            "multicollinearity": [test_results],
        }
        return result
    except Exception as ex:
        print("Error occurred while calculating variance_inflation_factor")
        print(ex)
        print(traceback.format_exc())
        return {}        

def calculate_linearity_metrics(dataframe, input_schema) -> dict:
    """A function to compute Pearson Correlations on sample data
    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs)
        and numerical_columns (predictors)
    Returns:
        (dict): Pearson Correlation results
    """
    try:
        monitoring_parameters = infer.set_monitoring_parameters(input_schema, check_schema=True)

        # Initialize metrics class
        linearity_metrics = diagnostics.LinearityMetrics(
            dataframe=dataframe,
            label_column=monitoring_parameters.get('label_column'),
            numerical_predictors=monitoring_parameters.get('numerical_columns')
        )

        # Run test
        test_results = linearity_metrics.pearson_correlation()

        result = {
            # Vanilla LinearityMetrics output
            "linearity": [test_results],
        }
        return result
    except:
        print("Error occurred while calculating calculate_linearity_metrics")
        print(ex)
        print(traceback.format_exc())
        return {}
    
def calculate_ljung_box_q_test(dataframe, input_schema) -> dict:
    """A function to run the Ljung-Box Q test on sample data
    Args:
        dataframe (pandas.DataFrame): Sample prod data containing scores (model outputs),
        labels (ground truths) and numerical_columns (predictors)
    Returns:
        (dict): Ljung-Box Q test results
    """
    try:
        monitoring_parameters = infer.set_monitoring_parameters(input_schema, check_schema=True)      
        # Initialize metrics class
        homoscedasticity_metrics = diagnostics.HomoscedasticityMetrics(
            dataframe=dataframe,
            label_column=monitoring_parameters.get('label_column'),
            score_column=monitoring_parameters.get('score_column'),
            numerical_predictors=monitoring_parameters.get('numerical_columns')
        )

        # Run test
        test_results = homoscedasticity_metrics.ljung_box_q_test()

        result = {
            # Vanilla HomoscedasticityMetrics output
            "homoscedasticity_ljung_box": [test_results],
        }
        return result
    except:
        print("Error occurred while calculating ljung_box_q_test")
        print(ex)
        print(traceback.format_exc())
        return {}
