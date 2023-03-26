from modelop.monitors.assertions import check_input_types
from pathlib import Path
import modelop.schema.infer as infer
import pandas as pd
import logging
import json

logger = logging.getLogger(__name__)

BUCKET_COL = ''
LABEL_COLUMN = None
SCORE_COLUMN = None

# modelop.init
def init(init_param):
    # global BINS
    global BUCKET_COLUMN
    # global POSITIVE_LABEL
    global LABEL_COLUMN
    global SCORE_COLUMN

    job_json = init_param

    # BINS = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850]

    if job_json is not None:
        logger.info(
            "Parameter 'job_json' is present and will be used to extract "
            "'label_column' and 'score_column'."
        )
        ##### Retrieving jobParameters
        try:
            print('Attempting to extract the BUCKET_COLUMN parameter from jobParameters.')
            extracted_job = json.loads(job_json['rawJson'])['jobParameters']
            BUCKET_COLUMN = extracted_job['BUCKET_COLUMN']
            print(f'Extracted BUCKET_COLUMN: {BUCKET_COLUMN}')
        except Exception as e:
            print('Unable to extract the BUCKET_COLUMN from jobParameters.')
            print(e)
        input_schema_definition = infer.extract_input_schema(job_json)
        monitoring_parameters = infer.set_monitoring_parameters(
            schema_json=input_schema_definition, check_schema=True
        )
        LABEL_COLUMN = monitoring_parameters['label_column']
        SCORE_COLUMN = monitoring_parameters['score_column']
    else:
        logger.info(
            "Parameter 'job_json' it not present, attempting to use "
            "'label_column' and 'score_column' instead."
        )
        if LABEL_COLUMN is None:
            missing_args_error = (
                "Parameter 'job_json' is not present,"
                " but 'label_column'. "
                "'label_column' input parameter is"
                " required if 'job_json' is not provided."
            )
            logger.error(missing_args_error)
            raise Exception(missing_args_error)
    check_input_types(
        inputs=[
            {"label_column": LABEL_COLUMN}
        ],
        types=(str),
    )

# modelop.metrics
def metrics(data: pd.DataFrame) -> dict:
    bucketed_data = round(data.groupby(BUCKET_COLUMN).agg(['mean', 'count'])[[LABEL_COLUMN, SCORE_COLUMN]], 4)
    incr = 0
    # for UI output
    dicto = {}
    # for doc output
    listo = []
    for i, row in bucketed_data.iterrows():
        values = {}
        values[f'{BUCKET_COLUMN}_bucket'] = str(i)
        values[f'{LABEL_COLUMN}_mean'] = row[LABEL_COLUMN]['mean']
        values[f'{SCORE_COLUMN}_mean'] = row[SCORE_COLUMN]['mean']
        values[f'{LABEL_COLUMN}_count'] = row[LABEL_COLUMN]['count']
        values[f'{SCORE_COLUMN}_count'] = row[SCORE_COLUMN]['count']
        listo.append(values)
        dicto[incr] = values
        incr += 1
    
    return {'RankOrder' : 
        [{
            'test_name': "Rank Order Break",
            'test_category': "rankorder",
            'test_type': "rankorder",
            'test_id': "rank_order_break",
            'values': dicto
        }],
        'RankOrderArray': listo
    }

def main():
    raw_json = Path('./example_job.json').read_text()
    init_param = {'rawJson': raw_json}
    init(init_param)
    print('initialized parameters from job_json.')
    print(BUCKET_COLUMN)
    print(LABEL_COLUMN)
    print(SCORE_COLUMN)
    data = pd.read_csv('./rob_test.csv')
    print('read data.')
    result = metrics(data)
    print(json.dumps(result, indent=3, sort_keys=True))
    print('done.')
    
if __name__ == '__main__':
	main()