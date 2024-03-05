import glob
import json
import logging
import os
import torch

import pandas as pd

from pathlib import Path
from langchain.llms import LlamaCpp, OpenAI

from modelop_tests.prompt_tests import (calculate_bias, validate_prompt_files, validate_rails_files,
                                        calculate_accuracy_of_responses)
from modelop_tests.nlp_tests import calculate_sentiment, examine_for_pii, perform_word_count, calculate_sbert_similarity

LOG = logging.getLogger("modelop_test_wrappers.llm_standardized_tests")
PROMPT_FILE_ASSETS = []
RAIL_FILE_ASSETS = []
QUESTION_COLUMN: str = ""
ANSWER_COLUMN: str = ""
PII_THRESHOLD: float = 0.5
VERIFIED_ANSWER_COLUMN: str = ""
LLM = None


#
# This is the model initialization function.  This function will be called once when the model is initially loaded.  At
# this time we can read information about the job that is resulting in this model being loaded, including the full
# job in the initialization parameter
#
# Note that in a monitor, the actual model on which the monitor is being run will be in the referenceModel parameter,
# and the monitor code itself will be in the model parameter.
#

# modelop.init
def init(init_param):
    global PROMPT_FILE_ASSETS
    global ANSWER_COLUMN
    global QUESTION_COLUMN
    global VERIFIED_ANSWER_COLUMN
    global PII_THRESHOLD
    global RAIL_FILE_ASSETS
    global LLM

    print("CUDA Available - " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("CUDA Devices Available: " + str(torch.cuda.device_count()))
        print("CUDA Device Attached: " + torch.cuda.get_device_name(0))

    job = json.loads(init_param["rawJson"])
    # Extract input schema
    try:
        input_schemas = job["referenceModel"]["storedModel"]["modelMetaData"]["inputSchema"]
    except Exception:
        LOG.warning("No input schema found on a reference storedModel. Using base storedModel for input schema")
        input_schemas = job["model"]["storedModel"]["modelMetaData"]["inputSchema"]
    if len(input_schemas) != 1:
        LOG.error("Found more than one input schema in model definition, aborting execution.")
        raise ValueError(f"Expected only 1 input schema definition, but found {len(input_schemas)}")
    schema_df = pd.DataFrame(input_schemas[0]["schemaDefinition"]["fields"]).set_index("name")
    ANSWER_COLUMN = schema_df.loc[schema_df['role'] == 'score'].index.values[0]
    QUESTION_COLUMN = schema_df.loc[schema_df['role'] == 'predictor'].index.values[0]
    VERIFIED_ANSWER_COLUMN = schema_df.loc[schema_df['role'] == 'label'].index.values[0]
    if not ANSWER_COLUMN or not QUESTION_COLUMN or not VERIFIED_ANSWER_COLUMN:
        LOG.warning("One or more column types were not available.  Some calculations will not be possible")
    LOG.info(f"Using answer column {ANSWER_COLUMN}, question column {QUESTION_COLUMN}, verified answer column {VERIFIED_ANSWER_COLUMN}")
    PII_THRESHOLD = job.get("jobParameters", {}).get("PII_THRESHOLD", 0.5)
    LOG.info(f"Using PII minimum threshold of {PII_THRESHOLD}")
    assets = job.get("referenceModel", {}).get("storedModel", {}).get("modelAssets", [])
    for asset in assets:
        if asset.get("assetRole", "") == 'PROMPT_TEMPLATE':
            output_file = Path('./tmp/' + asset["sourceCodeFilePath"])
            output_file.parent.mkdir(exist_ok=True, parents=True)
            output_file.write_text(asset["sourceCode"])
            asset["sourceCodeFilePath"] = './tmp/' + asset["sourceCodeFilePath"]
            PROMPT_FILE_ASSETS.append(asset)
    for modelAssociation in job.get("referenceModel", {}).get("storedModel", {}).get("associatedModels", []):
        for asset in modelAssociation.get("associatedModel", {}).get("modelAssets", modelAssociation.get("storedModel", {}).get("modelAssets", [])):
            if asset.get("assetRole", "") == 'PROMPT_TEMPLATE':
                output_file = Path('./tmp/' + asset["sourceCodeFilePath"])
                output_file.parent.mkdir(exist_ok=True, parents=True)
                output_file.write_text(asset["sourceCode"])
                asset["sourceCodeFilePath"] = './tmp/' + asset["sourceCodeFilePath"]
                PROMPT_FILE_ASSETS.append(asset)

    rail_assets = job.get("additionalAssets", [])
    for asset in rail_assets:
        if asset.get("assetRole", "") == "RAIL_FILE":
            RAIL_FILE_ASSETS.append(asset)

    llm_files = glob.glob('./*.gguf')
    if llm_files:
        if len(llm_files) > 1:
            raise FileNotFoundError("More than one GGUF file was present, so we can not know which one to utilize")
        LLM = LlamaCpp(model_path=os.path.abspath(llm_files[0]),
                       model_kwargs={"max_length": 10000}, n_ctx=2048,
                       n_gpu_layers=80 if torch.cuda.is_available() else 0, verbose=True, top_p=1, temperature=0.2)
    else:
        config_files = glob.glob('./config.json')
        if config_files:
            config_json = json.loads(Path(config_files[0]).read_text())
            LLM = OpenAI(**config_json)


# modelop.metrics
def metrics(questions_and_responses: pd.DataFrame, rails_tests: pd.DataFrame):
    global PROMPT_FILE_ASSETS
    global ANSWER_COLUMN
    global QUESTION_COLUMN
    global VERIFIED_ANSWER_COLUMN
    global PII_THRESHOLD
    global LLM

    results = {}

    results.update(validate_prompt_files(PROMPT_FILE_ASSETS))

    # Calculate Potential Subtle Bias
    bias_profile_prompt = (
        next((asset for asset in PROMPT_FILE_ASSETS if asset.get("name", "") == "bias_profile_prompt.json"), None))
    bias_questions_prompt = (
        next((asset for asset in PROMPT_FILE_ASSETS if asset.get("name", "") == "bias_questions_prompt.json"), None))
    if not bias_profile_prompt or not bias_questions_prompt:
        LOG.error("""
        You must have both a file named bias_profile_prompt.json and bias_questions_prompt.json on your target model
        that provides the profile of the individual for the questions as well as a prompt that generates appropriate
        questions for your use case""")
        raise ValueError("""
        You must have both a file named bias_profile_prompt.json and bias_questions_prompt.json on your target model
        that provides the profile of the individual for the questions as well as a prompt that generates appropriate
        questions for your use case""")
    results.update(calculate_bias(
        Path(bias_profile_prompt.get("sourceCodeFilePath", None)).absolute(),
        Path(bias_questions_prompt.get("sourceCodeFilePath", None)).absolute(),
        llm=LLM))

    if ANSWER_COLUMN:
        results.update(calculate_sentiment(questions_and_responses[ANSWER_COLUMN].tolist()))
    else:
        LOG.warning("Skipped calculating sentiment as a column with role of score was not found in input schema")

    if ANSWER_COLUMN:
        results.update(examine_for_pii(questions_and_responses[ANSWER_COLUMN], minimum_threshold=PII_THRESHOLD))
    else:
        LOG.warning("Skipped PII analysis as a column with role of score was not found in input schema")

    if ANSWER_COLUMN:
        results.update(perform_word_count(questions_and_responses[ANSWER_COLUMN]))
    else:
        LOG.warning("Skipped word count as a column with role of score was not found in input schema")

    if ANSWER_COLUMN and VERIFIED_ANSWER_COLUMN:
        results.update(calculate_sbert_similarity(questions_and_responses[[ANSWER_COLUMN, VERIFIED_ANSWER_COLUMN]]))
    else:
        LOG.warning(
            "Skipped sbert similarity as a column with role of score and a column with role of label is required")

    if ANSWER_COLUMN and VERIFIED_ANSWER_COLUMN and QUESTION_COLUMN:
        results.update(calculate_accuracy_of_responses(
            questions_and_responses[[QUESTION_COLUMN, ANSWER_COLUMN, VERIFIED_ANSWER_COLUMN]], llm=LLM))
    else:
        LOG.warning(
            "Skipped accuracy assessment as you must specify the predictor column, score column, and label column in "
            "your input schema")

    results.update(validate_rails_files(rails_tests, RAIL_FILE_ASSETS,llm=LLM))

    yield results


def main():
    job_json = json.loads(Path('test_data/example_job.json').read_text())

    init_param = {'rawJson': json.dumps(job_json)}
    init(init_param)

    questions_and_responses = pd.read_csv('test_data/example_responses.csv', quotechar='"', header=0)
    rails_tests = pd.read_csv('test_data/example_rails_test.csv', quotechar='"', header=0)

    print(json.dumps(next(metrics(questions_and_responses, rails_tests)), indent=2))


if __name__ == '__main__':
    main()
