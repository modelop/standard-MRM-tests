# Model Card for <<modelop.storedModel.modelMetaData.name>>

<!-- Provide a quick summary of what the model is/does. -->
<<modelop.storedModel.modelMetaData.description>>

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Model Use Case:** <<modelop.deployableModel.associatedModels.[associationRole=MODEL_USE_CASE].associatedModel.storedModel.modelMetaData.name>>

- **Developed by:** <<modelop.storedModel.createdBy>>
- **Model type:** <<modelop.storedModel.modelMetaData.modelMethodology>> - <<modelop.storedModel.modelMetaData.type>>
- **Model Documentation:** <a href="<<modelop.storedModel.modelAssets.[assetRole=MODEL_DOCUMENTATION].fileUrl>>"><<modelop.storedModel.modelAssets.[assetRole=MODEL_DOCUMENTATION].filename>></a>

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** <<modelop.storedModel.modelMetaData.repositoryInfo.repositoryRemote>> **branch:** <<modelop.storedModel.modelMetaData.repositoryInfo.repositoryBranch>>

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

[More Information Needed]

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

|<<modelop.modelTestResult.testResults.(Gender Response Similarity)>>|

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

```
<<modelop.storedModel.modelAssets.[assetRole=MODEL_EXAMPLE_SOURCE].sourceCode>>
```
## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->
[Need More Information]

#### Training Hyperparameters

- **Training regime:** [More Information Needed] <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->
### Dataset Card for <<modelop.storedModel.modelAssets.[assetRole=TEST_DATA].filename>>

#### Dataset Sources

- **Repository:** <<modelop.storedModel.modelAssets.[assetRole=TEST_DATA].fileUrl>>

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

- **Sentiment Analysisu of Responses:**<br>

<<modelopgraph.bargraph.sentiment_analysis>>

- **Comparison of Response to a Known Factual Response (Accuracy):**<br>

|<<modelop.modelTestResult.testResults.(Statement Accuracy)>>|

- **SBert Similarity of Response to a Reviewed Response

|<<modelop.modelTestResult.testResults.(Statement Similarity)>>|

### Results

[More Information Needed]

#### Summary

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Model Card Contact

<<modelop.storedModel.createdBy>>


