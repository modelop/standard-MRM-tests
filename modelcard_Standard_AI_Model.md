# Model Card for <<modelop.storedModel.modelMetaData.name>>

<!-- Provide a quick summary of what the model is/does. -->
<<modelop.storedModel.modelMetaData.description>>

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** <<modelop.storedModel.createdBy>>
- **Model type:** <<modelop.storedModel.modelMetaData.modelMethodology>> - <<modelop.storedModel.modelMetaData.type>>
- **Model Documentation:** <a href="<<modelop.storedModel.modelAssets.[assetRole=MODEL_DOCUMENTATION].fileUrl>>"><<modelop.storedModel.modelAssets.[assetRole=MODEL_DOCUMENTATION].filename>></a>

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** <<modelop.storedModel.modelMetaData.repositoryInfo.repositoryRemote>> **branch:** <<modelop.storedModel.modelMetaData.repositoryInfo.repositoryBranch>>

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section describes the Scope / Intended Usages. -->

Scope: <<modelop.storedModel.modelMetaData.custom.Overview.Scope>>

### Out-of-Scope Use

<!-- This section describes the Limitations and addresses misuse, malicious use, and uses that the model will not work well for. -->

Limitations: <<modelop.storedModel.modelMetaData.custom.Overview.Limitations>>


## How to Get Started with the Model

Use the code below to get started with the model.

- **Primary Source Name:** <<modelop.storedModel.modelAssets.[primaryModelSource=true].name>>
- **Primary Source Git URL:** <<modelop.storedModel.modelAssets.[primaryModelSource=true].repositoryInfo.repositoryRemote>>
- **Primary Source Git Branch:** <<modelop.storedModel.modelAssets.[primaryModelSource=true].repositoryInfo.repositoryBranch>>


## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

- **Training Data Name:** <<modelop.storedModel.modelAssets.[assetRole=TRAINING_DATA].filename>>

- **Training Data URL:** <<modelop.storedModel.modelAssets.[assetRole=TRAINING_DATA].fileUrl>>


### Training Hyperparameters

- **Training regime:** [More Information Needed] <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->
**Dataset Card for**: <<modelop.storedModel.modelAssets.[assetRole=TEST_DATA].filename>>
- **Repository:** <<modelop.storedModel.modelAssets.[assetRole=TEST_DATA].fileUrl>>

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

- **Performance Metrics**<br>

|<<modelop.modelTestResult.testResults.(performance)[0].values>>|

- **Stability Metrics**<br>
<<modelopgraph.stability.*>>


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


