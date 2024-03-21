# LLM Standardized Tests

This is a collection of tests that can be used on natural language oriented models that provides
for establishing baselines, and later detecting drift.  These same tests can also be utilized to
examine the implementation against the use case and provide insights as to how they are performing.

## Sentiment Analysis

Performs sentiment analysis of natural language statements utlizing the flair library.  Both the negative
and positive sentiment will be reported along with a confidence level for each category.  This will be
reported in both a flat value score, along with a bar chart of both the sentiment and the confidence.

### Required Columns in Schema

<b>Score Column</b> - Must be set in the schema to the column on which sentiment analysis should be
performed.  This should be of the type string, and be the statements that will be analyzed for sentiment.
This is typically a sentence or statement of some kind

## PII Analysis

Performs detection of PII information in the scoring column to detect any potential disclosure of PII
information.  It uses the Presidio open source library to provide this detection.  This library not only
looks for the patterns of PII, but also takes context into account.  It will provide the detected category
of possible PII along with a confidence score in that detection.  Set a job parameter of PII_THRESHOLD to any
float value between 0.0 and 1.0 to change the default confidence level of 0.5

### Required Columns in Schema

<b>Score Column</b> - Must be set in the schema to the column on which PII detection should be performed.
This column should be of type string.

## Word Count by Type

Performs a grammatical analysis of the provided statements to determine parts of speech, then returns the top
ten words of each grammer type (Nouns, Verbs, Adjectives) in bar chart form.  This allows for the establishing
of a baseline of the words being emitted for drift detection at later points in time, as well as for analysis of the
words being emitted against the use case.  Spacy is used for grammatical anlysis.

### Required Columns in Schema

<b>Score Column</b> - Must be set in the schema to the column on which word counts should be performed.  The type of
this columns should be a string, and typically it is a sentence of some kind so parts of speech can be determined.

## Semantic Textual Similarity

Performs a similarity analysis utilizing SBERT and cosine-similarities to determine how semantically similar to individual
statements are.  This can be used for comparing a response given compared to a human reviewed response, or a factual
statement from a document, as examples.

### Required Columns in Schema
<b>Score Column</b> - Must be set to the response or statement given by the model.
<b>Label Column</b> - Must be set to the reviewed response, or a factual statement, or any text to compare against.
