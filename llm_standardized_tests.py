import glob
import json
import re
import statistics
import guardrails as gd
import openai
import pandas as pd
import spacy

from collections import Counter
from pathlib import Path
from flair.data import Sentence
from flair.models import TextClassifier
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI  # import OpenAI model
from langchain.prompts import load_prompt
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from spacy.tokens import Doc

ALPHABETS = "([A-Za-z])"
PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
STARTERS = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
WEBSITES = "[.](com|net|org|io|gov|edu|me)"
DIGITS = "([0-9])"
MULTIPLE_DOTS = r'\.{2,}'
CLASSIFIER = TextClassifier.load('en-sentiment')
SENTIMENT_ANALYSIS_COLUMN = ""
PII_ANALYSIS_COLUMN = ""
WORD_COUNT_COLUMN = ""
PROMPT_FILE_ASSETS = []


def split_into_sentences(text: str) -> [str]:
	"""
	Split the text into sentences.

	If the text contains substrings "<prd>" or "<stop>", they would lead
	to incorrect splitting because they are used as markers for splitting.

	:param text: text to be split into sentences
	:type text: str

	:return: list of sentences
	:rtype: list[str]
	"""
	text = " " + text + "  "
	text = text.replace("\n", " ")
	text = re.sub(PREFIXES, "\\1<prd>",text)
	text = re.sub(WEBSITES, "<prd>\\1",text)
	text = re.sub(DIGITS + "[.]" + DIGITS, "\\1<prd>\\2",text)
	text = re.sub(MULTIPLE_DOTS, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
	if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
	text = re.sub("\s" + ALPHABETS + "[.] ", " \\1<prd> ",text)
	text = re.sub(ACRONYMS+" "+STARTERS, "\\1<stop> \\2",text)
	text = re.sub(ALPHABETS + "[.]" + ALPHABETS + "[.]" + ALPHABETS + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
	text = re.sub(ALPHABETS + "[.]" + ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
	text = re.sub(" "+SUFFIXES+"[.] "+STARTERS, " \\1<stop> \\2",text)
	text = re.sub(" "+SUFFIXES+"[.]", " \\1<prd>",text)
	text = re.sub(" " + ALPHABETS + "[.]", " \\1<prd>",text)
	if "”" in text: text = text.replace(".”", "”.")
	if "\"" in text: text = text.replace(".\"", "\".")
	if "!" in text: text = text.replace("!\"", "\"!")
	if "?" in text: text = text.replace("?\"", "\"?")
	text = text.replace(".", ".<stop>")
	text = text.replace("?", "?<stop>")
	text = text.replace("!", "!<stop>")
	text = text.replace("<prd>", ".")
	sentences = text.split("<stop>")
	sentences = [s.strip() for s in sentences]
	if sentences and not sentences[-1]: sentences = sentences[:-1]
	return sentences


def calculate_sentiment(sentence: str) -> (int, [], int, []):
	negative_confidence_values = []
	positive_confidence_values = []
	negative_count = 0
	positive_count = 0
	sentences = split_into_sentences(sentence)
	for sentence in sentences:
		flair_sentence = Sentence(sentence)
		CLASSIFIER.predict(flair_sentence)
		if flair_sentence.labels[0].value == 'NEGATIVE':
			negative_count += 1
			negative_confidence_values.append(flair_sentence.labels[0].score)
		else:
			positive_count += 1
			positive_confidence_values.append(flair_sentence.labels[0].score)

	return negative_count, negative_confidence_values, positive_count, positive_confidence_values


def process_pii_finding(finding: RecognizerResult, source_document: str) -> dict:
	result = {
		"entity_type" : finding.entity_type,
		"score": finding.score,
		"content": source_document[finding.start:finding.end]
	}
	return result


def count_token_type(document: Doc, token_type: str):
	words = [token.text
			 for token in document
			 if (not token.is_stop and
				 not token.is_punct and
				 token.pos_ == token_type)]
	word_freq = Counter(words)
	most_frequent_words = word_freq.most_common(10)
	return most_frequent_words


def create_word_count_charts(data: pd.DataFrame) -> dict:
	global WORD_COUNT_COLUMN

	results = {
	}
	nlp = spacy.load('en_core_web_lg')
	document = nlp(data[WORD_COUNT_COLUMN].str.cat(sep=' '))
	nouns = count_token_type(document, "NOUN")
	data = []
	categories = []
	for noun in nouns:
		data.append(noun[1])
		categories.append(noun[0])
	results["Most Frequent Nouns"] = {
		"title": "Most Frequent Nouns",
		"x_axis_label": "Noun",
		"y_axis_label": "Occurrences",
		"rotated": False,
		"data": {
			"nouns": data
		},
		"categories": categories
	}
	verbs = count_token_type(document, "VERB")
	data = []
	categories = []
	for verb in verbs:
		data.append(verb[1])
		categories.append(verb[0])
	results["Most Frequent Verbs"] = {
		"title": "Most Frequent Verbs",
		"x_axis_label": "Verb",
		"y_axis_label": "Occurrences",
		"rotated": False,
		"data": {
			"verbs": data
		},
		"categories": categories
	}
	adjectives = count_token_type(document, "ADJ")
	data = []
	categories = []
	for verb in verbs:
		data.append(verb[1])
		categories.append(verb[0])
	results["Most Frequent Adjectives"] = {
		"title": "Most Frequent Adjectives",
		"x_axis_label": "Adjective",
		"y_axis_label": "Occurrences",
		"rotated": False,
		"data": {
			"adjectives": data
		},
		"categories": categories
	}

	return results


def answer_question(question: str):
	llm = OpenAI(temperature=0.7)

	# Chain 1: Generating a rephrased version of the user's question
	prompt_template = load_prompt("rephrase_prompt.json")
	question_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="statement")

	# Chain 2: Generating assumptions made in the statement
	prompt_template = load_prompt("assumptions_prompt.json")
	assumptions_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="assertions")

	# Chain 3: Fact checking the assumptions
	prompt_template = load_prompt("fact_check_prompt.json")
	fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="facts")

	# Final Chain: Generating the final answer to the user's question based on the facts and assumptions
	prompt_template = load_prompt("answer_prompt.json")
	answer_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="answer")
	overall_chain = SequentialChain(
		chains=[question_chain, assumptions_chain, fact_checker_chain, answer_chain],
		input_variables=["question"],
		output_variables=["answer", "assertions", "facts"],
		verbose=True
	)
	result = overall_chain({"question": question})

	print(f"Facts: \n{result['facts']}")
	print(f"\n\nAnswer:{result['answer']}\n\n")

	return {'facts': result['facts'], 'answer': result['answer']}


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
	global SENTIMENT_ANALYSIS_COLUMN
	global PII_ANALYSIS_COLUMN
	global WORD_COUNT_COLUMN

	job = json.loads(init_param["rawJson"])
	SENTIMENT_ANALYSIS_COLUMN = job.get('jobParameters', {}).get("sentimentAnalysisColumn", "")
	PII_ANALYSIS_COLUMN = job.get('jobParameters', {}).get("piiAnalysisColumn", "")
	WORD_COUNT_COLUMN = job.get('jobParameters', {}).get("wordCountColumn", "")
	assets = job.get("referenceModel", {}).get("storedModel", {}).get("modelAssets", [])
	for asset in assets:
		if asset.get("assetRole", "") == 'PROMPT_TEMPLATE':
			output_file = Path('./tmp/' + asset["sourceCodeFilePath"])
			output_file.parent.mkdir(exist_ok=True, parents=True)
			output_file.write_text(asset["sourceCode"])
			asset["sourceCodeFilePath"] = './tmp/' + asset["sourceCodeFilePath"]
			PROMPT_FILE_ASSETS.append(asset)


def validate_prompt_files() -> dict:
	global PROMPT_FILE_ASSETS
	results = {"invalidFiles": False, "prompt_files": []}

	for prompt_file in PROMPT_FILE_ASSETS:
		try:
			prompt = load_prompt(prompt_file.get("sourceCodeFilePath", ""))
			results["prompt_files"].append({"file": prompt_file.get("sourceCodeFilePath", "unknown").replace("./tmp/", ""),
											"valid": True, "input_variables": prompt.input_variables,
											"template": prompt.template})
		except Exception as ve:
			results["prompt_files"].append({"file": prompt_file.get("sourceCodeFilePath", "unknown"),
											"valid": False, "reason": str(ve)})
			if not results["invalidFiles"]:
				results["invalidFiles"] = True

	return results


def perform_nlp_analysis(data: pd.DataFrame)->dict:
	global SENTIMENT_ANALYSIS_COLUMN
	global PII_ANALYSIS_COLUMN

	negative_confidence_values = []
	positive_confidence_values = []
	negative_count = 0
	positive_count = 0
	for index, row in data.iterrows():
		result = calculate_sentiment(row[SENTIMENT_ANALYSIS_COLUMN])
		negative_count += result[0]
		negative_confidence_values = [*negative_confidence_values, *result[1]]
		positive_count += result[2]
		positive_confidence_values = [*positive_confidence_values, *result[3]]

	print(f"Total Negative Percentage: { negative_count / (negative_count + positive_count)}, Median Negative Confidence: {statistics.median(negative_confidence_values)}")
	print(f"Total Positive Percentage: { positive_count / (negative_count + positive_count)}, Median Positive Confidence: {statistics.median(positive_confidence_values)}")

	results = {
		'sentiment_negative_percentage': negative_count / (negative_count + positive_count),
		'sentiment_negative_median_confidence': statistics.median(negative_confidence_values),
		'sentiment_positive_percentage': positive_count / (negative_count + positive_count),
		'sentiment_positive_median_confidence': statistics.median(positive_confidence_values),
		'sentiment_analysis': {
			'title': 'Sentiment Analysis of Dataset',
			'x_axis_label': 'Sentiment',
			'y_axis_label': 'Percentage',
			'rotated': False,
			'data': {
				'sentiment': [negative_count / (negative_count + positive_count), positive_count / (negative_count + positive_count)],
				'confidence': [statistics.median(negative_confidence_values), statistics.median(positive_confidence_values)]
			},
			'categories': ['Negative Sentiment', 'Positive Sentiment']
		},
		"PII Findings": []
	}

	analyzer = AnalyzerEngine()
	document = data[PII_ANALYSIS_COLUMN].str.cat(sep=' ')
	analyzer_findings = analyzer.analyze(text=document, language='en')
	for finding in analyzer_findings:
		results["PII Findings"].append(process_pii_finding(finding, document))

	results.update(create_word_count_charts(data))

	return results


def calculate_facts(prompts: pd.DataFrame)->dict:
	results = {"Questions, Facts, and Answers": []}
	for index, row in prompts.iterrows():
		llm = OpenAI(temperature=0.7)
		questions = llm.predict(row['prompt'])
		for question in iter(questions.splitlines()):
			if question.strip():
				result = answer_question({'question': question})
				results["Questions, Facts, and Answers"].append(
					{"question": question, "answer": result["answer"], "facts": result["facts"]})
	return results

def check_statement_accuracy(question: str, answer: str, fact: str) -> dict:
	llm = OpenAI(temperature=0.0)

	prompt_template = load_prompt("verify_statement_accuracy_prompt.json")
	accuracy_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="accuracy")

	prompt_template = load_prompt("break_down_score_prompt.json")
	score_breakdown_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="score_breakdown")

	prompt_template = load_prompt("revalidate_score_prompt.json")
	final_score = LLMChain(llm=llm, prompt=prompt_template, output_key="final_score")

	overall_chain = SequentialChain(chains=[accuracy_chain, score_breakdown_chain, final_score],
									input_variables=["question", "factual_answer", "answer"],
									output_variables=["accuracy", "score_breakdown","final_score"],
									verbose=True)
	result = overall_chain({"question": question, "answer": answer, "factual_answer": fact})
	result["final_score"] = result["final_score"].replace("\n", "")
	result["accuracy"] = result["accuracy"].replace("\n", "")
	return result

def calculate_accuracy_of_responses(responses: pd.DataFrame)-> dict:
	result = {"Statement Accuracy" : []}
	for index, row in responses.iterrows():
		result["Statement Accuracy"].append(check_statement_accuracy(row['user_input'], row['response'], row['factual_response']))

	return result

def perform_rails_checks(rails: pd.DataFrame)-> dict:
	result = {"Rails Compliance": []}
	failed_rails = 0
	rail_files = glob.glob('./*.rail')
	guard_rails = []
	for rail_file in rail_files:
		guard_rails.append(gd.Guard.from_rail(rail_file))

	for index, row in rails.iterrows():
		validated_response = {"answer": row["answer"]}
		for guard_rail in guard_rails:
			raw_llm_response, validated_response = guard_rail(openai.Completion.create,
														  prompt_params=validated_response,
														  engine="text-davinci-003",
														  max_tokens=2048,
														  temperature=0)
		passes = True
		if row["answer"].strip() == validated_response["answer"].strip():
			if row["should_be_filtered"]:
				passes = False
				failed_rails += 1
		elif not row["should_be_filtered"]:
			passes = False
			failed_rails += 1

		result["Rails Compliance"].append({"raw_answer": row["answer"], "validated_repsonse": validated_response["answer"], "passes": passes})

	result["num_failed_rails_compliance"] = failed_rails
	return result

# modelop.metrics
def metrics(data: pd.DataFrame, prompts: pd.DataFrame, rails: pd.DataFrame):
	results = validate_prompt_files()
	results.update(perform_nlp_analysis(data))
	results.update(calculate_facts(prompts))
	results.update(calculate_accuracy_of_responses(data))
	results.update(perform_rails_checks(rails))

	yield results


def main():
	raw_json = Path('example_job.json').read_text()
	init_param = {'rawJson': raw_json}
	init(init_param)

	dataset = pd.read_csv('./test_data/example_responses.csv', quotechar='"', header=0)
	prompts = pd.read_csv('./test_data/questions_for_testing.csv', quotechar='"', header=0)
	rails = pd.read_csv('./test_data/guardrail_questions.csv', quotechar='"', header=0)

	print(json.dumps(next(metrics(dataset, prompts, rails)), indent=2))


if __name__ == '__main__':
	main()
