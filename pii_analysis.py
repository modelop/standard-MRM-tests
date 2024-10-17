import pandas as pd

from presidio_analyzer import AnalyzerEngine, RecognizerResult


def process_pii_finding(finding: RecognizerResult, source_document: str) -> dict:
    result = {
        "entity_type": finding.entity_type,
        "score": finding.score,
        "content": source_document[finding.start:finding.end]
    }
    return result


def examine_for_pii(score: pd.DataFrame, minimum_threshold: float = 0.5):
    results = {"PII Findings": []}

    pii_analyzer = AnalyzerEngine()
    all_text = score.astype('string').str.cat(sep=' ')
    analyzer_findings = pii_analyzer.analyze(text=all_text, language='en', score_threshold=minimum_threshold)
    num_pii_violations = 0
    for finding in analyzer_findings:
        pii_result = process_pii_finding(finding, all_text)
        results["PII Findings"].append(pii_result)
        if pii_result["score"] >= minimum_threshold:
            num_pii_violations += 1
    results["num_PII_violations"] = num_pii_violations

    return results
