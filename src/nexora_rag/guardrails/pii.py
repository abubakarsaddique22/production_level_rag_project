from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE", "PK_CNIC", "PK_PHONE"]

# Pakistani formats jo Presidio ke default recognizers nahi pakarte
cnic = PatternRecognizer(
    supported_entity="PK_CNIC",
    patterns=[Pattern("cnic", r"\b\d{5}-\d{7}-\d\b", 0.85)],
)
pk_phone = PatternRecognizer(
    supported_entity="PK_PHONE",
    patterns=[Pattern("pk_phone", r"(?:\+92|0092|0)3\d{2}[- ]?\d{7}\b", 0.85)],
)

nlp_engine = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}).create_engine()

analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
analyzer.registry.add_recognizer(cnic)
analyzer.registry.add_recognizer(pk_phone)
anonymizer = AnonymizerEngine()


def mask_pii(text: str) -> str:
    """PII ko <ENTITY_TYPE> se badal deta hai, baqi text waisa hi rehta hai."""
    if not text:
        return text
    found = analyzer.analyze(text=text, entities=ENTITIES, language="en")
    if not found:
        return text
    # presidio-analyzer and presidio-anonymizer each define their own RecognizerResult class
    return anonymizer.anonymize(text=text, analyzer_results=found).text  # type: ignore[arg-type]