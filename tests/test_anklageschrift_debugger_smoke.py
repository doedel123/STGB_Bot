import os
import unittest
from unittest.mock import patch

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("MISTRAL_API_KEY", "test-key")
os.environ.setdefault("RAGIE_API_KEY", "test-key")

from agent.nodes.fact_extraction import fact_extraction_node
from agent.nodes.allegation_validation import allegation_validation_node


class AnklageschriftDebuggerSmokeTest(unittest.TestCase):
    def test_detects_contradiction_and_unbelegte_behauptung(self):
        sample_text = (
            "Am 10.01.2025 lieferte X dem Y die Ware in Berlin. "
            "Am 10.01.2025 wurde die Ware jedoch nicht geliefert. "
            "Die Staatsanwaltschaft behauptet, X habe gewerbsmaessig und planmaessig gehandelt. "
            "Dies ergibt sich offensichtlich aus den Umstaenden."
        )

        base_state = {
            "raw_text": sample_text,
            "pdf_content": sample_text,
            "document_structure": {},
        }

        with patch("agent.nodes.fact_extraction.llm.invoke", side_effect=RuntimeError("offline")):
            extracted = fact_extraction_node(base_state)

        self.assertTrue(extracted.get("facts"), "Es wurden keine Fakten extrahiert.")
        self.assertTrue(extracted.get("allegations"), "Es wurden keine Behauptungen extrahiert.")

        validated = allegation_validation_node(
            {
                "facts": extracted["facts"],
                "allegations": extracted["allegations"],
                "citations": extracted.get("citations", []),
            }
        )

        contradictions = validated.get("contradictions", [])
        self.assertTrue(contradictions, "Es wurde kein Widerspruch erkannt.")

        strengths = [
            entry.get("support_strength")
            for entry in validated.get("validation_report", {}).values()
        ]
        self.assertTrue(
            any(strength in {"none", "weak"} for strength in strengths),
            "Keine unbelegte oder schwach belegte Behauptung erkannt.",
        )


if __name__ == "__main__":
    unittest.main()
