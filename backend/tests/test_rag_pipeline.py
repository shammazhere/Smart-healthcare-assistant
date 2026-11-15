# ============================================================
# test_rag_pipeline.py — Unit tests for RAG pipeline & patient records
# ============================================================
import os
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

from rag_engine import (
    get_store,
    build_context,
    build_context_with_sources,
    tokenize,
    expand_query,
    Chunk,
)
from db import load_all_records, list_patient_ids, load_patient_record


class TestRAGPipeline(unittest.TestCase):

    def test_database_loads_patients(self):
        records = load_all_records()
        self.assertGreater(len(records), 0)
        self.assertIn("patient_001", records)

        patient = load_patient_record("patient_001")
        self.assertIsNotNone(patient)
        self.assertEqual(patient["profile"]["name"], "Ramesh Kumar")
        self.assertEqual(patient["profile"]["age"], 68)

    def test_rag_store_initialization(self):
        store = get_store()
        self.assertTrue(store.has_patient("patient_001"))
        patients = store.patient_list()
        self.assertGreater(len(patients), 0)

        overview = store.overview("patient_001")
        self.assertIsNotNone(overview)
        self.assertEqual(overview["name"], "Ramesh Kumar")
        self.assertIn("Type 2 Diabetes Mellitus", overview["conditions"])

    def test_rag_retrieval_medications(self):
        context, hits = build_context_with_sources("patient_001", "What medicines do I take?", k=3)
        self.assertIn("PATIENT SUMMARY", context)
        self.assertIn("Metformin", context)
        self.assertGreater(len(hits), 0)

    def test_rag_retrieval_allergies(self):
        context, hits = build_context_with_sources("patient_001", "Am I allergic to penicillin?", k=3)
        self.assertIn("Penicillin", context)

    def test_query_expansion_synonyms(self):
        expanded = expand_query("What pills should I take for bp?")
        # Should expand "pills" -> medication/tablet, "bp" -> hypertension/amlodipine
        self.assertTrue(any("medicat" in t or "hyperten" in t for t in expanded))

    def test_tokenization_and_stemming(self):
        tokens = tokenize("Taking medications daily")
        self.assertIn("medication", tokens)
        self.assertIn("tak", tokens)


if __name__ == "__main__":
    unittest.main()
