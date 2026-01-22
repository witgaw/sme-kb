"""Evaluation using sme-synth-data-gen dataset and scoring."""

import json
from pathlib import Path
from typing import Any

from scripts.evaluate import evaluate as synth_evaluate
from scripts.evaluate import format_markdown_report

from pipeline.rag_pipeline import RAGPipeline


class SyntheticEvaluator:
    """Evaluate RAG system against synthetic dataset with ground truth.

    Wraps sme-synth-data-gen evaluation framework.
    """

    def __init__(
        self,
        ground_truth_path: str | Path,
        rag_pipeline: RAGPipeline,
        skip_ocr: bool = True,
        skip_db: bool = True,
    ):
        """Initialize evaluator.

        Args:
            ground_truth_path: Path to ground truth JSON file.
            rag_pipeline: Configured RAG pipeline instance.
            skip_ocr: Skip questions requiring OCR.
            skip_db: Skip questions requiring database queries.
        """
        self.ground_truth = self._load_ground_truth(ground_truth_path)
        self.rag = rag_pipeline
        self.skip_ocr = skip_ocr
        self.skip_db = skip_db

    def _load_ground_truth(self, path: str | Path) -> dict[str, Any]:
        """Load ground truth from JSON file."""
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _get_questions(self) -> list[dict[str, Any]]:
        """Extract and filter questions from ground truth."""
        all_questions = []

        for category in [
            "exact_match_questions",
            "multi_document_questions",
            "negative_questions",
            "qualitative_questions",
        ]:
            all_questions.extend(self.ground_truth.get(category, []))

        # Filter
        filtered = []
        for q in all_questions:
            if self.skip_ocr and q.get("requires_ocr"):
                continue
            if self.skip_db and q.get("requires_database"):
                continue
            filtered.append(q)

        return filtered

    def run_evaluation(
        self,
        progress_callback: callable | None = None,
    ) -> dict[str, Any]:
        """Run full evaluation.

        Args:
            progress_callback: Optional callback(question_id, current, total).

        Returns:
            Evaluation results dict with auto_scored, human_review, etc.
        """
        questions = self._get_questions()

        if not questions:
            return {"error": "No questions to evaluate", "summary": {}}

        # Generate answers for each question
        submissions = {}
        for i, q in enumerate(questions, 1):
            qid = q["id"]
            question_text = q.get("question_pl", q.get("question_en", ""))

            if progress_callback:
                progress_callback(qid, i, len(questions))

            try:
                result = self.rag.query(question_text, return_sources=False)
                submissions[qid] = result.get("answer", "")
            except Exception as e:
                submissions[qid] = f"[ERROR: {e}]"

        # Run synth-data-gen evaluation
        results = synth_evaluate(submissions, ground_truth=self.ground_truth)

        return results

    def format_report(self, results: dict[str, Any] | None = None) -> str:
        """Format evaluation results as markdown report.

        Args:
            results: Evaluation results. Runs evaluation if None.

        Returns:
            Markdown formatted report string.
        """
        if results is None:
            results = self.run_evaluation()

        rubrics = results.get("rubrics")
        return format_markdown_report(results, rubrics)

    def print_summary(self, results: dict[str, Any] | None = None) -> None:
        """Print evaluation summary to console.

        Args:
            results: Evaluation results. Runs evaluation if None.
        """
        if results is None:
            results = self.run_evaluation()

        summary = results.get("summary", {})

        print("\n" + "=" * 60)
        print("RAG EVALUATION SUMMARY")
        print("=" * 60)

        if summary:
            total = summary.get("auto_scored_total_score", 0)
            max_score = summary.get("auto_scored_max_score", 0)
            pct = summary.get("auto_scored_percentage", 0)

            print(f"\nAuto-scored: {total}/{max_score} ({pct}%)")
            print(f"  Full credit (1.0): {summary.get('full_credit_count', 0)}")
            print(f"  Partial credit (0.5): {summary.get('partial_credit_count', 0)}")
            print(f"  Wrong (0.0): {summary.get('wrong_count', 0)}")

            if summary.get("temporal_total", 0) > 0:
                t_pass = summary.get("temporal_pass", 0)
                t_total = summary.get("temporal_total", 0)
                print(f"\nTemporal filter: {t_pass}/{t_total} passed")

            print(f"\nSemantic analysis: {summary.get('human_review_count', 0)} questions")

            if summary.get("not_answered_count", 0) > 0:
                print(f"Not answered: {summary.get('not_answered_count', 0)}")

        print("\n" + "=" * 60)
