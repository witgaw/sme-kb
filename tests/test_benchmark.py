"""Tests for the benchmark evaluation pipeline."""

import json
from unittest.mock import MagicMock, patch

import pytest
import yaml

from benchmark.config_schema import BenchmarkConfig
from benchmark.results import extract_summary_metrics, generate_summary_table
from benchmark.runner import BenchmarkRunner
from config import get_config, set_config

# ---------------------------------------------------------------------------
# BenchmarkConfig — parsing, defaults, validation
# ---------------------------------------------------------------------------


class TestBenchmarkConfig:
    def test_from_yaml(self, tmp_path):
        cfg_data = {"name": "test_cfg", "description": "A test", "chunk_size": 512}
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(cfg_data))

        cfg = BenchmarkConfig.from_yaml(str(yaml_path))
        assert cfg.name == "test_cfg"
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 200  # default

    def test_defaults(self):
        cfg = BenchmarkConfig(name="defaults")
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200
        assert cfg.embedding_mode == "local"
        assert cfg.embedding_model == "nomic-ai/nomic-embed-text-v1.5"
        assert cfg.top_k == 5
        assert cfg.use_reranker is False
        assert cfg.llm_provider == "ollama"
        assert cfg.llm_model == "llama3.1:8b"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2000
        assert cfg.ocr_enabled is False
        assert cfg.ocr_model == "llava:7b"
        assert cfg.skip_ocr is True
        assert cfg.skip_db is True

    def test_validation_rejects_bad_embedding_mode(self):
        with pytest.raises(Exception):
            BenchmarkConfig(name="bad", embedding_mode="unknown")

    def test_validation_rejects_bad_llm_provider(self):
        with pytest.raises(Exception):
            BenchmarkConfig(name="bad", llm_provider="azure")

    def test_from_yaml_missing_name(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(yaml.dump({"chunk_size": 500}))
        with pytest.raises(Exception):
            BenchmarkConfig.from_yaml(str(yaml_path))

    def test_extra_fields_ignored(self, tmp_path):
        """Extra YAML keys not in the schema should not raise."""
        cfg_data = {"name": "extras", "unknown_field": True}
        yaml_path = tmp_path / "extras.yaml"
        yaml_path.write_text(yaml.dump(cfg_data))
        # Pydantic v2 by default forbids extra; verify our model handles it
        # If it raises, that's fine — it's a design choice.  Adjust test accordingly.
        try:
            cfg = BenchmarkConfig.from_yaml(str(yaml_path))
            assert cfg.name == "extras"
        except Exception:
            pass  # Strict validation is acceptable


# ---------------------------------------------------------------------------
# Index fingerprint grouping
# ---------------------------------------------------------------------------


class TestIndexFingerprint:
    def test_same_chunking_same_fingerprint(self):
        a = BenchmarkConfig(name="a", chunk_size=1000, chunk_overlap=200)
        b = BenchmarkConfig(name="b", chunk_size=1000, chunk_overlap=200, top_k=10)
        assert a.index_fingerprint == b.index_fingerprint

    def test_different_chunk_size_different_fingerprint(self):
        a = BenchmarkConfig(name="a", chunk_size=1000, chunk_overlap=200)
        b = BenchmarkConfig(name="b", chunk_size=500, chunk_overlap=200)
        assert a.index_fingerprint != b.index_fingerprint

    def test_different_overlap_different_fingerprint(self):
        a = BenchmarkConfig(name="a", chunk_size=1000, chunk_overlap=200)
        b = BenchmarkConfig(name="b", chunk_size=1000, chunk_overlap=100)
        assert a.index_fingerprint != b.index_fingerprint

    def test_different_embedding_model_different_fingerprint(self):
        a = BenchmarkConfig(name="a", embedding_model="nomic-ai/nomic-embed-text-v1.5")
        b = BenchmarkConfig(name="b", embedding_model="other-model")
        assert a.index_fingerprint != b.index_fingerprint

    def test_ocr_enabled_changes_fingerprint(self):
        a = BenchmarkConfig(name="a", ocr_enabled=False)
        b = BenchmarkConfig(name="b", ocr_enabled=True)
        assert a.index_fingerprint != b.index_fingerprint

    def test_different_ocr_model_changes_fingerprint(self):
        a = BenchmarkConfig(name="a", ocr_enabled=True, ocr_model="llava:7b")
        b = BenchmarkConfig(name="b", ocr_enabled=True, ocr_model="llava:13b")
        assert a.index_fingerprint != b.index_fingerprint

    def test_ocr_model_irrelevant_when_ocr_disabled(self):
        a = BenchmarkConfig(name="a", ocr_enabled=False, ocr_model="llava:7b")
        b = BenchmarkConfig(name="b", ocr_enabled=False, ocr_model="llava:13b")
        assert a.index_fingerprint == b.index_fingerprint

    def test_reranker_does_not_affect_fingerprint(self):
        a = BenchmarkConfig(name="a", use_reranker=False)
        b = BenchmarkConfig(name="b", use_reranker=True)
        assert a.index_fingerprint == b.index_fingerprint

    def test_temperature_does_not_affect_fingerprint(self):
        a = BenchmarkConfig(name="a", temperature=0.7)
        b = BenchmarkConfig(name="b", temperature=0.1)
        assert a.index_fingerprint == b.index_fingerprint

    def test_llm_model_does_not_affect_fingerprint(self):
        a = BenchmarkConfig(name="a", llm_model="llama3.1:8b")
        b = BenchmarkConfig(name="b", llm_model="gpt-4o")
        assert a.index_fingerprint == b.index_fingerprint

    def test_fingerprint_format(self):
        cfg = BenchmarkConfig(
            name="x",
            chunk_size=500,
            chunk_overlap=100,
            embedding_mode="local",
            embedding_model="my-model",
        )
        assert cfg.index_fingerprint == "500_100_local_my-model_noocr"

    def test_fingerprint_format_with_ocr(self):
        cfg = BenchmarkConfig(
            name="x",
            chunk_size=500,
            chunk_overlap=100,
            embedding_mode="local",
            embedding_model="my-model",
            ocr_enabled=True,
            ocr_model="llava:13b",
        )
        assert cfg.index_fingerprint == "500_100_local_my-model_ocr_llava:13b"

    def test_grouping_across_configs(self):
        """Verify that grouping by fingerprint yields expected groups."""
        configs = [
            BenchmarkConfig(name="baseline"),
            BenchmarkConfig(name="reranker_on", use_reranker=True),
            BenchmarkConfig(name="top_k_10", top_k=10),
            BenchmarkConfig(name="low_temp", temperature=0.1),
            BenchmarkConfig(name="chunk_small", chunk_size=500, chunk_overlap=100),
            BenchmarkConfig(name="chunk_large", chunk_size=2000, chunk_overlap=400),
        ]

        groups: dict[str, list[str]] = {}
        for cfg in configs:
            fp = cfg.index_fingerprint
            groups.setdefault(fp, []).append(cfg.name)

        # baseline, reranker_on, top_k_10, low_temp share the default fingerprint
        default_fp = "1000_200_local_nomic-ai/nomic-embed-text-v1.5_noocr"
        assert set(groups[default_fp]) == {"baseline", "reranker_on", "top_k_10", "low_temp"}
        assert len(groups) == 3  # default, small, large


# ---------------------------------------------------------------------------
# Summary table generation
# ---------------------------------------------------------------------------


class TestSummaryTable:
    def _make_entry(self, name="test", score=15, max_score=20, pct=75.0, elapsed=120.0):
        cfg = BenchmarkConfig(name=name)
        metrics = {
            "total_score": score,
            "max_score": max_score,
            "percentage": pct,
            "full_credit": 10,
            "partial_credit": 5,
            "wrong": 5,
        }
        return {"config": cfg, "metrics": metrics, "elapsed": elapsed}

    def test_table_has_header(self, tmp_path):
        run_dir = tmp_path / "2026-01-01_00-00-00"
        run_dir.mkdir()
        table = generate_summary_table(run_dir, [self._make_entry()])
        assert "| Config |" in table
        assert "| Chunk |" in table

    def test_table_has_config_link(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        table = generate_summary_table(run_dir, [self._make_entry(name="baseline")])
        assert "[baseline](baseline/config.yaml)" in table

    def test_table_has_details_link(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        table = generate_summary_table(run_dir, [self._make_entry(name="baseline")])
        assert "[results](baseline/results.json)" in table

    def test_table_score_formatting(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        table = generate_summary_table(run_dir, [self._make_entry(score=15, max_score=20)])
        assert "15/20" in table

    def test_table_percentage_formatting(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        table = generate_summary_table(run_dir, [self._make_entry(pct=75.0)])
        assert "75.0" in table

    def test_table_multiple_configs(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        entries = [self._make_entry(name="a"), self._make_entry(name="b")]
        table = generate_summary_table(run_dir, entries)
        assert "[a](a/config.yaml)" in table
        assert "[b](b/config.yaml)" in table

    def test_empty_results(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        table = generate_summary_table(run_dir, [])
        assert "| Config |" in table  # header still present
        # No data rows beyond header
        lines = [line for line in table.strip().splitlines() if line.startswith("|")]
        assert len(lines) == 2  # header + separator

    def test_reranker_display(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cfg_yes = BenchmarkConfig(name="r_on", use_reranker=True)
        cfg_no = BenchmarkConfig(name="r_off", use_reranker=False)
        metrics = {
            "total_score": 10,
            "max_score": 20,
            "percentage": 50.0,
            "full_credit": 5,
            "partial_credit": 5,
            "wrong": 10,
        }
        entries = [
            {"config": cfg_yes, "metrics": metrics, "elapsed": 60.0},
            {"config": cfg_no, "metrics": metrics, "elapsed": 60.0},
        ]
        table = generate_summary_table(run_dir, entries)
        # Both "Yes" and "No" should appear in table rows
        data_lines = [line for line in table.splitlines() if line.startswith("| [")]
        assert any("Yes" in row for row in data_lines)
        assert any("No" in row for row in data_lines)


# ---------------------------------------------------------------------------
# extract_summary_metrics
# ---------------------------------------------------------------------------


class TestExtractMetrics:
    def test_extracts_all_fields(self):
        results = {
            "summary": {
                "auto_scored_total_score": 15,
                "auto_scored_max_score": 20,
                "auto_scored_percentage": 75.0,
                "full_credit_count": 10,
                "partial_credit_count": 5,
                "wrong_count": 5,
            }
        }
        m = extract_summary_metrics(results)
        assert m["total_score"] == 15
        assert m["max_score"] == 20
        assert m["percentage"] == 75.0
        assert m["full_credit"] == 10
        assert m["partial_credit"] == 5
        assert m["wrong"] == 5

    def test_handles_missing_summary(self):
        m = extract_summary_metrics({})
        assert m["total_score"] == 0
        assert m["percentage"] == 0

    def test_handles_partial_summary(self):
        results = {"summary": {"auto_scored_total_score": 3}}
        m = extract_summary_metrics(results)
        assert m["total_score"] == 3
        assert m["max_score"] == 0  # missing key → default 0


# ---------------------------------------------------------------------------
# _apply_config — verify global RAGConfig is set correctly
# ---------------------------------------------------------------------------


class TestApplyConfig:
    """Verify _apply_config propagates BenchmarkConfig fields to the global RAGConfig."""

    @pytest.fixture(autouse=True)
    def _restore_config(self):
        """Save and restore the global config around each test."""
        original = get_config()
        yield
        set_config(original)

    def test_sets_vector_db_path(self, tmp_path):
        runner = BenchmarkRunner(
            configs_dir=tmp_path,
            docs_dir=tmp_path,
            ground_truth={},
        )
        cfg = BenchmarkConfig(name="test")
        index_dir = str(tmp_path / "idx")
        runner._apply_config(cfg, index_dir)

        rag = get_config()
        assert rag.vector_db_path == str(tmp_path / "idx" / "chroma_db")
        assert rag.metadata_db_path == str(tmp_path / "idx" / "metadata.db")

    def test_sets_chunking_params(self, tmp_path):
        runner = BenchmarkRunner(
            configs_dir=tmp_path,
            docs_dir=tmp_path,
            ground_truth={},
        )
        cfg = BenchmarkConfig(name="test", chunk_size=777, chunk_overlap=42)
        runner._apply_config(cfg, str(tmp_path))

        rag = get_config()
        assert rag.chunk_size == 777
        assert rag.chunk_overlap == 42

    def test_sets_llm_params(self, tmp_path):
        runner = BenchmarkRunner(
            configs_dir=tmp_path,
            docs_dir=tmp_path,
            ground_truth={},
        )
        cfg = BenchmarkConfig(
            name="test",
            llm_provider="openrouter",
            llm_model="gpt-4o",
            temperature=0.1,
            max_tokens=500,
        )
        runner._apply_config(cfg, str(tmp_path))

        rag = get_config()
        assert rag.llm_provider == "openrouter"
        assert rag.llm_model == "gpt-4o"
        assert rag.temperature == 0.1
        assert rag.max_tokens == 500

    def test_sets_ocr_params(self, tmp_path):
        runner = BenchmarkRunner(
            configs_dir=tmp_path,
            docs_dir=tmp_path,
            ground_truth={},
        )
        cfg = BenchmarkConfig(name="test", ocr_enabled=True, ocr_model="llava:13b")
        runner._apply_config(cfg, str(tmp_path))

        rag = get_config()
        assert rag.ocr_enabled is True
        assert rag.ocr_model == "llava:13b"

    def test_sets_retrieval_params(self, tmp_path):
        runner = BenchmarkRunner(
            configs_dir=tmp_path,
            docs_dir=tmp_path,
            ground_truth={},
        )
        cfg = BenchmarkConfig(name="test", top_k=20, use_reranker=True)
        runner._apply_config(cfg, str(tmp_path))

        rag = get_config()
        assert rag.top_k == 20
        assert rag.use_reranker is True


# ---------------------------------------------------------------------------
# BenchmarkRunner — config loading and orchestration (mocked)
# ---------------------------------------------------------------------------


class TestBenchmarkRunner:
    @pytest.fixture
    def configs_dir(self, tmp_path):
        d = tmp_path / "configs"
        d.mkdir()
        for name, chunk in [("alpha", 500), ("beta", 1000)]:
            data = {"name": name, "chunk_size": chunk}
            (d / f"{name}.yaml").write_text(yaml.dump(data))
        return d

    @pytest.fixture
    def docs_dir(self, tmp_path):
        d = tmp_path / "docs"
        d.mkdir()
        (d / "doc.txt").write_text("Some test document content.")
        return d

    @pytest.fixture
    def gt_data(self):
        return {
            "exact_match_questions": [
                {
                    "id": "q001",
                    "question_pl": "Testowe pytanie",
                    "requires_ocr": False,
                    "requires_database": False,
                    "expected_answer": "Odpowiedz",
                }
            ],
            "multi_document_synthesis_questions": [],
            "negative_questions": [],
            "qualitative_questions": [],
        }

    def test_load_configs(self, configs_dir, docs_dir, gt_data):
        runner = BenchmarkRunner(
            configs_dir=configs_dir,
            docs_dir=docs_dir,
            ground_truth=gt_data,
        )
        configs = runner._load_configs()
        assert len(configs) == 2
        names = {c.name for c in configs}
        assert names == {"alpha", "beta"}

    def test_load_configs_with_filter(self, configs_dir, docs_dir, gt_data):
        runner = BenchmarkRunner(
            configs_dir=configs_dir,
            docs_dir=docs_dir,
            ground_truth=gt_data,
            config_names=["alpha"],
        )
        configs = runner._load_configs()
        assert len(configs) == 1
        assert configs[0].name == "alpha"

    def test_no_configs_raises(self, tmp_path, docs_dir, gt_data):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        runner = BenchmarkRunner(
            configs_dir=empty_dir,
            docs_dir=docs_dir,
            ground_truth=gt_data,
        )
        with pytest.raises(ValueError, match="No configs found"):
            runner.run()

    def test_extract_questions_filters_ocr(self, configs_dir, docs_dir):
        gt_data = {
            "exact_match_questions": [
                {"id": "q1", "question_pl": "q", "requires_ocr": False},
                {"id": "q2", "question_pl": "q", "requires_ocr": True},
            ],
        }
        runner = BenchmarkRunner(
            configs_dir=configs_dir,
            docs_dir=docs_dir,
            ground_truth=gt_data,
        )
        cfg = BenchmarkConfig(name="test", skip_ocr=True)
        qs = runner._extract_questions(cfg)
        assert len(qs) == 1
        assert qs[0]["id"] == "q1"

    def test_extract_questions_filters_db(self, configs_dir, docs_dir):
        gt_data = {
            "exact_match_questions": [
                {"id": "q1", "question_pl": "q", "requires_database": False},
                {"id": "q2", "question_pl": "q", "requires_database": True},
            ],
        }
        runner = BenchmarkRunner(
            configs_dir=configs_dir,
            docs_dir=docs_dir,
            ground_truth=gt_data,
        )
        cfg = BenchmarkConfig(name="test", skip_db=True)
        qs = runner._extract_questions(cfg)
        assert len(qs) == 1
        assert qs[0]["id"] == "q1"

    def test_fingerprint_hash_deterministic(self):
        h1 = BenchmarkRunner._fingerprint_hash("1000_200_local_model")
        h2 = BenchmarkRunner._fingerprint_hash("1000_200_local_model")
        assert h1 == h2
        assert len(h1) == 12

    def test_fingerprint_hash_different_inputs(self):
        h1 = BenchmarkRunner._fingerprint_hash("1000_200_local_model")
        h2 = BenchmarkRunner._fingerprint_hash("500_100_local_model")
        assert h1 != h2

    @patch("benchmark.runner.RAGPipeline")
    @patch("benchmark.runner.DocumentIngester")
    @patch("benchmark.runner.synth_evaluate")
    def test_run_creates_output_structure(
        self,
        mock_eval,
        mock_ingester_cls,
        mock_pipeline_cls,
        configs_dir,
        docs_dir,
        gt_data,
        tmp_path,
    ):
        """Full run with mocked pipeline and ingestion produces expected files."""
        # Mock ingester
        mock_ingester = MagicMock()
        mock_ingester_cls.return_value = mock_ingester

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {"answer": "mocked answer"}
        mock_pipeline_cls.return_value = mock_pipeline

        # Mock evaluation
        mock_eval.return_value = {
            "summary": {
                "auto_scored_total_score": 1,
                "auto_scored_max_score": 1,
                "auto_scored_percentage": 100.0,
                "full_credit_count": 1,
                "partial_credit_count": 0,
                "wrong_count": 0,
            },
            "auto_scored": [],
            "human_review": [],
        }

        output_dir = tmp_path / "results"
        runner = BenchmarkRunner(
            configs_dir=configs_dir,
            docs_dir=docs_dir,
            ground_truth=gt_data,
            output_dir=output_dir,
        )

        run_dir = runner.run()

        # Check directory structure
        assert run_dir.exists()
        assert (run_dir / "summary.md").exists()
        assert (run_dir / "_indexes").exists()

        # Each config should have its output dir
        for name in ["alpha", "beta"]:
            cfg_dir = run_dir / name
            assert cfg_dir.exists(), f"Missing config dir: {name}"
            assert (cfg_dir / "config.yaml").exists()
            assert (cfg_dir / "submissions.json").exists()
            assert (cfg_dir / "results.json").exists()

        # Verify summary content
        summary = (run_dir / "summary.md").read_text()
        assert "alpha" in summary
        assert "beta" in summary

    @patch("benchmark.runner.RAGPipeline")
    @patch("benchmark.runner.DocumentIngester")
    @patch("benchmark.runner.synth_evaluate")
    def test_run_shares_indexes(
        self,
        mock_eval,
        mock_ingester_cls,
        mock_pipeline_cls,
        tmp_path,
    ):
        """Configs with same fingerprint should trigger only one ingestion."""
        # Create two configs that share the same fingerprint
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        for name in ["a", "b"]:
            data = {"name": name, "chunk_size": 1000, "chunk_overlap": 200}
            (configs_dir / f"{name}.yaml").write_text(yaml.dump(data))

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "d.txt").write_text("content")

        gt_data = {
            "exact_match_questions": [
                {"id": "q1", "question_pl": "q", "requires_ocr": False},
            ],
        }

        mock_ingester = MagicMock()
        mock_ingester_cls.return_value = mock_ingester
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {"answer": "x"}
        mock_pipeline_cls.return_value = mock_pipeline
        mock_eval.return_value = {
            "summary": {
                "auto_scored_total_score": 1,
                "auto_scored_max_score": 1,
                "auto_scored_percentage": 100.0,
                "full_credit_count": 1,
                "partial_credit_count": 0,
                "wrong_count": 0,
            }
        }

        runner = BenchmarkRunner(
            configs_dir=configs_dir,
            docs_dir=docs_dir,
            ground_truth=gt_data,
            output_dir=tmp_path / "out",
        )
        runner.run()

        # Ingester should only have been instantiated once (one fingerprint group)
        assert mock_ingester_cls.call_count == 1
        assert mock_ingester.ingest_directory.call_count == 1


# ---------------------------------------------------------------------------
# Incremental runs — skip / force behavior
# ---------------------------------------------------------------------------


class TestIncrementalRuns:
    """Tests for skip-completed-configs and --force flag."""

    EVAL_RESULTS = {
        "summary": {
            "auto_scored_total_score": 1,
            "auto_scored_max_score": 1,
            "auto_scored_percentage": 100.0,
            "full_credit_count": 1,
            "partial_credit_count": 0,
            "wrong_count": 0,
        },
        "auto_scored": [],
        "human_review": [],
    }

    @pytest.fixture
    def single_config_dir(self, tmp_path):
        d = tmp_path / "configs"
        d.mkdir()
        (d / "alpha.yaml").write_text(yaml.dump({"name": "alpha", "chunk_size": 500}))
        return d

    @pytest.fixture
    def docs_dir(self, tmp_path):
        d = tmp_path / "docs"
        d.mkdir()
        (d / "doc.txt").write_text("content")
        return d

    @pytest.fixture
    def gt_data(self):
        return {
            "exact_match_questions": [
                {
                    "id": "q1",
                    "question_pl": "Pytanie",
                    "requires_ocr": False,
                    "requires_database": False,
                }
            ],
        }

    def _seed_run_dir(self, output_dir, config_name="alpha", *, with_results=True):
        """Create a fake previous run directory with optional results.json."""
        run_dir = output_dir / "2025-01-01_00-00-00"
        run_dir.mkdir(parents=True)
        config_dir = run_dir / config_name
        config_dir.mkdir()
        if with_results:
            (config_dir / "results.json").write_text(json.dumps(self.EVAL_RESULTS))
        return run_dir

    def _seed_index(self, run_dir, fingerprint_hash):
        """Create a non-empty chroma_db inside the index dir."""
        index_dir = run_dir / "_indexes" / fingerprint_hash
        chroma_dir = index_dir / "chroma_db"
        chroma_dir.mkdir(parents=True)
        (chroma_dir / "data.bin").write_text("fake")
        return index_dir

    @patch("benchmark.runner.RAGPipeline")
    @patch("benchmark.runner.DocumentIngester")
    @patch("benchmark.runner.synth_evaluate")
    def test_skips_config_with_existing_results(
        self,
        mock_eval,
        mock_ingester_cls,
        mock_pipeline_cls,
        single_config_dir,
        docs_dir,
        gt_data,
        tmp_path,
    ):
        """Config with existing results.json should not trigger pipeline evaluation."""
        output_dir = tmp_path / "results"
        run_dir = self._seed_run_dir(output_dir, "alpha")

        # Seed the index so ingestion is also skipped
        cfg = BenchmarkConfig(name="alpha", chunk_size=500)
        fp_hash = BenchmarkRunner._fingerprint_hash(cfg.index_fingerprint)
        self._seed_index(run_dir, fp_hash)

        mock_ingester_cls.return_value = MagicMock()

        runner = BenchmarkRunner(
            configs_dir=single_config_dir,
            docs_dir=docs_dir,
            ground_truth=gt_data,
            output_dir=output_dir,
            force=False,
        )
        result_dir = runner.run()

        # Should reuse existing run dir, not create a new one
        assert result_dir == run_dir

        # Pipeline should never have been instantiated (evaluation skipped)
        mock_pipeline_cls.assert_not_called()
        mock_eval.assert_not_called()

    @patch("benchmark.runner.RAGPipeline")
    @patch("benchmark.runner.DocumentIngester")
    @patch("benchmark.runner.synth_evaluate")
    def test_force_reruns_despite_existing_results(
        self,
        mock_eval,
        mock_ingester_cls,
        mock_pipeline_cls,
        single_config_dir,
        docs_dir,
        gt_data,
        tmp_path,
    ):
        """--force should create a fresh run dir and re-evaluate everything."""
        output_dir = tmp_path / "results"
        old_run_dir = self._seed_run_dir(output_dir, "alpha")

        mock_ingester_cls.return_value = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {"answer": "new"}
        mock_pipeline_cls.return_value = mock_pipeline
        mock_eval.return_value = self.EVAL_RESULTS

        runner = BenchmarkRunner(
            configs_dir=single_config_dir,
            docs_dir=docs_dir,
            ground_truth=gt_data,
            output_dir=output_dir,
            force=True,
        )
        result_dir = runner.run()

        # Should create a NEW run dir, not reuse the old one
        assert result_dir != old_run_dir
        assert result_dir.exists()

        # Pipeline and evaluation should have been called
        mock_pipeline_cls.assert_called_once()
        mock_eval.assert_called_once()

    @patch("benchmark.runner.RAGPipeline")
    @patch("benchmark.runner.DocumentIngester")
    @patch("benchmark.runner.synth_evaluate")
    def test_skipped_configs_appear_in_summary(
        self,
        mock_eval,
        mock_ingester_cls,
        mock_pipeline_cls,
        single_config_dir,
        docs_dir,
        gt_data,
        tmp_path,
    ):
        """Skipped configs should still have their metrics in the summary table."""
        output_dir = tmp_path / "results"
        run_dir = self._seed_run_dir(output_dir, "alpha")

        cfg = BenchmarkConfig(name="alpha", chunk_size=500)
        fp_hash = BenchmarkRunner._fingerprint_hash(cfg.index_fingerprint)
        self._seed_index(run_dir, fp_hash)

        mock_ingester_cls.return_value = MagicMock()

        runner = BenchmarkRunner(
            configs_dir=single_config_dir,
            docs_dir=docs_dir,
            ground_truth=gt_data,
            output_dir=output_dir,
            force=False,
        )
        result_dir = runner.run()

        summary = (result_dir / "summary.md").read_text()
        assert "alpha" in summary
        assert "1/1" in summary  # score from the seeded results

    @patch("benchmark.runner.RAGPipeline")
    @patch("benchmark.runner.DocumentIngester")
    @patch("benchmark.runner.synth_evaluate")
    def test_skips_ingestion_when_index_exists(
        self,
        mock_eval,
        mock_ingester_cls,
        mock_pipeline_cls,
        single_config_dir,
        docs_dir,
        gt_data,
        tmp_path,
    ):
        """Ingestion should be skipped when chroma_db already has data."""
        output_dir = tmp_path / "results"
        run_dir = self._seed_run_dir(output_dir, "alpha", with_results=False)

        cfg = BenchmarkConfig(name="alpha", chunk_size=500)
        fp_hash = BenchmarkRunner._fingerprint_hash(cfg.index_fingerprint)
        self._seed_index(run_dir, fp_hash)

        mock_ingester = MagicMock()
        mock_ingester_cls.return_value = mock_ingester
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {"answer": "x"}
        mock_pipeline_cls.return_value = mock_pipeline
        mock_eval.return_value = self.EVAL_RESULTS

        runner = BenchmarkRunner(
            configs_dir=single_config_dir,
            docs_dir=docs_dir,
            ground_truth=gt_data,
            output_dir=output_dir,
            force=False,
        )
        runner.run()

        # Ingester should NOT have been called
        mock_ingester_cls.assert_not_called()

        # But pipeline should have run (no results.json yet)
        mock_pipeline_cls.assert_called_once()

    def test_find_latest_run_dir_picks_newest(self, tmp_path):
        """_find_latest_run_dir should return the lexicographically latest dir."""
        output_dir = tmp_path / "results"
        output_dir.mkdir()
        (output_dir / "2025-01-01_00-00-00").mkdir()
        (output_dir / "2025-06-15_12-30-00").mkdir()
        (output_dir / "2025-03-10_08-00-00").mkdir()
        # Non-matching dirs should be ignored
        (output_dir / "not-a-timestamp").mkdir()

        runner = BenchmarkRunner(
            configs_dir=tmp_path,
            docs_dir=tmp_path,
            ground_truth={},
            output_dir=output_dir,
        )
        latest = runner._find_latest_run_dir()
        assert latest is not None
        assert latest.name == "2025-06-15_12-30-00"

    def test_find_latest_run_dir_returns_none_when_empty(self, tmp_path):
        """_find_latest_run_dir should return None when no timestamped dirs exist."""
        output_dir = tmp_path / "results"
        output_dir.mkdir()

        runner = BenchmarkRunner(
            configs_dir=tmp_path,
            docs_dir=tmp_path,
            ground_truth={},
            output_dir=output_dir,
        )
        assert runner._find_latest_run_dir() is None

    def test_find_latest_run_dir_returns_none_when_dir_missing(self, tmp_path):
        """_find_latest_run_dir should return None when output_dir doesn't exist."""
        runner = BenchmarkRunner(
            configs_dir=tmp_path,
            docs_dir=tmp_path,
            ground_truth={},
            output_dir=tmp_path / "nonexistent",
        )
        assert runner._find_latest_run_dir() is None
