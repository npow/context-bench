"""Tests for the CLI entry point."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from context_bench.__main__ import (
    DATASET_LOADERS,
    _derive_name,
    _load_dataset,
    build_parser,
    main,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestArgParsing:
    def test_minimal_args(self):
        parser = build_parser()
        args = parser.parse_args(["--proxy", "http://localhost:7878", "--dataset", "hotpotqa"])
        assert args.proxy == ["http://localhost:7878"]
        assert args.dataset == ["hotpotqa"]
        assert args.model == "gpt-4"
        assert args.max_examples is None
        assert args.output == "table"
        assert args.score_field == "f1"
        assert args.threshold == 0.7

    def test_multiple_proxies(self):
        parser = build_parser()
        args = parser.parse_args([
            "--proxy", "http://localhost:7878",
            "--proxy", "http://localhost:8787",
            "--dataset", "hotpotqa",
        ])
        assert args.proxy == ["http://localhost:7878", "http://localhost:8787"]

    def test_multiple_datasets(self):
        parser = build_parser()
        args = parser.parse_args([
            "--proxy", "http://localhost:7878",
            "--dataset", "hotpotqa",
            "--dataset", "gsm8k",
        ])
        assert args.dataset == ["hotpotqa", "gsm8k"]

    def test_names_paired_with_proxies(self):
        parser = build_parser()
        args = parser.parse_args([
            "--proxy", "http://localhost:7878",
            "--proxy", "http://localhost:8787",
            "--name", "kompact",
            "--name", "headroom",
            "--dataset", "hotpotqa",
        ])
        assert args.name == ["kompact", "headroom"]

    def test_max_examples_short_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            "--proxy", "http://localhost:7878",
            "--dataset", "hotpotqa",
            "-n", "50",
        ])
        assert args.max_examples == 50

    def test_output_json(self):
        parser = build_parser()
        args = parser.parse_args([
            "--proxy", "http://localhost:7878",
            "--dataset", "hotpotqa",
            "--output", "json",
        ])
        assert args.output == "json"

    def test_custom_model(self):
        parser = build_parser()
        args = parser.parse_args([
            "--proxy", "http://localhost:7878",
            "--dataset", "hotpotqa",
            "--model", "claude-sonnet-4-5-20250929",
        ])
        assert args.model == "claude-sonnet-4-5-20250929"

    def test_custom_score_field_and_threshold(self):
        parser = build_parser()
        args = parser.parse_args([
            "--proxy", "http://localhost:7878",
            "--dataset", "hotpotqa",
            "--score-field", "recall",
            "--threshold", "0.5",
        ])
        assert args.score_field == "recall"
        assert args.threshold == 0.5

    def test_missing_proxy_exits(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--dataset", "hotpotqa"])

    def test_missing_dataset_exits(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--proxy", "http://localhost:7878"])


# ---------------------------------------------------------------------------
# Name derivation
# ---------------------------------------------------------------------------


class TestDeriveName:
    def test_localhost_with_port(self):
        assert _derive_name("http://localhost:7878") == "localhost"

    def test_hostname(self):
        assert _derive_name("http://myhost.example.com/v1") == "myhost.example.com"

    def test_ip_address(self):
        assert _derive_name("http://192.168.1.1:8080") == "192.168.1.1"


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------


class TestDatasetResolution:
    def test_known_datasets_all_present(self):
        expected = {
            "hotpotqa", "gsm8k", "bfcl", "apigen", "swebench", "swebench-verified", "swebench-lite",
            "natural-questions", "musique", "narrativeqa", "triviaqa", "frames", "quality",
            "longbench", "longbench-v2", "infinitebench", "nolima", "bbh", "meetingbank", "govreport",
            "humaneval", "mbpp", "multi-news", "dialogsum", "qmsum", "summscreenfd",
            "contract-nli", "scifact", "qasper",
            "mmlu", "arc-challenge", "truthfulqa", "gpqa",
            "hellaswag", "winogrande", "mmlu-pro",
            "drop", "math", "mgsm",
            "ifeval", "alpaca-eval",
            "mt-bench",
        }
        assert set(DATASET_LOADERS.keys()) == expected

    def test_unknown_dataset_raises(self):
        with pytest.raises(SystemExit, match="Unknown dataset"):
            _load_dataset("nonexistent", max_examples=None)

    def test_jsonl_path_loads_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"id": 0, "context": "hello", "answer": "world"}) + "\n")
            f.write(json.dumps({"id": 1, "context": "foo", "answer": "bar"}) + "\n")
            path = f.name

        examples = _load_dataset(path, max_examples=None)
        assert len(examples) == 2
        assert examples[0]["context"] == "hello"

        Path(path).unlink()

    def test_jsonl_respects_max_examples(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(10):
                f.write(json.dumps({"id": i, "context": f"ctx{i}"}) + "\n")
            path = f.name

        examples = _load_dataset(path, max_examples=3)
        assert len(examples) == 3

        Path(path).unlink()

    def test_known_dataset_lazy_imports(self):
        """Loading a known dataset calls the correct module function."""
        fake_data = [{"id": 0, "context": "ctx", "answer": "ans"}]
        mock_loader = mock.MagicMock(return_value=fake_data)

        with mock.patch("importlib.import_module") as mock_import:
            mock_mod = mock.MagicMock()
            mock_mod.hotpotqa = mock_loader
            mock_import.return_value = mock_mod

            result = _load_dataset("hotpotqa", max_examples=5)

        mock_import.assert_called_once_with("context_bench.datasets.huggingface")
        mock_loader.assert_called_once_with(n=5)
        assert result == fake_data


# ---------------------------------------------------------------------------
# Integration: main() wires everything together
# ---------------------------------------------------------------------------


class TestMainIntegration:
    def test_main_table_output(self, capsys):
        """main() with table output runs end-to-end (mocked HTTP)."""
        fake_response = {
            "choices": [{"message": {"content": "Paris"}}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"id": 0, "context": "France capital", "question": "What is the capital?", "answer": "Paris"}) + "\n")
            f.write(json.dumps({"id": 1, "context": "Germany capital", "question": "What is the capital?", "answer": "Berlin"}) + "\n")
            path = f.name

        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode()
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            main(["--proxy", "http://localhost:7878", "--dataset", path, "--output", "table"])

        captured = capsys.readouterr()
        assert "Evaluation Results" in captured.out
        assert "mean_score" in captured.out

        Path(path).unlink()

    def test_main_json_output(self, capsys):
        """main() with JSON output produces valid JSON."""
        fake_response = {
            "choices": [{"message": {"content": "42"}}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"id": 0, "context": "math", "question": "6*7?", "answer": "42"}) + "\n")
            path = f.name

        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode()
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            main(["--proxy", "http://localhost:7878", "--dataset", path, "--output", "json"])

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "rows" in parsed
        assert "summary" in parsed

        Path(path).unlink()

    def test_main_multiple_proxies(self, capsys):
        """main() with multiple proxies creates separate systems."""
        fake_response = {
            "choices": [{"message": {"content": "answer"}}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"id": 0, "context": "ctx", "answer": "answer"}) + "\n")
            path = f.name

        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode()
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            main([
                "--proxy", "http://localhost:7878",
                "--proxy", "http://localhost:8787",
                "--name", "proxy_a",
                "--name", "proxy_b",
                "--dataset", path,
                "--output", "json",
            ])

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        systems = {r["system"] for r in parsed["rows"]}
        assert systems == {"proxy_a", "proxy_b"}

        Path(path).unlink()

    def test_main_auto_names_from_url(self, capsys):
        """When --name is omitted, names are derived from proxy URLs."""
        fake_response = {
            "choices": [{"message": {"content": "ok"}}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"id": 0, "context": "ctx", "answer": "ok"}) + "\n")
            path = f.name

        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode()
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            main([
                "--proxy", "http://localhost:7878",
                "--dataset", path,
                "--output", "json",
            ])

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["rows"][0]["system"] == "localhost"

        Path(path).unlink()

    def test_main_score_field_and_threshold_wired(self, capsys):
        """--score-field and --threshold are passed to metrics."""
        fake_response = {
            "choices": [{"message": {"content": "Paris"}}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"id": 0, "context": "ctx", "question": "q?", "answer": "Paris"}) + "\n")
            path = f.name

        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode()
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            main([
                "--proxy", "http://localhost:7878",
                "--dataset", path,
                "--score-field", "recall",
                "--threshold", "0.5",
                "--output", "json",
            ])

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        # Should have computed metrics
        assert "summary" in parsed
        summary = parsed["summary"]
        assert len(summary) > 0

        Path(path).unlink()

    def test_main_dataset_tag_added(self, capsys):
        """Each example gets a 'dataset' metadata tag."""
        fake_response = {
            "choices": [{"message": {"content": "ok"}}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"id": 0, "context": "ctx", "answer": "ok"}) + "\n")
            path = f.name

        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode()
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        from context_bench.runner import evaluate as real_evaluate

        # Capture what gets passed to evaluate()
        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            with mock.patch("context_bench.runner.evaluate", wraps=real_evaluate) as mock_eval:
                main(["--proxy", "http://localhost:7878", "--dataset", path])

        # Check that the dataset was tagged
        call_kwargs = mock_eval.call_args
        dataset_arg = call_kwargs[1]["dataset"] if "dataset" in call_kwargs[1] else call_kwargs[0][1]
        assert all("dataset" in ex for ex in dataset_arg)
        assert dataset_arg[0]["dataset"] == path

        Path(path).unlink()

    def test_empty_dataset_exits(self):
        """main() exits if no examples are loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write empty file
            path = f.name

        with pytest.raises(SystemExit, match="No examples loaded"):
            main(["--proxy", "http://localhost:7878", "--dataset", path])

        Path(path).unlink()
