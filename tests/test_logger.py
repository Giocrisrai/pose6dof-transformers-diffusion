"""Tests para src/utils/logger.py."""
import logging
import os
from pathlib import Path
from src.utils.logger import get_logger, log_run_metadata


class TestGetLogger:
    def test_returns_logger_instance(self):
        logger = get_logger("test1")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test1"

    def test_default_level_info(self):
        # Sin env var TFM_LOG_LEVEL
        os.environ.pop("TFM_LOG_LEVEL", None)
        logger = get_logger("test_default_level")
        assert logger.level == logging.INFO

    def test_explicit_level_debug(self):
        logger = get_logger("test_debug", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_env_var_level(self, monkeypatch):
        monkeypatch.setenv("TFM_LOG_LEVEL", "WARNING")
        logger = get_logger("test_env_warn")
        assert logger.level == logging.WARNING

    def test_idempotent_returns_same_logger(self):
        l1 = get_logger("test_same")
        l2 = get_logger("test_same")
        assert l1 is l2

    def test_handlers_not_duplicated(self):
        l1 = get_logger("test_no_dup")
        n_handlers_first = len(l1.handlers)
        l2 = get_logger("test_no_dup")
        n_handlers_second = len(l2.handlers)
        assert n_handlers_first == n_handlers_second

    def test_with_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        logger = get_logger("test_file", log_file=log_file)
        logger.info("hello")
        # Forzar flush
        for h in logger.handlers:
            h.flush()
        assert log_file.exists()
        content = log_file.read_text()
        assert "hello" in content
        assert "INFO" in content


class TestLogRunMetadata:
    def test_logs_without_error(self, tmp_path):
        # propagate=False, capturamos via file handler
        log_file = tmp_path / "metadata.log"
        logger = get_logger("test_metadata", level="INFO", log_file=log_file)
        log_run_metadata(logger, "MyRun", {"epochs": 30, "lr": 1e-4, "device": "mps"})
        for h in logger.handlers:
            h.flush()
        content = log_file.read_text()
        assert "MyRun" in content
        assert "epochs" in content and "30" in content
        assert "lr" in content
        assert "device" in content and "mps" in content
