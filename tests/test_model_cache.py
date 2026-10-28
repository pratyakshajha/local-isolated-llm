import unittest
from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from huggingface_hub.utils import CacheNotFound


class ModelCacheTests(unittest.TestCase):
    def model_cache(self):
        self.assertIsNotNone(find_spec("app.services.model_cache"))
        from app.services import model_cache

        return model_cache

    def cache_for(self, snapshots):
        return SimpleNamespace(
            repos=[
                SimpleNamespace(
                    repo_id="google/gemma-3-1b-it",
                    repo_type="model",
                    revisions=snapshots,
                )
            ]
        )

    def snapshot(self, names):
        return SimpleNamespace(
            files=[SimpleNamespace(file_name=name) for name in names]
        )

    def test_reports_complete_cached_snapshot(self):
        model_cache = self.model_cache()

        cache = self.cache_for(
            [self.snapshot({"config.json", "tokenizer.json", "model.safetensors"})]
        )
        with patch.object(model_cache, "scan_cache_dir", return_value=cache):
            self.assertEqual(
                model_cache.get_downloaded_model_ids({"google/gemma-3-1b-it"}),
                {"google/gemma-3-1b-it"},
            )

    def test_enumerates_all_complete_cached_models_without_filter(self):
        model_cache = self.model_cache()
        cache = self.cache_for(
            [self.snapshot({"config.json", "tokenizer.json", "model.safetensors"})]
        )
        cache.repos.append(
            SimpleNamespace(
                repo_id="other/uncatalogued-model",
                repo_type="model",
                revisions=[
                    self.snapshot(
                        {"config.json", "tokenizer.json", "model.safetensors"}
                    )
                ],
            )
        )

        with patch.object(model_cache, "scan_cache_dir", return_value=cache):
            self.assertEqual(
                model_cache.get_downloaded_model_ids(),
                {"google/gemma-3-1b-it", "other/uncatalogued-model"},
            )
            self.assertEqual(
                model_cache.get_downloaded_model_ids({"other/uncatalogued-model"}),
                {"other/uncatalogued-model"},
            )

    def test_rejects_incomplete_snapshot_assets(self):
        model_cache = self.model_cache()

        incomplete_snapshots = (
            self.snapshot({"config.json", "tokenizer.json"}),
            self.snapshot({"tokenizer.json", "model.safetensors"}),
            self.snapshot({"config.json", "model.safetensors"}),
            self.snapshot({"config.json", "vocab.json", "model.safetensors"}),
        )
        for snapshot in incomplete_snapshots:
            with self.subTest(files={file.file_name for file in snapshot.files}):
                with patch.object(
                    model_cache, "scan_cache_dir", return_value=self.cache_for([snapshot])
                ):
                    self.assertEqual(
                        model_cache.get_downloaded_model_ids({"google/gemma-3-1b-it"}),
                        set(),
                    )

    def test_rejects_sharded_snapshot_with_missing_shard(self):
        model_cache = self.model_cache()

        with TemporaryDirectory() as directory:
            index_path = Path(directory) / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps({"weight_map": {"a": "shard-1.safetensors", "b": "shard-2.safetensors"}}),
                encoding="utf-8",
            )
            snapshot = SimpleNamespace(
                files=[
                    SimpleNamespace(file_name="config.json"),
                    SimpleNamespace(file_name="tokenizer.json"),
                    SimpleNamespace(
                        file_name="model.safetensors.index.json", file_path=index_path
                    ),
                    SimpleNamespace(file_name="shard-1.safetensors"),
                ]
            )
            with patch.object(
                model_cache, "scan_cache_dir", return_value=self.cache_for([snapshot])
            ):
                self.assertEqual(
                    model_cache.get_downloaded_model_ids({"google/gemma-3-1b-it"}),
                    set(),
                )

    def test_reports_complete_model_bin_shards_in_cache(self):
        model_cache = self.model_cache()

        with TemporaryDirectory() as directory:
            index_path = Path(directory) / "model.bin.index.json"
            index_path.write_text(
                json.dumps(
                    {"weight_map": {"a": "model-00001-of-00002.bin", "b": "model-00002-of-00002.bin"}}
                ),
                encoding="utf-8",
            )
            snapshot = SimpleNamespace(
                files=[
                    SimpleNamespace(file_name="config.json"),
                    SimpleNamespace(file_name="tokenizer.json"),
                    SimpleNamespace(file_name="model.bin.index.json", file_path=index_path),
                    SimpleNamespace(file_name="model-00001-of-00002.bin"),
                    SimpleNamespace(file_name="model-00002-of-00002.bin"),
                ]
            )
            with patch.object(
                model_cache, "scan_cache_dir", return_value=self.cache_for([snapshot])
            ):
                self.assertEqual(
                    model_cache.get_downloaded_model_ids({"google/gemma-3-1b-it"}),
                    {"google/gemma-3-1b-it"},
                )

    def test_rejects_weight_index_that_points_to_non_weight_file(self):
        model_cache = self.model_cache()

        with TemporaryDirectory() as directory:
            index_path = Path(directory) / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps({"weight_map": {"weight": "config.json"}}),
                encoding="utf-8",
            )
            snapshot = SimpleNamespace(
                files=[
                    SimpleNamespace(file_name="config.json"),
                    SimpleNamespace(file_name="tokenizer.json"),
                    SimpleNamespace(
                        file_name="model.safetensors.index.json", file_path=index_path
                    ),
                ]
            )
            with patch.object(
                model_cache, "scan_cache_dir", return_value=self.cache_for([snapshot])
            ):
                self.assertEqual(
                    model_cache.get_downloaded_model_ids({"google/gemma-3-1b-it"}),
                    set(),
                )

    def test_missing_or_unreadable_cache_returns_no_downloaded_models(self):
        model_cache = self.model_cache()

        for error in (
            CacheNotFound("missing", Path("missing")),
            ValueError("not a cache directory"),
        ):
            with self.subTest(error=type(error).__name__):
                with patch.object(model_cache, "scan_cache_dir", side_effect=error):
                    self.assertEqual(
                        model_cache.get_downloaded_model_ids({"google/gemma-3-1b-it"}),
                        set(),
                    )


if __name__ == "__main__":
    unittest.main()
