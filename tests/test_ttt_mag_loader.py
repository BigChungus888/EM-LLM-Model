import importlib.machinery
import sys
import types
import unittest
from unittest import mock

if "datasets" not in sys.modules:
    datasets_module = types.ModuleType("datasets")
    datasets_module.load_dataset = mock.Mock()
    datasets_module.load_from_disk = mock.Mock()
    datasets_module.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)
    sys.modules["datasets"] = datasets_module

if "omegaconf" not in sys.modules:
    omegaconf_module = types.ModuleType("omegaconf")
    omegaconf_module.OmegaConf = mock.Mock()
    omegaconf_module.__spec__ = importlib.machinery.ModuleSpec("omegaconf", loader=None)
    sys.modules["omegaconf"] = omegaconf_module

if "accelerate" not in sys.modules:
    accelerate_module = types.ModuleType("accelerate")
    accelerate_module.init_empty_weights = mock.Mock()
    accelerate_module.load_checkpoint_and_dispatch = mock.Mock()
    accelerate_module.__spec__ = importlib.machinery.ModuleSpec("accelerate", loader=None)
    sys.modules["accelerate"] = accelerate_module

from benchmark import pred


class _AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class TitansMAGLoaderSelectionTests(unittest.TestCase):
    def _model_config(self, model_type):
        return _AttrDict(
            type=model_type,
            path="/tmp/checkpoint",
            tokenizer_path="/tmp/tokenizer",
            use_hf_acc=False,
            n_local=8,
            n_init=2,
            max_block_size=4,
            max_cached_block=8,
            exc_block_size=4,
            repr_topk=1,
            n_mem=8,
            similarity_refinement_kwargs={},
            contiguity_buffer_kwargs={},
        )

    def test_ttt_mag_uses_native_loader_and_skips_hf_patch_path(self):
        model_config = self._model_config("ttt-mag")
        fake_tokenizer = object()
        fake_model = object()
        titans_cls = mock.Mock()
        titans_cls.from_pretrained.return_value = fake_model

        with mock.patch.object(pred.AutoTokenizer, "from_pretrained", return_value=fake_tokenizer):
            with mock.patch.object(pred, "import_titans_mag_runtime", return_value=titans_cls):
                with mock.patch.object(pred, "patch_titans_mag_model", return_value="patched-titans") as patch_native:
                    with mock.patch.object(pred, "patch_hf") as patch_hf:
                        with mock.patch.object(pred.AutoModelForCausalLM, "from_pretrained") as auto_loader:
                            with mock.patch.object(pred.torch.cuda, "device_count", return_value=1):
                                model, tokenizer = pred.get_model_and_tokenizer(model_config)

        self.assertEqual(model, "patched-titans")
        self.assertIs(tokenizer, fake_tokenizer)
        titans_cls.from_pretrained.assert_called_once_with(
            model_config.path,
            torch_dtype="auto",
            trust_remote_code=True,
            device_map="cuda",
        )
        patch_native.assert_called_once_with(fake_model, **model_config)
        patch_hf.assert_not_called()
        auto_loader.assert_not_called()

    def test_ttt_mag_rejects_multi_gpu_loading(self):
        model_config = self._model_config("ttt-mag")

        with mock.patch.object(pred.torch.cuda, "device_count", return_value=2):
            with self.assertRaisesRegex(ValueError, "single-GPU only"):
                pred.load_titans_mag_model(model_config)

    def test_em_llm_keeps_hf_patch_loader(self):
        model_config = self._model_config("em-llm")
        fake_tokenizer = object()
        fake_model = object()

        with mock.patch.object(pred.AutoTokenizer, "from_pretrained", return_value=fake_tokenizer):
            with mock.patch.object(pred.AutoModelForCausalLM, "from_pretrained", return_value=fake_model) as auto_loader:
                with mock.patch.object(pred, "patch_hf", return_value="patched-hf") as patch_hf:
                    with mock.patch.object(pred, "patch_titans_mag_model") as patch_native:
                        model, tokenizer = pred.get_model_and_tokenizer(model_config)

        self.assertEqual(model, "patched-hf")
        self.assertIs(tokenizer, fake_tokenizer)
        auto_loader.assert_called_once()
        patch_hf.assert_called_once_with(fake_model, "em-llm", **model_config)
        patch_native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
