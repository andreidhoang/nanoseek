import torch

from model import create_mla_from_config as create_mla_pkg
from model import create_moe_from_config as create_moe_pkg
from model.config import get_nanoseek_config
from model.model import create_mla_from_config as create_mla_model
from model.model import create_moe_from_config as create_moe_model


def test_mla_factory_parity_and_layer_idx():
    config = get_nanoseek_config()
    mla_pkg = create_mla_pkg(config, layer_idx=3)
    mla_model = create_mla_model(config, layer_idx=3)

    assert mla_pkg.layer_idx == 3
    assert mla_model.layer_idx == 3
    assert mla_pkg.q_lora_rank == mla_model.q_lora_rank == config.mla.q_lora_rank
    assert mla_pkg.kv_lora_rank == mla_model.kv_lora_rank == config.mla.kv_lora_rank


def test_moe_factory_parity():
    config = get_nanoseek_config()
    moe_pkg = create_moe_pkg(config)
    moe_model = create_moe_model(config)

    assert moe_pkg.n_routed_experts == moe_model.n_routed_experts == config.moe.n_routed_experts
    assert moe_pkg.n_activated_experts == moe_model.n_activated_experts == config.moe.num_experts_per_tok


def test_mla_factory_forward_shapes():
    config = get_nanoseek_config()
    mla = create_mla_pkg(config)
    x = torch.randn(2, 8, config.hidden_size)
    out, cache = mla(x, use_cache=True)

    assert out.shape == (2, 8, config.hidden_size)
    assert cache is not None
    kv_compressed, k_pe = cache
    assert kv_compressed.shape[-1] == config.mla.kv_lora_rank
    assert k_pe.shape[-1] == config.mla.qk_rope_head_dim
