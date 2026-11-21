from transformers import PretrainedConfig


class HobbyLMConfig(PretrainedConfig):
    """Bidirectional mixture-of-experts text encoder (dense embedding output)."""

    model_type = "hobbylm"

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=512,
        num_hidden_layers=12,
        num_dense_layers=1,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        dense_intermediate_size=1536,
        expert_intermediate_size=192,
        num_experts=32,
        num_experts_per_tok=8,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=1024,
        eot_token_id=50256,
        pad_token_id=50256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_dense_layers = num_dense_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.dense_intermediate_size = dense_intermediate_size
        self.expert_intermediate_size = expert_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.eot_token_id = eot_token_id
        super().__init__(pad_token_id=pad_token_id, **kwargs)
