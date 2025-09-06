import torch
from safetensors import safe_open
from inference.utils import load_into
from inference.configs import CLIPG_CONFIG, CLIPL_CONFIG, T5_CONFIG
from inference.t5 import T5
from inference.clip import CLIPEmbeddings, CLIPEncoder

class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        tokens = list(map(lambda a: a[0], token_weight_pairs[0]))
        out, pooled = self([tokens])
        if pooled is not None:
            first_pooled = pooled[0:1].cpu()
        else:
            first_pooled = pooled
        output = [out[0:1]]
        return torch.cat(output, dim=-2).cpu(), first_pooled


class CLIPTextModel_(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        super().__init__()
        self.embeddings = CLIPEmbeddings(embed_dim, dtype=torch.float32, device=device)
        self.encoder = CLIPEncoder(
            num_layers,
            embed_dim,
            heads,
            intermediate_size,
            intermediate_activation,
            dtype,
            device,
        )
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim, dtype=dtype, device=device)

    def forward(
        self, input_tokens, intermediate_output=None, final_layer_norm_intermediate=True
    ):
        x = self.embeddings(input_tokens)
        causal_mask = (
            torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
            .fill_(float("-inf"))
            .triu_(1)
        )
        x, i = self.encoder(
            x, mask=causal_mask, intermediate_output=intermediate_output
        )
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)
        pooled_output = x[
            torch.arange(x.shape[0], device=x.device),
            input_tokens.to(dtype=torch.int, device=x.device).argmax(dim=-1),
        ]
        return x, i, pooled_output


class CLIPTextModel(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.text_model = CLIPTextModel_(config_dict, dtype, device)
        embed_dim = config_dict["hidden_size"]
        self.text_projection = torch.nn.Linear(
            embed_dim, embed_dim, bias=False, dtype=dtype, device=device
        )
        self.text_projection.requires_grad_(False)
        self.text_projection.weight.copy_(torch.eye(embed_dim))
        self.text_projection.requires_grad_(True)
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return (x[0], x[1], out, x[2])


class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        device="cpu",
        max_length=77,
        layer="last",
        layer_idx=None,
        textmodel_json_config=None,
        dtype=None,
        model_class=CLIPTextModel,
        special_tokens={"start": 49406, "end": 49407, "pad": 49407},
        layer_norm_hidden_state=True,
        return_projected_pooled=True,
    ):
        super().__init__()
        assert layer in self.LAYERS
        self.transformer = model_class(textmodel_json_config, dtype, device)
        self.num_layers = self.transformer.num_layers
        self.max_length = max_length
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.set_clip_options({"layer": layer_idx})
        self.options_default = (
            self.layer,
            self.layer_idx,
            self.return_projected_pooled,
        )

    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get(
            "projected_pooled", self.return_projected_pooled
        )
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        tokens = torch.LongTensor(tokens).to(device)
        outputs = self.transformer(
            tokens,
            intermediate_output=self.layer_idx,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
        )
        self.transformer.set_input_embeddings(backup_embeds)
        if self.layer == "last":
            z = outputs[0]
        else:
            z = outputs[1]
        pooled_output = None
        if len(outputs) >= 3:
            if (
                not self.return_projected_pooled
                and len(outputs) >= 4
                and outputs[3] is not None
            ):
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()
        return z.float(), pooled_output


class SDXLClipG(SDClipModel):
    """Wraps the CLIP-G model into the SD-CLIP-Model interface"""

    def __init__(
        self, config, device="cpu", layer="penultimate", layer_idx=None, dtype=None
    ):
        if layer == "penultimate":
            layer = "hidden"
            layer_idx = -2
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config=config,
            dtype=dtype,
            special_tokens={"start": 49406, "end": 49407, "pad": 0},
            layer_norm_hidden_state=False,
        )


class T5XXLModel(SDClipModel):
    """Wraps the T5-XXL model into the SD-CLIP-Model interface for convenience"""

    def __init__(self, config, device="cpu", layer="last", layer_idx=None, dtype=None):
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config=config,
            dtype=dtype,
            special_tokens={"end": 1, "pad": 0},
            model_class=T5,
        )


class ClipG:
    def __init__(self, model_folder: str, device: str = "cpu"):
        with safe_open(
            f"{model_folder}/clip_g.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = SDXLClipG(CLIPG_CONFIG, device=device, dtype=torch.float32)
            load_into(f, self.model.transformer, "", device, torch.float32)


class ClipL:
    def __init__(self, model_folder: str, device: str = "cpu"):
        with safe_open(
            f"{model_folder}/clip_l.safetensors", framework="pt", device=device
        ) as f:
            self.model = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device=device,
                dtype=torch.float32,
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            load_into(f, self.model.transformer, "", "cuda", torch.float32)


class T5XXL:
    def __init__(self, model_folder: str, device: str = "cpu", dtype=torch.float32):
        with safe_open(
            f"{model_folder}/t5xxl.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = T5XXLModel(T5_CONFIG, device=device, dtype=dtype)
            load_into(f, self.model.transformer, "", device, dtype)
