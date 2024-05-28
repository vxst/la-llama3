import torch
from torch.functional import F
from tokenizer import get_tokenizer
from rope import apply_rope


config = {
    "dim": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 128256,
    "multiple_of": 1024,
    "ffn_dim_multiplier": 1.3,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0
}


def gen_model_layer_names(n):
    return [f"layers.{n}.attention{name}.weight" for name in [".wq", ".wk", ".wv", ".wo", "_norm"]]


def gen_ff_layer_names(n):
    return [f"layers.{n}.feed_forward.w{i}.weight" for i in [1, 2, 3]]\
            + [f"layers.{n}.ffn_norm.weight"]


def rms_norm(x, norm_weight):
    rms_total = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + config["norm_eps"])
    return x / rms_total * norm_weight


def self_attention(x, wq, wk, wv):
    n = x.shape[0]
    q = apply_rope((x @ wq.T).float())
    k = apply_rope((x @ wk.T).float())
    v = (x @ wv.T).float()
    qk = q @ k.T / (128**0.5) + torch.triu(torch.full((n, n), float("-inf")), 1)
    qk = F.softmax(qk, dim=1)
    return qk @ v


def attention_layer(x, wq, wk, wv, wo, norm_weight):
    x_norm = rms_norm(x, norm_weight)
    wq = wq.view(config["n_heads"], config["dim"] // config["n_heads"], config["dim"]).float()
    wk = wk.view(config["n_kv_heads"], config["dim"] // config["n_heads"], config["dim"]).float()
    wv = wv.view(config["n_kv_heads"], config["dim"] // config["n_heads"], config["dim"]).float()
    heads = [self_attention(x_norm, wq[i], wk[i//4], wv[i//4]) for i in range(config["n_heads"])]
    return torch.cat(heads, dim=-1) @ wo.T.float() + x


def ff_layer(x, w1, w2, w3, norm_weight):
    x = rms_norm(x, norm_weight)
    w1 = w1.float()
    w2 = w2.float()
    w3 = w3.float()
    return (F.silu(x @ w1.T) * (x @ w3.T)) @ w2.T


def main():
    with torch.inference_mode():
        model = torch.load("Meta-Llama-3-8B/consolidated.00.pth")

        sentence = "For God doth know that in the day ye eat thereof, then your eyes shall be opened, and ye shall be as gods, knowing good and "
        sentence2 = "But ye shall receive "

        tokenizer = get_tokenizer()
        tokens = torch.tensor(
            tokenizer.encode('<|begin_of_text|>' + sentence + '<|end_of_text|>'
                             + '<|begin_of_text|>' + sentence2 + '<|end_of_text|>'
                             , allowed_special={'<|begin_of_text|>', '<|end_of_text|>'})
        )
        end_of_text = tokenizer.encode('<|end_of_text|>', allowed_special={'<|end_of_text|>'})[0]
        eots = [i for i, x in enumerate(tokens) if x == end_of_text]

        embedding = torch.nn.Embedding(config["vocab_size"], config["dim"])
        embedding.weight.data.copy_(model["tok_embeddings.weight"])
        embedded_tokens = embedding(tokens)

        x = embedded_tokens
        for i in range(config["n_layers"]):
            mha_result = attention_layer(x, *[model[name] for name in gen_model_layer_names(i)])
            ffn_result = ff_layer(mha_result, *[model[name] for name in gen_ff_layer_names(i)])
            x = mha_result + ffn_result
            print(f"Layer {i} done")
        x = rms_norm(x, model["norm.weight"])

        out_tokens = (x @ model["output.weight"].T.float()).argmax(dim=-1)
        print(out_tokens)
        results = [out_tokens[i-2] for i in eots]
        print(tokenizer.decode(results))


if __name__ == "__main__":
    main()
