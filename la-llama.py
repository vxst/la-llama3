import torch
from torch.functional import F
from tokenizer import get_tokenizer
from rope import apply_rope
from dct import dct_and_idct
import re


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
    return {
        "r": qk @ v,
        "k": k,
        "v": v,
    }

def self_attention_get_k_v(add_x, wk, wv, n):
    k = apply_rope((add_x @ wk.T).float(), idx=n)
    v = (add_x @ wv.T).float()
    return k, v
    
def self_attention_append(add_x, wq, k, v, n):
    q = apply_rope((add_x @ wq.T).float(), idx=n)
    qk = q @ k.T / (128**0.5) + torch.triu(torch.full((n+1, n+1), float("-inf")), 1)
    qk = F.softmax(qk, dim=1)
    return qk @ v


def attention_layer(x, wq, wk, wv, wo, norm_weight):
    x_norm = rms_norm(x, norm_weight)
    wq = wq.view(config["n_heads"], config["dim"] // config["n_heads"], config["dim"]).float()
    wk = wk.view(config["n_kv_heads"], config["dim"] // config["n_heads"], config["dim"]).float()
    wv = wv.view(config["n_kv_heads"], config["dim"] // config["n_heads"], config["dim"]).float()
    heads = [self_attention(x_norm, wq[i], wk[i//4], wv[i//4]) for i in range(config["n_heads"])]
    heads_k = torch.cat([head["k"] for head in heads], dim=-1)
    heads_v = torch.cat([head["v"] for head in heads], dim=-1)
    heads_r = torch.cat([head["r"] for head in heads], dim=-1)
    return heads_r @ wo.T.float() + x, heads_k, heads_v


def attention_layer_append(add_x, k, v, n, wq, wk, wv, wo, norm_weight):
    add_x_norm = rms_norm(add_x, norm_weight)
    k = k.float()
    v = v.float()
    wq = wq.view(config["n_heads"], config["dim"] // config["n_heads"], config["dim"]).float()
    wk = wk.view(config["n_kv_heads"], config["dim"] // config["n_heads"], config["dim"]).float()
    wv = wv.view(config["n_kv_heads"], config["dim"] // config["n_heads"], config["dim"]).float()
    heads_kv = [self_attention_get_k_v(add_x_norm, wk[i//4], wv[i//4], n) for i in range(config["n_heads"])]
    add_k = torch.cat([head[0] for head in heads_kv]).reshape(-1, config["dim"])
    add_v = torch.cat([head[1] for head in heads_kv]).reshape(-1, config["dim"])
    heads_k = torch.cat([k, add_k], dim=0)
    heads_v = torch.cat([v, add_v], dim=0)
    heads = [self_attention_append(
                add_x_norm,
                wq[i],
                heads_k[:, i*128:(i+1)*128], heads_v[:, i*128:(i+1)*128], n) for i in range(config["n_heads"])]
    heads_r = torch.cat(heads, dim=-1)
    return heads_r @ wo.T.float(), heads_k, heads_v


def ff_layer(x, w1, w2, w3, norm_weight):
    x = rms_norm(x, norm_weight)
    w1 = w1.float()
    w2 = w2.float()
    w3 = w3.float()
    return (F.silu(x @ w1.T) * (x @ w3.T)) @ w2.T


def main():
    with torch.inference_mode():
        model = torch.load("Meta-Llama-3-8B/consolidated.00.pth")
        sentence = "So God created mankind in his own image, in the image of God"

        tokenizer = get_tokenizer()
        tokens = torch.tensor(
            tokenizer.encode('<|begin_of_text|>' + sentence
                             , allowed_special={'<|begin_of_text|>'})
        )

        embedding = torch.nn.Embedding(config["vocab_size"], config["dim"])
        embedding.weight.data.copy_(model["tok_embeddings.weight"])
        embedded_tokens = embedding(tokens)
        print(embedded_tokens.shape)


        x = embedded_tokens
        k_cache = []
        v_cache = []
        for i in range(config["n_layers"]):
            mha_result, k, v = attention_layer(x, *[model[name] for name in gen_model_layer_names(i)])
            k_cache.append(dct_and_idct(k))
            v_cache.append(v.to(torch.float8_e5m2))
            ffn_result = ff_layer(mha_result, *[model[name] for name in gen_ff_layer_names(i)])
            x = mha_result.to(torch.float32) + ffn_result
            print(f"Layer {i} done")

        next_token = (x @ model["output.weight"].T.float()).argmax(dim=-1)[-1]
        print(tokenizer.decode([next_token,]))
        current_seq_len = tokens.shape[0]
        tokens = torch.cat([tokens, torch.tensor([next_token])], dim=0)
        
        for n in range(current_seq_len, 30):
            embedded_tokens = torch.cat([embedded_tokens, embedding(next_token).reshape(1, -1)], dim=0)
            x = embedded_tokens
            for i in range(config["n_layers"]):
                x_0 = x[-1:, ]
                mha_result, k, v = attention_layer_append(x_0, k_cache[i], v_cache[i], n, *[model[name] for name in gen_model_layer_names(i)])
                k_cache[i] = dct_and_idct(k)
                v_cache[i] = v.to(torch.float8_e5m2)
                mha_result = mha_result + x
                ffn_result = ff_layer(mha_result, *[model[name] for name in gen_ff_layer_names(i)])
                x = mha_result + ffn_result
                print(".", end="", flush=True)
            next_token = (x @ model["output.weight"].T.float()).argmax(dim=-1)[-1]
            tokens = torch.cat([tokens, torch.tensor([next_token])], dim=0)
            print(tokenizer.decode([next_token,]))
            if re.match(r"^\.", tokenizer.decode([next_token,])):
                break
        print(tokenizer.decode(tokens.tolist()[1:]))


if __name__ == "__main__":
    main()
