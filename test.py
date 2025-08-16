from transformer_lens import HookedTransformer
from circuit_reuse.eap_graph import EAPGraph

m = HookedTransformer.from_pretrained("gpt2", trust_remote_code=True)
g = EAPGraph(m.cfg, upstream_nodes=["mlp","head"], downstream_nodes=["mlp","head"])
print(g.upstream_hooks[:5], g.downstream_hooks[:5])
