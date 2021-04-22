import torch

def contract(a, b, name):
    assert name in a.names
    assert name in b.names

    a_idx = a.names.index(name)
    b_idx = b.names.index(name)

    a_anon = a.rename(None)
    b_anon = b.rename(None)
    if a_idx != len(a.names) - 1:
        a_anon = a_anon.permute(a_idx, -1)
    if b_idx != 0:
        b_anon = b_anon.permute(b_idx, 0)
    result = torch.matmul(a_anon, b_anon)
    result.names = a.names[:a_idx] + a.names[a_idx+1:] + b.names[:b_idx] + b.names[b_idx+1:]
    return result


p1 = torch.randn(20, names=['embed'])
e1 = torch.randn(20, names=['embed'])
p2 = torch.randn(20, names=['embed'])
e2 = torch.randn(20, names=['embed'])

x1 = p1 + e1
x2 = p2 + e2

Q = torch.randn(20, 5, names=['embed', 'attn'])
K = torch.randn(20, 5, names=['embed', 'attn'])

print(torch.dot(torch.matmul(x1, Q), torch.matmul(x2, K)))

QK = contract(Q.rename(embed='q_embed'), K, 'attn')
p1_p2_sim = contract(
    contract(p1.rename(embed='q_embed'), QK, 'q_embed'), p2, 'embed')
e1_p2_sim = contract(
    contract(e1.rename(embed='q_embed'), QK, 'q_embed'), p2, 'embed')
p1_e2_sim = contract(
    contract(p1.rename(embed='q_embed'), QK, 'q_embed'), e2, 'embed')
e1_e2_sim = contract(
    contract(e1.rename(embed='q_embed'), QK, 'q_embed'), e2, 'embed')

print(p1_p2_sim + e1_p2_sim + p1_e2_sim + e1_e2_sim)
