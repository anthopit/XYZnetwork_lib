import torch
import torch.nn.functional as F
def info_nce_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, p=2, dim=-1)
    z2 = F.normalize(z2, p=2, dim=-1)
    sim_matrix = torch.matmul(z1, z2.t())

    pos_sim = torch.diag(sim_matrix)
    neg_sim = torch.exp(sim_matrix) / (torch.exp(sim_matrix).sum(dim=-1, keepdim=True) - torch.exp(pos_sim).unsqueeze(-1))
    neg_sim = neg_sim.sum(dim=-1)

    loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + neg_sim)).mean()
    return loss / temperature