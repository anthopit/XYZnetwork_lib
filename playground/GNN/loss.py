import torch
import torch.nn.functional as F

# def info_nce_loss(z1, z2, temperature=0.1):
#     z1 = F.normalize(z1, p=2, dim=-1)
#     z2 = F.normalize(z2, p=2, dim=-1)
#     sim_matrix = torch.matmul(z1, z2.t())
#
#     pos_sim = torch.diag(sim_matrix)
#     neg_sim = torch.exp(sim_matrix) / (torch.exp(sim_matrix).sum(dim=-1, keepdim=True) - torch.exp(pos_sim).unsqueeze(-1))
#     neg_sim = neg_sim.sum(dim=-1)
#
#     loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + neg_sim)).mean()
#     return loss / temperature

def info_nce_loss(readout_anchor, readout_positive, tau=0.5, norm=True):
    """
    The InfoNCE (NT-XENT) loss in contrastive learning. The implementation
    follows the paper `A Simple Framework for Contrastive Learning of
    Visual Representations <https://arxiv.org/abs/2002.05709>`.
    Args:
        readout_anchor, readout_positive: Tensor of shape [batch_size, feat_dim]
        tau: Float. Usually in (0,1].
        norm: Boolean. Whether to apply normlization.
    """

    batch_size = readout_anchor.shape[0]
    sim_matrix = torch.einsum("ik,jk->ij", readout_anchor, readout_positive)

    if norm:
        readout_anchor_abs = readout_anchor.norm(dim=1)
        readout_positive_abs = readout_positive.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum("i,j->ij", readout_anchor_abs, readout_positive_abs)

    sim_matrix = torch.exp(sim_matrix / tau)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss