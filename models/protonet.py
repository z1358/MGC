import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models import FewShotModel



class protonet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)

    def _forward(self, instance_embs, support_idx, query_idx, open_idx, label, testing=False, cls_params=None):
        emb_dim = instance_embs.size(-1)  # 640

        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(
            *(query_idx.shape + (-1,)))
        source_open = instance_embs[open_idx.contiguous().view(-1)].contiguous().view(*(open_idx.shape + (-1,)))

        # get mean of the support
        proto = support.mean(dim=1)  # Ntask x NK x d 
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

        query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
        source_open = source_open.view(-1, emb_dim).unsqueeze(1)
        query_open = torch.cat([query, source_open], 0)

        if self.args.use_euclidean:
            logits = -(query_open - proto).pow(2).sum(2) / self.args.temperature
        else:
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query_open = query_open.permute([1, 0, 2]).contiguous()
            query_open = F.normalize(query_open, dim=-1)  # normalize for cosine distance

            logits = torch.bmm(query_open, proto.permute([0, 2, 1])) / self.args.temperature
            logits = logits.view(-1, num_proto)

        logits_reg = None
        cls_logits = None
        bcls_logits = None
        return logits, logits_reg, cls_logits, bcls_logits

