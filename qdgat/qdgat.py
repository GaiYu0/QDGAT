import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import allennlp_util as util
from transformers.activations import gelu

class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # compatible with xavier_initializer in TensorFlow
        fan_avg = (self.in_features + self.out_features) / 2.
        bound = np.sqrt(3. / fan_avg)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

def generate_scaled_var_drop_mask(shape, keep_prob):
    assert keep_prob > 0. and keep_prob <= 1.
    mask = torch.rand(shape, device='cuda').le(keep_prob)
    mask = mask.float() / keep_prob
    return mask

class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)

class BERTLayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BertFeedForward(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size):
        super(BertFeedForward, self).__init__()
        self.dense = nn.Linear(input_size, intermediate_size)
        self.affine = nn.Linear(intermediate_size, output_size)
        self.act_fn = gelu
        self.LayerNorm = BERTLayerNorm(intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.affine(hidden_states)
        return hidden_states

class QDGAT(nn.Module):
    def __init__(self, node_dim, edge_type_num, iter_num=4, alpha=0.2):
        super().__init__()
        self.iter_num = iter_num
        self.stemDropout = 0.82
        self.readDropout = 0.85
        self.memoryDropout = 0.85
        self.alpha = 0.2
        self.node_dim = node_dim
        self.build_loc_ctx_init()
        self.build_extract_textual_command()
        self.build_propagate_message()
        self.edge_type_num = edge_type_num + 1
        self.wes = [nn.Parameter(torch.zeros(size=(2*node_dim, 1)).cuda()) for _ in range(edge_type_num)]
        assert self.iter_num == 4
        for a in self.wes:
          nn.init.xavier_uniform_(a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def build_loc_ctx_init(self):
        self.initKB = Linear(self.node_dim, self.node_dim)
        self.x_loc_drop = nn.Dropout(1 - self.stemDropout)

        self.initMem = nn.Parameter(torch.randn(1, self.node_dim))

    def build_extract_textual_command(self):
        self.qInput = Linear(self.node_dim, self.node_dim)
        for t in range(self.iter_num):
            qInput_layer2 = Linear(self.node_dim, self.node_dim)
            setattr(self, "qInput%d" % t, qInput_layer2)
        self.cmd_inter2logits = Linear(self.node_dim, 1)

    def build_propagate_message(self):
        self.read_drop = nn.Dropout(1 - self.readDropout)
        self.project_x_loc = Linear(self.node_dim, self.node_dim)
        self.project_x_ctx = Linear(self.node_dim, self.node_dim)
        self.queries = Linear(3*self.node_dim, self.node_dim)
        self.keys = Linear(3*self.node_dim, self.node_dim)
        self.vals = Linear(3*self.node_dim, self.node_dim)
        self.proj_keys = Linear(self.node_dim, self.node_dim)
        self.proj_vals = Linear(self.node_dim, self.node_dim)
        self.mem_update = Linear(2*self.node_dim, self.node_dim)
        self.combine_kb = Linear(2*self.node_dim, self.node_dim)

    def forward(self, ents, ent_mask, q_encoding, adj, types, is_print=False):
        x_loc, x_ctx, x_ctx_var_drop = self.loc_ctx_init(ents, ent_mask)
        for t in range(self.iter_num):
            x_ctx = self.run_message_passing_iter(
                q_encoding, x_loc, x_ctx,
                x_ctx_var_drop, ent_mask, adj, types, t, is_print)

        x_out = self.combine_kb(torch.cat([x_loc, x_ctx], dim=-1))
        return x_out

    def extract_textual_command(self, q_encoding, t):
        qInput_layer2 = getattr(self, "qInput%d" % t)
        q_cmd = qInput_layer2(F.elu(self.qInput(q_encoding)))
        return q_cmd

    def propagate_message0(self, cmd, x_loc, x_ctx, x_ctx_var_drop, x_mask, adj, types):

        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc(self.read_drop(x_loc))
        proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx))
        x_joint = torch.cat(
            [x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)

        queries = self.queries(x_joint)#w6
        keys = self.keys(x_joint) * self.proj_keys(cmd)[:, None, :]
        vals = self.vals(x_joint) * self.proj_vals(cmd)[:, None, :]
        batch_size = keys.size(0)
        N = keys.size(1)

        edge_score = (
            torch.bmm(queries, torch.transpose(keys, 1, 2)) /
            np.sqrt(self.node_dim))
        edge_score=util.replace_masked_values(edge_score, adj>0, -1e30)

        edge_prob = F.softmax(edge_score, dim=-1)

        edge_prob=util.replace_masked_values(edge_prob, adj>0, 0)
        message = torch.bmm(edge_prob, vals)

        x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1))
        return x_ctx_new



    def propagate_message(self, cmd, x_loc, x_ctx, x_ctx_var_drop, x_mask, adj, types, is_print=False):
        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc(self.read_drop(x_loc))
        proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx))
        x_joint = torch.cat(
            [x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)

        queries = self.queries(x_joint)#w6
        keys = self.keys(x_joint) * self.proj_keys(cmd)[:, None, :]
        vals = self.vals(x_joint) * self.proj_vals(cmd)[:, None, :]
        batch_size = keys.size(0)
        N = keys.size(1)

        aij = torch.cat([queries.repeat(1, 1, N).view(batch_size, N*N, -1), keys.repeat(1, N, 1)], dim=-1).view(batch_size, N, N, -1)
        edge_score = torch.zeros(adj.shape, device=adj.device)
        for i in range(self.edge_type_num):
          if (types >> i)&1==1:
            z = torch.matmul((aij*(adj.unsqueeze(-1)==i).float()), self.wes[i-1]).squeeze()
            edge_score += z

        e = self.leakyrelu(edge_score)#[bs, N, N]

        zero_vec = -9e15*torch.ones_like(e)#[N,N]
        edge_prob = torch.where(adj > 0, e, zero_vec)
        edge_prob = F.softmax(edge_prob, dim=-1)

        message = torch.bmm(edge_prob, vals)

        x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1))
        return x_ctx_new

    def run_message_passing_iter(
            self, q_encoding, x_loc, x_ctx,
            x_ctx_var_drop, x_mask, adj, types, t, is_print=False):
        cmd = self.extract_textual_command(
                q_encoding, t)
        x_ctx = self.propagate_message(
            cmd, x_loc, x_ctx, x_ctx_var_drop, x_mask, adj, types, is_print)
        return x_ctx

    def loc_ctx_init(self, ents, ent_mask):
        x_loc = ents

        x_ctx = self.initMem.expand(x_loc.size())

        x_ctx_var_drop = generate_scaled_var_drop_mask(
            x_ctx.size(),
            keep_prob=(self.memoryDropout if self.training else 1.))
        mask = ent_mask[:,:,None].expand(-1,-1,1024).float()
        x_ctx_var_drop = x_ctx_var_drop*mask
        return x_loc, x_ctx, x_ctx_var_drop

