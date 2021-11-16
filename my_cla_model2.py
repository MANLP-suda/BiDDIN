from abc import abstractclassmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import math
# from torch.nn import MultiheadAttention


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M) # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1,2,0) # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, vector

        return attn_pool, alpha
        
class SelfAttention(nn.Module):

    def __init__(self, input_dim, att_type='general'):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.att_type = att_type
        self.scalar = nn.Linear(self.input_dim,1,bias=True)

    def forward(self, M, x=None):
        """
        now M -> (batch, seq_len, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        if self.att_type == 'general':
            scale = self.scalar(M) # seq_len, batch, 1
#            scale = torch.tanh(scale)
            alpha = F.softmax(scale, dim=0).permute(0,2,1) # batch, 1, seq_len
            attn_pool = torch.bmm(alpha, M)[:,0,:] # batch, vector/input_dim
        if self.att_type == 'general2':
            scale = self.scalar(M) # seq_len, batch, 1
#            print('scale', scale.size())
#            scale = F.tanh(scale)
            alpha = F.softmax(scale, dim=0).permute(0,2,1) # batch, 1, seq_len
#            print('alpha', alpha.size())
            att_vec_bag = []
            for i in range(M.size()[1]):
                alp = alpha[:,:,i]
#                print ('alp',alp.size())
                vec = M[:, i, :]
#                print ('vec',vec.size())
                alp = alp.repeat(1,self.input_dim)
#                print ('alp',alp.size())
                att_vec = torch.mul(alp, vec) # batch, vector/input_dim
                att_vec = att_vec + vec
#                att_vec = torch.bmm(alp, vec)[:,0,:] # batch, vector/input_dim
#                print('att_vec', att_vec.size())
                att_vec_bag.append(att_vec)
            attn_pool = torch.cat(att_vec_bag, -1)  # batch. input_dim * 3

        return attn_pool, alpha
        
class LenFirstSelfAttention(nn.Module):

    def __init__(self, input_dim, att_type='general'):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.att_type = att_type
        self.scalar = nn.Linear(self.input_dim,1,bias=True)

    def forward(self, M, x=None):
        """
        previous M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        if self.att_type == 'general':
            scale = self.scalar(M) # seq_len, batch, 1
            scale = F.tanh(scale)
            alpha = F.softmax(scale, dim=0).permute(1,2,0) # batch, 1, seq_len
            attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, vector/input_dim
        if self.att_type == 'general2':
            scale = self.scalar(M) # seq_len, batch, 1
            scale = F.tanh(scale)
            alpha = F.softmax(scale, dim=0).permute(1,2,0) # batch, 1, seq_len
            att_vec_bag = []
            for i in range(M.size()[0]):
                alp = alpha[:,:,i]
                vec = M[i,:, :]
                att_vec = torch.bmm(alp, vec.transpose(0,1))[:,0,:] # batch, vector/input_dim
                att_vec_bag.append(att_vec)
            attn_pool = torch.cat(att_vec_bag, -1)

        return attn_pool, alpha

class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            #torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1,2,0) # batch, vector, seqlen
            x_ = x.unsqueeze(1) # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_*mask.unsqueeze(1) # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True) # batch, 1, 1
            alpha = alpha_masked/alpha_sum # batch, 1, 1 ; normalized
            #import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0,1) # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1) # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_,x_],2) # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_)) # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2) # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, mem_dim

        return attn_pool, alpha


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.P = torch.zeros((1, max_len, num_hiddens))
        self.dropout = nn.Dropout(dropout)
        X = torch.arange(max_len, dtype=torch.float32).reshape(\
            -1, 1) / torch.pow(10000, torch.arange(\
                0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    
    def forward(self, X):
        
        # X: seq_len, batch_size, num_hiddens
        
        X = X.transpose(0, 1)
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X.transpose(0, 1))


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    def forward(self, Q, K, V, attn_mask=None):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask:
            scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, cand_dim, embed_dim, n_heads):
        # cand_dim = query_dim
        # embed_dim = key_dim = value_dim
        super(MultiHeadAttention, self).__init__()
        self.transform = nn.Linear(cand_dim, embed_dim, bias=False)
        assert embed_dim % n_heads == 0
        self.cand_dim = cand_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dim_per_head = self.embed_dim // self.n_heads
        self.W_Q = nn.Linear(self.embed_dim, self.dim_per_head * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.embed_dim, self.dim_per_head * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.embed_dim, self.dim_per_head * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.dim_per_head, self.embed_dim, bias=False)

    def forward(self, input_Q, input_K, input_V):
        # input_Q: [seq_len, batch_size, cand_dim]  or [batch_size, cand_dim]
        # input_K: [seq_len, batch_size, embed_dim]
        # input_V: [seq_len, batch_size, embed_dim]
        if len(input_Q.shape) == 2:
            input_Q = input_Q.unsqueeze(0)
        if input_Q.size(-1) != input_K.size(-1):
            input_Q = self.transform(input_Q)   # [1/seq_len, batch_size, embed_dim]
        residual = input_Q  # [1/seq_len, batch_size, embed_dim]
        batch_size = input_Q.size(1)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1,2) # [batch_size, n_heads, 1/seq_len, dim_per_head]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1,2) # [batch_size, n_heads, seq_len, dim_per_head]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1,2) # [batch_size, n_heads, seq_len, dim_per_head]

        context, attn = ScaledDotProductAttention(self.dim_per_head)(Q, K, V)
        # context: [batch_size, n_heads, 1/seq_len, dim_per_head]
        # attn: [batch_size, n_heads, 1/seq_len, seq_len]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.dim_per_head).transpose(0, 1) # [1/seq_len, batch_size, embed_size]
        output = self.fc(context) # [1/seq_len, batch_size, embed_size]
        attn = attn.reshape(attn.size(2), -1, attn.size(3))  # [1/seq_len, batch_size, seq_len]
        return nn.LayerNorm(self.embed_dim).cuda()(output + residual), attn



class DialogueRNNTriCell(nn.Module):

    def __init__(self, D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, listener_state=False,
                            context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNNTriCell, self).__init__()

        self.D_m_T = D_m_T
        self.D_m_A = D_m_A
        self.D_m_V = D_m_V
        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e

        self.listener_state = listener_state
        self.g_cell_t = nn.GRUCell(D_m+D_p,D_g)
        self.p_cell_t = nn.GRUCell(D_m+D_g,D_p)
        self.e_cell_t = nn.GRUCell(D_p,D_e)
        
        self.g_cell_a = nn.GRUCell(D_m+D_p,D_g)
        self.p_cell_a = nn.GRUCell(D_m+D_g,D_p)
        self.e_cell_a = nn.GRUCell(D_p,D_e)
        
        self.g_cell_v = nn.GRUCell(D_m+D_p,D_g)
        self.p_cell_v = nn.GRUCell(D_m+D_g,D_p)
        self.e_cell_v = nn.GRUCell(D_p,D_e)
        
        self.dense_t = nn.Linear(D_m_T,D_m)
        self.dense_a = nn.Linear(D_m_A,D_m)
        self.dense_v = nn.Linear(D_m_V,D_m)
        
        self.my_self_att1 = SelfAttention(D_g,att_type = 'general2')
        self.my_self_att2 = SelfAttention(D_g,att_type = 'general2')
        self.my_self_att3 = SelfAttention(D_g,att_type = 'general2')
        
        self.dense1 = nn.Linear(self.D_g*3,self.D_g,bias=True)
        self.dense2 = nn.Linear(self.D_g*3,self.D_g,bias=True)
        self.dense3 = nn.Linear(self.D_g*3,self.D_g,bias=True)
        
#        self.dense_u = nn.Linear(D_m,D_m)
        self.self_attention = nn.Linear(D_g,1,bias=True)
        if listener_state:
            self.l_cell = nn.GRUCell(D_m+D_p,D_p)

        self.dropout = nn.Dropout(dropout)
        self.positional_embedding = PositionalEncoding(D_g)

        if context_attention=='simple':
            self.attention1 = SimpleAttention(D_g)
            self.attention2 = SimpleAttention(D_g)
            self.attention3 = SimpleAttention(D_g)
            
            self.attention4 = SimpleAttention(D_g)
            self.attention5 = SimpleAttention(D_g)
            
            self.attention6 = SimpleAttention(D_g)
            self.attention7 = SimpleAttention(D_g)
            
            self.attention8 = SimpleAttention(D_g)
            self.attention9 = SimpleAttention(D_g)
            
#            self.attention = SimpleAttention(D_g)
        else:
            
            self.attention1 = MatchingAttention(D_g, D_m, D_a, context_attention)
            self.attention2 = MatchingAttention(D_g, D_m, D_a, context_attention)
            self.attention3 = MatchingAttention(D_g, D_m, D_a, context_attention)
            
            self.attention4 = MatchingAttention(D_g, D_m, D_a, context_attention)
            self.attention5 = MatchingAttention(D_g, D_m, D_a, context_attention)
            
            self.attention6 = MatchingAttention(D_g, D_m, D_a, context_attention)
            self.attention7 = MatchingAttention(D_g, D_m, D_a, context_attention)
            
            self.attention8 = MatchingAttention(D_g, D_m, D_a, context_attention)
            self.attention9 = MatchingAttention(D_g, D_m, D_a, context_attention)
#            self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)

    def rnn_cell(self,U,c_,qmask,qm_idx,q0,e0,p_cell,e_cell):
        U_c_ = torch.cat([U,c_], dim=1).unsqueeze(1).expand(-1,qmask.size()[1],-1)
        qs_ = p_cell(U_c_.contiguous().view(-1,self.D_m+self.D_g),
                q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
        qs_ = self.dropout(qs_)

        if self.listener_state:
            U_ = U.unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_m)
            ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).\
                    expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_p)
            U_ss_ = torch.cat([U_,ss_],1)
            ql_ = self.l_cell(U_ss_,q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
            ql_ = self.dropout(ql_)
        else:
            ql_ = q0
        qmask_ = qmask.unsqueeze(2)
        q_ = ql_*(1-qmask_) + qs_*qmask_
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0]==0\
                else e0
        e_ = e_cell(self._select_parties(q_,qm_idx), e0)
        e_ = self.dropout(e_)
        return q_,e_

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        return q0_sel

    def forward(self, Ut, Uv, Ua, qmask, g_hist_t, g_hist_v, g_hist_a, q0_t, q0_v, q0_a, e0_t, e0_v, e0_a,k=1):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e-
        """
        # pad_t = torch.zeros(Ut.size()[0],Ut.size()[1]).type(Ua.type())
        # pad_a = torch.zeros(Ua.size()[0],Ua.size()[1]).type(Ua.type())

        # Ut = torch.cat([Ut,pad_a],dim = -1)
        # Ua = torch.cat([pad_t,Ua],dim = -1)

        Ut = self.dense_t(Ut)
        Ua = self.dense_a(Ua)
        Uv = self.dense_v(Uv)
        # Ut = self.dense_u(Ut)
        # Ua = self.dense_u(Ua)

        qm_idx = torch.argmax(qmask, 1)
        q0_sel_t = self._select_parties(q0_t, qm_idx)
        q0_sel_a = self._select_parties(q0_a, qm_idx)
        q0_sel_v = self._select_parties(q0_v, qm_idx)

        g_t = self.g_cell_t(torch.cat([Ut,q0_sel_t], dim=1),
                torch.zeros(Ut.size()[0],self.D_g).type(Ut.type()) if g_hist_t.size()[0]==0 else
                g_hist_t[-1])
        g_t = self.dropout(g_t)
        
        g_v = self.g_cell_v(torch.cat([Uv,q0_sel_v], dim=1),
                torch.zeros(Uv.size()[0],self.D_g).type(Uv.type()) if g_hist_v.size()[0]==0 else
                g_hist_v[-1])
        g_v = self.dropout(g_v)

        g_a = self.g_cell_a(torch.cat([Ua,q0_sel_a], dim=1),
                torch.zeros(Ua.size()[0],self.D_g).type(Ua.type()) if g_hist_a.size()[0]==0 else
                g_hist_a[-1])
        g_a = self.dropout(g_a)
        
        if g_hist_t.size()[0]==0:
            c_t = torch.zeros(Ut.size()[0],self.D_g).type(Ut.type())
            alpha = None
        if g_hist_a.size()[0]==0:
            c_a = torch.zeros(Ua.size()[0],self.D_g).type(Ua.type())
            alpha = None
        if g_hist_v.size()[0]==0:
            c_v = torch.zeros(Uv.size()[0],self.D_g).type(Uv.type())
            alpha = None
        else:
            # c_tt, alpha_tt = self.attention(g_hist_t[:,-2:],Ut)
            # c_at, alpha_at = self.attention(g_hist_t[:,-2:],Ua)
            # c_ta, alpha_ta = self.attention(g_hist_a[:,-2:],Ut)
            # c_aa, alpha_aa = self.attention(g_hist_a[:,-2:],Ua)

            g_hist_a = self.positional_embedding(g_hist_a)
            g_hist_v = self.positional_embedding(g_hist_v)
            g_hist_t = self.positional_embedding(g_hist_t)


            
            c_tt, alpha_tt = self.attention1(g_hist_t,Ut)
            c_vv, alpha_vv = self.attention2(g_hist_v,Uv)
            c_aa, alpha_aa = self.attention3(g_hist_a,Ua)
            
            #T & A
            c_at, alpha_at = self.attention4(g_hist_t,Ua)
            c_ta, alpha_ta = self.attention5(g_hist_a,Ut)
            
            #T & V
            c_vt, alpha_vt = self.attention6(g_hist_t,Uv)
            c_tv, alpha_tv = self.attention7(g_hist_v,Ut)
            
            #A & V
            c_va, alpha_va = self.attention8(g_hist_a,Uv)
            c_av, alpha_av = self.attention9(g_hist_v,Ua)
            

            
            
            alpha = alpha_tt + alpha_vv + alpha_aa + alpha_ta + alpha_at + alpha_tv + alpha_vt + alpha_va + alpha_av 

            c_ttav = torch.cat([c_tt.unsqueeze(1),c_ta.unsqueeze(1),c_tv.unsqueeze(1)],1)
#            print ('c_tta',c_ttav.size()) # batch, 3, D_g
            c_aatv = torch.cat([c_aa.unsqueeze(1),c_at.unsqueeze(1),c_av.unsqueeze(1)],1)
            c_vvta = torch.cat([c_vv.unsqueeze(1),c_vt.unsqueeze(1),c_va.unsqueeze(1)],1)
#            print ('c_aat',c_aatv.size())
            
            c_t,alp1 = self.my_self_att1(c_ttav)
#            print ('c_t',c_t.size())  # batch, D_g * 3
            c_t = self.dense1(c_t)
            
            c_a,alp2 = self.my_self_att2(c_aatv)
#            print ('c_a',c_a.size())
            c_a = self.dense2(c_a)
            
            c_v,alp3 = self.my_self_att3(c_vvta)

            c_v = self.dense3(c_v)

        
        q_t, e_t = self.rnn_cell(Ut,c_t,qmask,qm_idx,q0_t,e0_t,self.p_cell_t,self.e_cell_t)
        q_a, e_a = self.rnn_cell(Ua,c_a,qmask,qm_idx,q0_a,e0_a,self.p_cell_a,self.e_cell_a)
        q_v, e_v = self.rnn_cell(Uv,c_v,qmask,qm_idx,q0_v,e0_v,self.p_cell_v,self.e_cell_v)
#        print(e_t.size())  # batch, De
        

        return g_t,q_t,e_t,g_v,q_v,e_v,g_a,q_a,e_a,alpha
        
class DialogueRNNTri(nn.Module):

    def __init__(self, D_m_T, D_m_V, D_m_A,D_m, D_g, D_p, D_e, listener_state=False,
                            context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNNTri, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = DialogueRNNTriCell(D_m_T, D_m_V, D_m_A,D_m, D_g, D_p, D_e,
                            listener_state, context_attention, D_a, dropout)
        self.self_attention = nn.Linear(D_e,1,bias=True)

    def forward(self, Ut, Uv, Ua, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        g_hist_t = torch.zeros(0).type(Ut.type()) # 0-dimensional tensor
        g_hist_v = torch.zeros(0).type(Uv.type()) # 0-dimensional tensor
        g_hist_a = torch.zeros(0).type(Ua.type()) # 0-dimensional tensor
        q_t = torch.zeros(qmask.size()[1], qmask.size()[2],
                                    self.D_p).type(Ut.type()) # batch, party, D_p
        e_t = torch.zeros(0).type(Ut.type()) # batch, D_e
        et = e_t
        
        q_v = torch.zeros(qmask.size()[1], qmask.size()[2],
                                    self.D_p).type(Uv.type()) # batch, party, D_p
        e_v = torch.zeros(0).type(Uv.type()) # batch, D_e
        ev = e_v

        q_a = torch.zeros(qmask.size()[1], qmask.size()[2],
                                    self.D_p).type(Ua.type()) # batch, party, D_p
        e_a = torch.zeros(0).type(Ua.type()) # batch, D_e
        ea = e_a

        alpha = []
        for u_t, u_v, u_a, qmask_ in zip(Ut, Uv, Ua, qmask):
            g_t,q_t,e_t,g_v,q_v,e_v,g_a,q_a,e_a,alpha_ = self.dialogue_cell(u_t, u_v, u_a, qmask_, g_hist_t, g_hist_v, g_hist_a, q_t,q_v,q_a, e_t,e_v,e_a,k=5)
            g_hist_t = torch.cat([g_hist_t, g_t.unsqueeze(0)],0)
            g_hist_v = torch.cat([g_hist_v, g_v.unsqueeze(0)],0)
            g_hist_a = torch.cat([g_hist_a, g_a.unsqueeze(0)],0)
            et = torch.cat([et, e_t.unsqueeze(0)],0)
            ev = torch.cat([ev, e_v.unsqueeze(0)],0)
            # et = torch.cat([e_t.unsqueeze(0),et],0)
            ea = torch.cat([ea, e_a.unsqueeze(0)],0)
#            print('et', et.size())  # finally  seq_len, batch, De
            if type(alpha_)!=type(None):
                alpha.append(alpha_[:,0,:])

        
        e = torch.cat([et, ev, ea],dim = -1)
#        print ('e', e.size())  # seq_len, batch, De*3
        score_t = torch.cat([et.unsqueeze(2), ev.unsqueeze(2), ea.unsqueeze(2)],dim = 2)
#        print ('score1', score_t.size())  # seq_len, batch, 3, De
        score_t = self.self_attention(score_t)
#        print ('score2', score_t.size())  # seq_len, batch, 3, 1
        score_t = F.softmax(score_t,dim=-1)
        score_t = score_t.squeeze(-1)     # seq_len, batch, 3
        score_t = score_t.repeat(1,1,1,self.D_e).view(len(e),len(e[0]),-1)  # seq_len, batch_size, De*3
#        print('score_t', score_t.size())

        e = torch.mul(e,score_t)  # seq_len, batch_size, De*3
        # e = e + e_temp
#       print ('e', e.size())  


        

        return e,alpha # seq_len, batch, D_e*3


class IEMOCAPBiModelTri(nn.Module):

    def __init__(self, D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, D_h, 
            n_classes=7, listener_state=False, context_attention='simple', D_a=100, dropout_rec=0.5, dropout=0.5):
        super(IEMOCAPBiModelTri, self).__init__()

        self.D_m         = D_m
        self.D_g         = D_g
        self.D_p         = D_p
        self.D_e         = D_e
        self.D_h         = D_h
        self.n_classes = n_classes
        self.dropout     = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout)
        # self.dialog_rnn  = DialogueRNN(D_m, D_g, D_p, D_e,listener_state,
        #                             context_attention, D_a, dropout_rec)
        self.dialog_rnn_f = DialogueRNNTri(D_m_T, D_m_V, D_m_A,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.dialog_rnn_r = DialogueRNNTri(D_m_T, D_m_V, D_m_A,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.positional_embedding = PositionalEncoding(D_e*6)
        self.linear      = nn.Linear(6*D_e, 3*D_h)
        self.smax_fc     = nn.Linear(3*D_h, n_classes)
        self.matchatt = MatchingAttention(6*D_e,6*D_e,att_type='general')
        # self.multihead_attn = MultiHeadAttention(6*D_e, 8) # embed_dim, n_heads
        self.multihead_attn = MultiHeadAttention(6*D_e, 6*D_e, 4)

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, Ut, Uv, Ua, qmask, umask,att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        emotions_f, alpha_f = self.dialog_rnn_f(Ut, Uv, Ua, qmask) # seq_len, batch, D_e
        # emotions_f = self.positional_embedding(emotions_f)
        emotions_f = self.dropout_rec(emotions_f)  # seq_len, batch_size, De*3
        rev_Ut = self._reverse_seq(Ut, umask)
        rev_Uv = self._reverse_seq(Uv, umask)
        rev_Ua = self._reverse_seq(Ua, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b, alpha_b = self.dialog_rnn_r(rev_Ut, rev_Uv, rev_Ua, rev_qmask)
        # emotions_b = self.positional_embedding(emotions_b)
        emotions_b = self._reverse_seq(emotions_b, umask)
        emotions_b = self.dropout_rec(emotions_b)  # seq_len, batch_size, De*3
        emotions = torch.cat([emotions_f,emotions_b],dim=-1) # seq_len, batch_size, De*6
        # emotions = self.positional_embedding(emotions) # seq_len, batch_size, De*6
        # print('emotions_f', emotions_f.size())
        # print('emotions_b', emotions_b.size())
        # print('emotions', emotions.size())
        if att2:
            att_emotions, alpha = self.multihead_attn(emotions, emotions, emotions)
            alpha = list(alpha)
            hidden = F.relu(self.linear(att_emotions))  # seq_len, batch_size, Dh*3

            '''
            att_emotions = []
            alpha = []
            for t in emotions:
                # t size: batch_size, De*6
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)

                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0) # seq_len, batch_size, De*6
            # print("att_emotions", att_emotions.size())
            
            hidden = F.relu(self.linear(att_emotions))  # seq_len, batch_size, Dh*3
            # print("hidden", hidden.size())
            '''
        else:
            hidden = F.relu(self.linear(emotions))
        #hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2) # seq_len, batch, n_classes

        return log_prob, alpha, alpha_f, alpha_b

class IEMOCAPUniModelTri(nn.Module):

    def __init__(self, D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, D_h, 
            n_classes=7, listener_state=False, context_attention='simple', D_a=100, dropout_rec=0.5, dropout=0.5):
        super(IEMOCAPUniModelTri, self).__init__()

        self.D_m         = D_m
        self.D_g         = D_g
        self.D_p         = D_p
        self.D_e         = D_e
        self.D_h         = D_h
        self.n_classes = n_classes
        self.dropout     = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout)
        # self.dialog_rnn  = DialogueRNN(D_m, D_g, D_p, D_e,listener_state,
        #                             context_attention, D_a, dropout_rec)
        self.dialog_rnn_f = DialogueRNNTri(D_m_T, D_m_V, D_m_A,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        # self.dialog_rnn_r = DialogueRNNTri(D_m_T, D_m_V, D_m_A,D_m, D_g, D_p, D_e,listener_state,
        #                             context_attention, D_a, dropout_rec)
        self.linear      = nn.Linear(3*D_e, 3*D_h)
        self.smax_fc     = nn.Linear(3*D_h, n_classes)
        self.matchatt = MatchingAttention(3*D_e,3*D_e,att_type='general')

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, Ut, Uv, Ua, qmask, umask,att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        emotions_f, alpha_f = self.dialog_rnn_f(Ut, Uv, Ua, qmask) # seq_len, batch, D_e
        emotions = self.dropout_rec(emotions_f)
        # rev_Ut = self._reverse_seq(Ut, umask)
        # rev_Uv = self._reverse_seq(Uv, umask)
        # rev_Ua = self._reverse_seq(Ua, umask)
        # rev_qmask = self._reverse_seq(qmask, umask)
        # emotions_b, alpha_b = self.dialog_rnn_r(rev_Ut, rev_Uv, rev_Ua, rev_qmask)
        # emotions_b = self._reverse_seq(emotions_b, umask)
        # emotions_b = self.dropout_rec(emotions_b)

        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        #hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2) # seq_len, batch, n_classes

        return log_prob, alpha, alpha_f, alpha_f


        
class AVECBiModelTri(nn.Module):

    def __init__(self, D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, D_h, attr, listener_state=False,
            context_attention='simple', D_a=100, dropout_rec=0.5, dropout=0.5):
        super(AVECBiModelTri, self).__init__()

        self.D_m         = D_m
        self.D_g         = D_g
        self.D_p         = D_p
        self.D_e         = D_e
        self.D_h         = D_h
        self.attr        = attr
        self.dropout     = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout)
        # self.dialog_rnn  = DialogueRNN(D_m, D_g, D_p, D_e,listener_state,
        #                             context_attention, D_a, dropout_rec)
        self.dialog_rnn_f = DialogueRNNTri(D_m_T, D_m_V, D_m_A,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.dialog_rnn_r = DialogueRNNTri(D_m_T, D_m_V, D_m_A,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.postional_embedding = PositionalEncoding(D_e*6)
        self.linear      = nn.Linear(6*D_e, 3*D_h)
        self.smax_fc     = nn.Linear(3*D_h, 1)
        self.matchatt = MatchingAttention(6*D_e,6*D_e,att_type='general')
        self.multihead_attn = MultiHeadAttention(6*D_e, 6*D_e, 4)

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, Ut, Uv, Ua, qmask, umask,att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        emotions_f, alpha_f = self.dialog_rnn_f(Ut, Uv, Ua, qmask) # seq_len, batch, D_e
        emotions_f = self.dropout_rec(emotions_f)
        rev_Ut = self._reverse_seq(Ut, umask)
        rev_Uv = self._reverse_seq(Uv, umask)
        rev_Ua = self._reverse_seq(Ua, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b, alpha_b = self.dialog_rnn_r(rev_Ut, rev_Uv, rev_Ua, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        emotions_b = self.dropout_rec(emotions_b)
        emotions = torch.cat([emotions_f,emotions_b],dim=-1)
        if att2:
            
            att_emotions, alpha = self.multihead_attn(emotions, emotions, emotions)
            # alpha = list(alpha.transpose(0, 1))
            alpha = list(alpha)
            hidden = F.relu(self.linear(att_emotions))  # seq_len, batch_size, Dh*3
            '''
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
            '''
        else:
            hidden = F.relu(self.linear(emotions))
        #hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        if self.attr!=4:
            pred = (self.smax_fc(hidden).squeeze()) # seq_len, batch
        else:
            pred = (self.smax_fc(hidden).squeeze()) # seq_len, batch
        return pred.transpose(0,1).contiguous().view(-1)



class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1) # batch*seq_len, 1
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/(torch.sum(mask) + 1)
        else:
            loss = self.loss(pred*mask_, target)\
                            /(torch.sum(self.weight[target]*mask_.squeeze()) + 1)
        return loss

class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred*mask, target)/torch.sum(mask)
        return loss

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor

else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor



class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight)==type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target)\
                            /torch.sum(self.weight[target])
        return loss
