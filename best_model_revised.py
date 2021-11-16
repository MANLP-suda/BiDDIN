import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

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
#            scale = F.tanh(scale)
            alpha = F.softmax(scale, dim=0).permute(0,2,1) # batch, 1, seq_len
#            print ('alpha',alpha.size())
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
                att_vec_bag.append(att_vec)
            attn_pool = torch.cat(att_vec_bag, -1)

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
        e0 -> batch, self.D_e
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
#            print ('c_tta',c_tta.size())
            c_aatv = torch.cat([c_aa.unsqueeze(1),c_at.unsqueeze(1),c_av.unsqueeze(1)],1)
            c_vvta = torch.cat([c_vv.unsqueeze(1),c_vt.unsqueeze(1),c_va.unsqueeze(1)],1)
#            print ('c_aat',c_aat.size())
            
            c_t,alp1 = self.my_self_att1(c_ttav)
#            print ('c_t',c_t.size())
            c_t = self.dense1(c_t)
            
            c_a,alp2 = self.my_self_att2(c_aatv)
#            print ('c_a',c_a.size())
            c_a = self.dense2(c_a)
            
            c_v,alp3 = self.my_self_att3(c_vvta)

            c_v = self.dense3(c_v)

        
        q_t, e_t = self.rnn_cell(Ut,c_t,qmask,qm_idx,q0_t,e0_t,self.p_cell_t,self.e_cell_t)
        q_a, e_a = self.rnn_cell(Ua,c_a,qmask,qm_idx,q0_a,e0_a,self.p_cell_a,self.e_cell_a)
        q_v, e_v = self.rnn_cell(Uv,c_v,qmask,qm_idx,q0_v,e0_v,self.p_cell_v,self.e_cell_v)
        

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
            if type(alpha_)!=type(None):
                alpha.append(alpha_[:,0,:])

        
        e = torch.cat([et, ev, ea],dim = -1)
#        print ('e', e.size())
        score_t = torch.cat([et.unsqueeze(2), ev.unsqueeze(2), ea.unsqueeze(2)],dim = 2)
#        print ('score1', score_t.size())
        score_t = self.self_attention(score_t)
#        print ('score2', score_t.size())
        score_t = F.softmax(score_t,dim=-1)
#        print ('score3', score_t.size())
        score_t = score_t.squeeze(-1)
#        print ('score4', score_t.size())
        score_t = score_t.repeat(1,1,1,self.D_e).view(len(e),len(e[0]),-1)

        e_temp = torch.mul(e,score_t)
        e = e + e_temp
#        print ('e', e.size())


        

        return e,alpha # seq_len, batch, D_e
        
class OurBiModelTri(nn.Module):

    def __init__(self, D_m_T,D_m_A,D_m, D_g, D_p, D_e, D_h,
                 n_classes=7, listener_state=False, context_attention='simple', D_a=100, dropout_rec=0.5,
                 dropout=0.5):
        super(OurBiModel, self).__init__()


        self.D_m_T     = D_m_T
        self.D_m_A     = D_m_A
        self.D_m       = D_m
        self.D_g       = D_g
        self.D_p       = D_p
        self.D_e       = D_e
        self.D_h       = D_h
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout+0.15)
        self.dialog_rnn_f = DialogueRNN(D_m_T,D_m_A,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.dialog_rnn_r = DialogueRNN(D_m_T,D_m_A,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.linear     = nn.Linear(4*D_e, 2*D_h)
        self.smax_fc    = nn.Linear(2*D_h, n_classes)
        self.matchatt = MatchingAttention(4*D_e,4*D_e,att_type='general2')

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


    def forward(self, Ut,Ua, qmask, umask,att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        emotions_f, alpha_f = self.dialog_rnn_f(Ut,Ua, qmask) # seq_len, batch, D_e
        emotions_f = self.dropout_rec(emotions_f)
        rev_Ut = self._reverse_seq(Ut, umask)
        rev_Ua = self._reverse_seq(Ua, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b, alpha_b = self.dialog_rnn_r(rev_Ut,rev_Ua, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        emotions_b = self.dropout_rec(emotions_b)
        emotions = torch.cat([emotions_f,emotions_b],dim=-1)
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

        return log_prob, alpha, alpha_f, alpha_b

class IEMOCAPBiModelTri(nn.Module):

    def __init__(self, D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, D_h, listener_state=False,
            context_attention='simple', D_a=100, dropout_rec=0.5, dropout=0.5):
        super(IEMOCAPBiModelTri, self).__init__()

        self.D_m         = D_m
        self.D_g         = D_g
        self.D_p         = D_p
        self.D_e         = D_e
        self.D_h         = D_h
        self.dropout     = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout)
        # self.dialog_rnn  = DialogueRNN(D_m, D_g, D_p, D_e,listener_state,
        #                             context_attention, D_a, dropout_rec)
        self.dialog_rnn_f = DialogueRNNTri(D_m_T, D_m_V, D_m_A,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.dialog_rnn_r = DialogueRNNTri(D_m_T, D_m_V, D_m_A,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.linear      = nn.Linear(6*D_e, 3*D_h)
        self.smax_fc     = nn.Linear(3*D_h, 1)
        self.matchatt = MatchingAttention(6*D_e,6*D_e,att_type='general')

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
        
        pred = (self.smax_fc(hidden).squeeze()) # seq_len, batch
        return pred.transpose(0,1).contiguous().view(-1)
        
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
        self.linear      = nn.Linear(6*D_e, 3*D_h)
        self.smax_fc     = nn.Linear(3*D_h, 1)
        self.matchatt = MatchingAttention(6*D_e,6*D_e,att_type='general')

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
        if self.attr!=4:
            pred = (self.smax_fc(hidden).squeeze()) # seq_len, batch
        else:
            pred = (self.smax_fc(hidden).squeeze()) # seq_len, batch
        return pred.transpose(0,1).contiguous().view(-1)

class DialogueRNNCell(nn.Module):

    def __init__(self, D_m_T, D_m_A, D_m, D_g, D_p, D_e, listener_state=False,
                            context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNNCell, self).__init__()

        self.D_m_T = D_m_T
        self.D_m_A = D_m_A
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
        self.dense_t = nn.Linear(D_m_T,D_m)
        self.dense_a = nn.Linear(D_m_A,D_m)
        self.dense_u = nn.Linear(D_m,D_m)
        self.self_attention = nn.Linear(D_g,1,bias=True)
        self.my_self_att1 = SelfAttention(D_g,att_type = 'general2')
        self.my_self_att2 = SelfAttention(D_g,att_type = 'general2')
        self.dense1 = nn.Linear(self.D_g*2,self.D_g,bias=True)
        self.dense2 = nn.Linear(self.D_g*2,self.D_g,bias=True)
        if listener_state:
            self.l_cell = nn.GRUCell(D_m+D_p,D_p)

        self.dropout = nn.Dropout(dropout)
        
        if context_attention=='simple':
            
            self.attention = SimpleAttention(D_g)
        else:
            
            self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)
        
#        self.attention = []
        if context_attention=='simple':
            self.attention1 = SimpleAttention(D_g)
            self.attention2 = SimpleAttention(D_g)
            self.attention3 = SimpleAttention(D_g)
            self.attention4 = SimpleAttention(D_g)
#            for _ in range(4):
#                self.attention.append(SimpleAttention(D_g))
        else:
            self.attention1 = MatchingAttention(D_g, D_m, D_a, context_attention)
            self.attention2 = MatchingAttention(D_g, D_m, D_a, context_attention)
            self.attention3 = MatchingAttention(D_g, D_m, D_a, context_attention)
            self.attention4 = MatchingAttention(D_g, D_m, D_a, context_attention)
#            for _ in range(4):
#                self.attention.append(MatchingAttention(D_g, D_m, D_a, context_attention))

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

    def forward(self, Ut, Ua, qmask, g_hist_t, g_hist_a, q0_t, q0_a, e0_t, e0_a,k=1):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        """
        # pad_t = torch.zeros(Ut.size()[0],Ut.size()[1]).type(Ua.type())
        # pad_a = torch.zeros(Ua.size()[0],Ua.size()[1]).type(Ua.type())

        # Ut = torch.cat([Ut,pad_a],dim = -1)
        # Ua = torch.cat([pad_t,Ua],dim = -1)

        Ut = self.dense_t(Ut)
        Ua = self.dense_a(Ua)
        # Ut = self.dense_u(Ut)
        # Ua = self.dense_u(Ua)

        qm_idx = torch.argmax(qmask, 1)
        q0_sel_t = self._select_parties(q0_t, qm_idx)
        q0_sel_a = self._select_parties(q0_a, qm_idx)

        g_t = self.g_cell_t(torch.cat([Ut,q0_sel_t], dim=1),
                torch.zeros(Ut.size()[0],self.D_g).type(Ut.type()) if g_hist_t.size()[0]==0 else
                g_hist_t[-1])
        g_t = self.dropout(g_t)

        g_a = self.g_cell_a(torch.cat([Ua,q0_sel_a], dim=1),
                torch.zeros(Ua.size()[0],self.D_g).type(Ua.type()) if g_hist_a.size()[0]==0 else
                g_hist_a[-1])
        g_t = self.dropout(g_t)
        g_a = self.dropout(g_a)
        if g_hist_t.size()[0]==0:
            c_t = torch.zeros(Ut.size()[0],self.D_g).type(Ut.type())
            alpha = None
        if g_hist_a.size()[0]==0:
            c_a = torch.zeros(Ua.size()[0],self.D_g).type(Ua.type())
            alpha = None
        else:
            # c_tt, alpha_tt = self.attention(g_hist_t[:,-2:],Ut)
            # c_at, alpha_at = self.attention(g_hist_t[:,-2:],Ua)
            # c_ta, alpha_ta = self.attention(g_hist_a[:,-2:],Ut)
            # c_aa, alpha_aa = self.attention(g_hist_a[:,-2:],Ua)
            
            c_tt, alpha_tt = self.attention1(g_hist_t,Ut)
            c_at, alpha_at = self.attention2(g_hist_t,Ua)
            c_ta, alpha_ta = self.attention3(g_hist_a,Ut)
            c_aa, alpha_aa = self.attention4(g_hist_a,Ua)
            
#            c_tt, alpha_tt = self.attention(g_hist_t,Ut)
#            c_at, alpha_at = self.attention(g_hist_t,Ua)
#            c_ta, alpha_ta = self.attention(g_hist_a,Ut)
#            c_aa, alpha_aa = self.attention(g_hist_a,Ua)
            
            alpha = alpha_tt + alpha_ta + alpha_at + alpha_aa

            # score_t = torch.cat([c_tt.unsqueeze(1),c_ta.unsqueeze(1)],dim=1)
            # score_t = self.self_attention(score_t)
            # score_t = F.softmax(score_t,dim=-1)
            # score_t = score_t.repeat(1,1,self.D_g)
            # score_tt = score_t[:,0,:].squeeze()
            # score_ta = score_t[:,1,:].squeeze()

            # score_a = torch.cat([c_at.unsqueeze(1),c_aa.unsqueeze(1)],dim=1)
            # score_a = self.self_attention(score_a)
            # score_a = F.softmax(score_a,dim=-1)
            # score_a = score_a.repeat(1,1,self.D_g)
            # score_at = score_a[:,0,:].squeeze()
            # score_aa = score_a[:,1,:].squeeze()


            # c_t = torch.mul(c_tt,score_tt) + torch.mul(c_ta,score_ta) 
            # c_a = torch.mul(c_at,score_at) + torch.mul(c_aa,score_aa) 
            
#            c_t = c_tt + c_ta
#            c_a = c_aa + c_at
            
            c_tta = torch.cat([c_tt.unsqueeze(1),c_ta.unsqueeze(1)],1)
#            print ('c_tta',c_tta.size())
            c_aat = torch.cat([c_aa.unsqueeze(1),c_at.unsqueeze(1)],1)
#            print ('c_aat',c_aat.size())
            
            c_t,alp1 = self.my_self_att1(c_tta)
#            print ('c_t',c_t.size())
            c_t = self.dense1(c_t)
            c_a,alp2 = self.my_self_att2(c_aat)
#            print ('c_a',c_a.size())
            c_a = self.dense2(c_a)

        
        q_t, e_t = self.rnn_cell(Ut,c_t,qmask,qm_idx,q0_t,e0_t,self.p_cell_t,self.e_cell_t)
        q_a, e_a = self.rnn_cell(Ua,c_a,qmask,qm_idx,q0_a,e0_a,self.p_cell_a,self.e_cell_a)
        

        return g_t,q_t,e_t,g_a,q_a,e_a,alpha

class DialogueRNN(nn.Module):

    def __init__(self, D_m_T, D_m_A,D_m, D_g, D_p, D_e, listener_state=False,
                            context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = DialogueRNNCell(D_m_T, D_m_A,D_m, D_g, D_p, D_e,
                            listener_state, context_attention, D_a, dropout)
        self.self_attention = nn.Linear(D_e,1,bias=True)

    def forward(self, Ut, Ua, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        g_hist_t = torch.zeros(0).type(Ut.type()) # 0-dimensional tensor
        g_hist_a = torch.zeros(0).type(Ua.type()) # 0-dimensional tensor
        q_t = torch.zeros(qmask.size()[1], qmask.size()[2],
                                    self.D_p).type(Ut.type()) # batch, party, D_p
        e_t = torch.zeros(0).type(Ut.type()) # batch, D_e
        et = e_t

        q_a = torch.zeros(qmask.size()[1], qmask.size()[2],
                                    self.D_p).type(Ua.type()) # batch, party, D_p
        e_a = torch.zeros(0).type(Ua.type()) # batch, D_e
        ea = e_a

        alpha = []
        for u_t,u_a,qmask_ in zip(Ut,Ua, qmask):
            g_t,q_t,e_t,g_a,q_a,e_a,alpha_ = self.dialogue_cell(u_t,u_a, qmask_, g_hist_t,g_hist_a, q_t,q_a, e_t,e_a,k=5)
            g_hist_t = torch.cat([g_hist_t, g_t.unsqueeze(0)],0)
            g_hist_a = torch.cat([g_hist_a, g_a.unsqueeze(0)],0)
            et = torch.cat([et, e_t.unsqueeze(0)],0)
            # et = torch.cat([e_t.unsqueeze(0),et],0)
            ea = torch.cat([ea, e_a.unsqueeze(0)],0)
            if type(alpha_)!=type(None):
                alpha.append(alpha_[:,0,:])

        
        e = torch.cat([et, ea],dim = -1)
        score_t = torch.cat([et.unsqueeze(2), ea.unsqueeze(2)],dim = 2)
        score_t = self.self_attention(score_t)
        score_t = F.softmax(score_t,dim=-1)
        score_t = score_t.repeat(1,1,1,self.D_e).view(len(e),len(e[0]),-1)
        
#        print ('score',score_t.size())

        e_temp = torch.mul(e,score_t)
        e = e + e_temp
#        print ('e', e.size())


        

        return e,alpha # seq_len, batch, D_e

class AVECModel(nn.Module):

    def __init__(self, D_m_T,D_m_A,D_m, D_g, D_p, D_e, D_h, attr, listener_state=False,
            context_attention='simple', D_a=100, dropout_rec=0.5, dropout=0.5):
        super(AVECModel, self).__init__()

        # self.D_m_T     = D_m_T
        # self.D_m_A     = D_m_A

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
        self.dialog_rnn = DialogueRNN(D_m_T,D_m_A,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.linear      = nn.Linear(D_e, D_h)
        self.smax_fc     = nn.Linear(D_h, 1)


    def forward(self, Ut,Ua, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        emotions,_ = self.dialog_rnn(Ut,Ua, qmask) # seq_len, batch, D_e
        emotions = self.dropout_rec(emotions)
        hidden = torch.tanh(self.linear(emotions))
        hidden = self.dropout(hidden)
        if self.attr!=4:
            pred = (self.smax_fc(hidden).squeeze()) # seq_len, batch
        else:
            pred = (self.smax_fc(hidden).squeeze()) # seq_len, batch
        return pred.transpose(0,1).contiguous().view(-1)

class AVECBiModel(nn.Module):

    def __init__(self, D_m_T, D_m_A, D_m, D_g, D_p, D_e, D_h, attr, listener_state=False,
            context_attention='simple', D_a=100, dropout_rec=0.5, dropout=0.5):
        super(AVECBiModel, self).__init__()

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
        self.dialog_rnn_f = DialogueRNN(D_m_T,D_m_A,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.dialog_rnn_r = DialogueRNN(D_m_T,D_m_A,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.linear      = nn.Linear(4*D_e, 2*D_h)
        self.smax_fc     = nn.Linear(2*D_h, 1)
        self.matchatt = MatchingAttention(4*D_e,4*D_e,att_type='general2')

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

    def forward(self, Ut,Ua, qmask, umask,att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        emotions_f, alpha_f = self.dialog_rnn_f(Ut,Ua, qmask) # seq_len, batch, D_e
        emotions_f = self.dropout_rec(emotions_f)
        rev_Ut = self._reverse_seq(Ut, umask)
        rev_Ua = self._reverse_seq(Ua, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b, alpha_b = self.dialog_rnn_r(rev_Ut,rev_Ua, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        emotions_b = self.dropout_rec(emotions_b)
        emotions = torch.cat([emotions_f,emotions_b],dim=-1)
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
        if self.attr!=4:
            pred = (self.smax_fc(hidden).squeeze()) # seq_len, batch
        else:
            pred = (self.smax_fc(hidden).squeeze()) # seq_len, batch
        return pred.transpose(0,1).contiguous().view(-1)


        # emotions,_ = self.dialog_rnn(U, qmask) # seq_len, batch, D_e
        # emotions = self.dropout_rec(emotions)
        # hidden = torch.tanh(self.linear(emotions))
        # hidden = self.dropout(hidden)
        # if self.attr!=4:
        #     pred = (self.smax_fc(hidden).squeeze()) # seq_len, batch
        # else:
        #     pred = (self.smax_fc(hidden).squeeze()) # seq_len, batch
        # return pred.transpose(0,1).contiguous().view(-1)
        
class OurBiModel(nn.Module):

    def __init__(self, D_m_T,D_m_A,D_m, D_g, D_p, D_e, D_h,
                 n_classes=7, listener_state=False, context_attention='simple', D_a=100, dropout_rec=0.5,
                 dropout=0.5):
        super(OurBiModel, self).__init__()


        self.D_m_T     = D_m_T
        self.D_m_A     = D_m_A
        self.D_m       = D_m
        self.D_g       = D_g
        self.D_p       = D_p
        self.D_e       = D_e
        self.D_h       = D_h
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout+0.15)
        self.dialog_rnn_f = DialogueRNN(D_m_T,D_m_A,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.dialog_rnn_r = DialogueRNN(D_m_T,D_m_A,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.linear     = nn.Linear(4*D_e, 2*D_h)
        self.smax_fc    = nn.Linear(2*D_h, n_classes)
        self.matchatt = MatchingAttention(4*D_e,4*D_e,att_type='general2')

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


    def forward(self, Ut,Ua, qmask, umask,att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        emotions_f, alpha_f = self.dialog_rnn_f(Ut,Ua, qmask) # seq_len, batch, D_e
        emotions_f = self.dropout_rec(emotions_f)
        rev_Ut = self._reverse_seq(Ut, umask)
        rev_Ua = self._reverse_seq(Ua, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b, alpha_b = self.dialog_rnn_r(rev_Ut,rev_Ua, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        emotions_b = self.dropout_rec(emotions_b)
        emotions = torch.cat([emotions_f,emotions_b],dim=-1)
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

        return log_prob, alpha, alpha_f, alpha_b

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

class CNNFeatureExtractor(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, output_size, filters, kernel_sizes, dropout):
        super(CNNFeatureExtractor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=K) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * filters, output_size)
        self.feature_dim = output_size


    def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        # if is_static:
        self.embedding.weight.requires_grad = False


    def forward(self, x, umask):
        
        num_utt, batch, num_words = x.size()
        
        x = x.type(LongTensor)  # (num_utt, batch, num_words)
        x = x.view(-1, num_words) # (num_utt, batch, num_words) -> (num_utt * batch, num_words)
        emb = self.embedding(x) # (num_utt * batch, num_words) -> (num_utt * batch, num_words, 300) 
        emb = emb.transpose(-2, -1).contiguous() # (num_utt * batch, num_words, 300)  -> (num_utt * batch, 300, num_words) 
        
        convoluted = [F.relu(conv(emb)) for conv in self.convs] 
        pooled = [F.max_pool1d(c, c.size(2)).squeeze() for c in convoluted] 
        concated = torch.cat(pooled, 1)
        features = F.relu(self.fc(self.dropout(concated))) # (num_utt * batch, 150) -> (num_utt * batch, 100)
        features = features.view(num_utt, batch, -1) # (num_utt * batch, 100) -> (num_utt, batch, 100)
        mask = umask.unsqueeze(-1).type(FloatTensor) # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1) # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(1, 1, self.feature_dim) #  (num_utt, batch, 1) -> (num_utt, batch, 100)
        features = (features * mask) # (num_utt, batch, 100) -> (num_utt, batch, 100)

        return features

class DailyDialogueModel(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, D_h,
                 vocab_size, n_classes=7, embedding_dim=300, 
                 cnn_output_size=100, cnn_filters=50, cnn_kernel_sizes=(3,4,5), cnn_dropout=0.5,
                 listener_state=False, context_attention='simple', D_a=100, dropout_rec=0.5,
                 dropout=0.5, att2=True):
        
        super(DailyDialogueModel, self).__init__()

        self.cnn_feat_extractor = CNNFeatureExtractor(vocab_size, embedding_dim, cnn_output_size, cnn_filters, cnn_kernel_sizes, cnn_dropout)
                
        self.D_m       = D_m
        self.D_g       = D_g
        self.D_p       = D_p
        self.D_e       = D_e
        self.D_h       = D_h
        self.dropout   = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout_rec)
        self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                    context_attention, D_a, dropout_rec)
        self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                    context_attention, D_a, dropout_rec)
        self.linear     = nn.Linear(2*D_e, 2*D_h)
        self.matchatt = MatchingAttention(2*D_e,2*D_e,att_type='general2')

        self.n_classes = n_classes
        self.smax_fc    = nn.Linear(2*D_h, n_classes)
        self.att2 = att2

        
    
    def init_pretrained_embeddings(self, pretrained_word_vectors):
        self.cnn_feat_extractor.init_pretrained_embeddings_from_numpy(pretrained_word_vectors)


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


    def forward(self, input_seq, qmask, umask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        U = self.cnn_feat_extractor(input_seq, umask)

        emotions_f, alpha_f = self.dialog_rnn_f(U, qmask) # seq_len, batch, D_e
        emotions_f = self.dropout_rec(emotions_f)
        rev_U = self._reverse_seq(U, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        emotions_b = self.dropout_rec(emotions_b)
        emotions = torch.cat([emotions_f, emotions_b], dim=-1)
        if self.att2:
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
        # hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2) # seq_len, batch, n_classes
        return log_prob, alpha, alpha_f, alpha_b

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


class OurModel(nn.Module):

    def __init__(self, D_m_T,D_m_A, D_g, D_p, D_e, D_h,
                 n_classes=7, listener_state=False, context_attention='simple', D_a=100, dropout_rec=0.5,
                 dropout=0.5):
        super(OurModel, self).__init__()

        self.D_m_T       = D_m_T
        self.D_m_A       = D_m_A
        self.D_g       = D_g
        self.D_p       = D_p
        self.D_e       = D_e
        self.D_h       = D_h
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout+0.15)
        self.dialog_rnn_f_T = DialogueRNN(D_m_T, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.dialog_rnn_r_T = DialogueRNN(D_m_T, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)

        self.dialog_rnn_f_A = DialogueRNN(D_m_A, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.dialog_rnn_r_A = DialogueRNN(D_m_A, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.linear     = nn.Linear(2*D_e, 2*D_h)
        self.smax_fc    = nn.Linear(8*D_h, n_classes)
        self.matchatt = MatchingAttention(2*D_e,2*D_e,att_type='general2')
        self.transform1 = nn.Linear(2*D_h,2*D_h,bias=False)
        self.transform2 = nn.Linear(2*D_h,1,bias=True)

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


    def forward(self, T,A, qmask, umask,att2=True,k=4):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        Alpha_f = []
        Alpha_b = []
        emotions_f_T, alpha_f = self.dialog_rnn_f_T(T, qmask) # seq_len, batch, D_e
        emotions_f_T = self.dropout_rec(emotions_f_T)
        rev_U = self._reverse_seq(T, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b_T, alpha_b = self.dialog_rnn_r_T(rev_U, rev_qmask)
        emotions_b_T = self._reverse_seq(emotions_b_T, umask)
        emotions_b_T = self.dropout_rec(emotions_b_T)
        emotions_T = torch.cat([emotions_f_T,emotions_b_T],dim=-1)
        Alpha_f += alpha_f
        Alpha_b += alpha_b

        emotions_f_A, alpha_f = self.dialog_rnn_f_A(A, qmask) # seq_len, batch, D_e
        emotions_f_A = self.dropout_rec(emotions_f_A)
        rev_U = self._reverse_seq(A, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b_A, alpha_b = self.dialog_rnn_r_A(rev_U, rev_qmask)
        emotions_b_A = self._reverse_seq(emotions_b_A, umask)
        emotions_b_A = self.dropout_rec(emotions_b_A)
        emotions_A = torch.cat([emotions_f_A,emotions_b_A],dim=-1)
        Alpha_f += alpha_f
        Alpha_b += alpha_b


        alpha = []

        # if att2:
        #     att_emotions = []
        #     for t in emotions_T:
        #         att_em, alpha_ = self.matchatt(emotions_T,t,mask=umask)
        #         att_emotions.append(att_em.unsqueeze(0))
        #         alpha.append(alpha_[:,0,:])
        #     att_emotions = torch.cat(att_emotions,dim=0)
        #     hidden_TtoT = F.tanh(self.linear(att_emotions))
        # else:
        #     hidden_TtoT = F.tanh(self.linear(emotions_T))

        if att2:
            att_emotions = []
            for t in range(len(emotions_T)):
                a = t - k // 2 if t - k // 2 >= 0 else 0
                b = t + k // 2 if t + k // 2 <= len(emotions_T) else len(emotions_T)
                
                att_em, alpha_ = self.matchatt(emotions_T[a:b],emotions_T[t])
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden_TtoT = F.relu(self.linear(att_emotions))
        else:
            hidden_TtoT = F.relu(self.linear(emotions_T))

        if att2:
            att_emotions = []
            for t in range(len(emotions_A)):
                a = t - k // 2 if t - k // 2 >= 0 else 0
                b = t + k // 2 if t + k // 2 <= len(emotions_A) else len(emotions_A)
                att_em, alpha_ = self.matchatt(emotions_T[a:b],emotions_A[t])
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden_AtoT = F.relu(self.linear(att_emotions))
        else:
            hidden_AtoT = F.relu(self.linear(emotions_T))

        if att2:
            att_emotions = []
            for t in range(len(emotions_A)):
                a = t - k // 2 if t - k // 2 >= 0 else 0
                b = t + k // 2 if t + k // 2 <= len(emotions_A) else len(emotions_A)
                att_em, alpha_ = self.matchatt(emotions_A[a:b],emotions_A[t])
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden_TtoA = F.relu(self.linear(att_emotions))
        else:
            hidden_TtoA = F.relu(self.linear(emotions_A))

        if att2:
            att_emotions = []
            for t in range(len(emotions_A)):
                a = t - k // 2 if t - k // 2 >= 0 else 0
                b = t + k // 2 if t + k // 2 <= len(emotions_T) else len(emotions_T)
                att_em, alpha_ = self.matchatt(emotions_T[a:b],emotions_A[t])
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden_AtoA = F.relu(self.linear(att_emotions))
        else:
            hidden_AtoA = F.relu(self.linear(emotions_A))

        hidden = torch.cat([hidden_AtoA,hidden_TtoT,hidden_AtoT,hidden_TtoA],dim=-1)

        new_hidden = hidden.view([len(emotions_T),len(emotions_T[0]),4,-1])

        new_hidden2 = nn.Tanh()(new_hidden)
        new_att = self.transform2(new_hidden2)
        score = F.softmax(new_att,dim=2)

        score = score.repeat(1,1,1,200).view(len(emotions_T),len(emotions_T[0]),-1)

        hidden = torch.mul(hidden,score)

        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2) # seq_len, batch, n_classes

        return log_prob, alpha, Alpha_f, Alpha_b



# class DialogueRNNCell(nn.Module):

#     def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
#                             context_attention='simple', D_a=100, dropout=0.5):
#         super(DialogueRNNCell, self).__init__()

#         self.D_m = D_m
#         self.D_g = D_g
#         self.D_p = D_p
#         self.D_e = D_e

#         self.listener_state = listener_state
#         self.g_cell = nn.GRUCell(D_m+D_p,D_g)
#         self.p_cell = nn.GRUCell(D_m+D_g,D_p)
#         self.e_cell = nn.GRUCell(D_p,D_e)
#         if listener_state:
#             self.l_cell = nn.GRUCell(D_m+D_p,D_p)

#         self.dropout = nn.Dropout(dropout)

#         if context_attention=='simple':
#             self.attention = SimpleAttention(D_g)
#         else:
#             self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)

#     def _select_parties(self, X, indices):
#         q0_sel = []
#         for idx, j in zip(indices, X):
#             q0_sel.append(j[idx].unsqueeze(0))
#         q0_sel = torch.cat(q0_sel,0)
#         return q0_sel

#     def forward(self, U, qmask, g_hist, q0, e0):
#         """
#         U -> batch, D_m
#         qmask -> batch, party
#         g_hist -> t-1, batch, D_g
#         q0 -> batch, party, D_p
#         e0 -> batch, self.D_e
#         """
#         qm_idx = torch.argmax(qmask, 1)
#         q0_sel = self._select_parties(q0, qm_idx)

#         g_ = self.g_cell(torch.cat([U,q0_sel], dim=1),
#                 torch.zeros(U.size()[0],self.D_g).type(U.type()) if g_hist.size()[0]==0 else
#                 g_hist[-1])
#         g_ = self.dropout(g_)
#         if g_hist.size()[0]==0:
#             c_ = torch.zeros(U.size()[0],self.D_g).type(U.type())
#             alpha = None
#         else:
#             c_, alpha = self.attention(g_hist,U)
#         # c_ = torch.zeros(U.size()[0],self.D_g).type(U.type()) if g_hist.size()[0]==0\
#         #         else self.attention(g_hist,U)[0] # batch, D_g
#         U_c_ = torch.cat([U,c_], dim=1).unsqueeze(1).expand(-1,qmask.size()[1],-1)
#         qs_ = self.p_cell(U_c_.contiguous().view(-1,self.D_m+self.D_g),
#                 q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
#         qs_ = self.dropout(qs_)

#         if self.listener_state:
#             U_ = U.unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_m)
#             ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).\
#                     expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_p)
#             U_ss_ = torch.cat([U_,ss_],1)
#             ql_ = self.l_cell(U_ss_,q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
#             ql_ = self.dropout(ql_)
#         else:
#             ql_ = q0
#         qmask_ = qmask.unsqueeze(2)
#         q_ = ql_*(1-qmask_) + qs_*qmask_
#         e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0]==0\
#                 else e0
#         e_ = self.e_cell(self._select_parties(q_,qm_idx), e0)
#         e_ = self.dropout(e_)

#         return g_,q_,e_,alpha

# class DialogueRNN(nn.Module):

#     def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
#                             context_attention='simple', D_a=100, dropout=0.5):
#         super(DialogueRNN, self).__init__()

#         self.D_m = D_m
#         self.D_g = D_g
#         self.D_p = D_p
#         self.D_e = D_e
#         self.dropout = nn.Dropout(dropout)

#         self.dialogue_cell = DialogueRNNCell(D_m, D_g, D_p, D_e,
#                             listener_state, context_attention, D_a, dropout)

#     def forward(self, U, qmask):
#         """
#         U -> seq_len, batch, D_m
#         qmask -> seq_len, batch, party
#         """

#         g_hist = torch.zeros(0).type(U.type()) # 0-dimensional tensor
#         q_ = torch.zeros(qmask.size()[1], qmask.size()[2],
#                                     self.D_p).type(U.type()) # batch, party, D_p
#         e_ = torch.zeros(0).type(U.type()) # batch, D_e
#         e = e_

#         alpha = []
#         for u_,qmask_ in zip(U, qmask):
#             g_, q_, e_, alpha_ = self.dialogue_cell(u_, qmask_, g_hist, q_, e_)
#             g_hist = torch.cat([g_hist, g_.unsqueeze(0)],0)
#             e = torch.cat([e, e_.unsqueeze(0)],0)
#             if type(alpha_)!=type(None):
#                 alpha.append(alpha_[:,0,:])

#         return e,alpha # seq_len, batch, D_e