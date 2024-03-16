import torch
from torch import nn
from einops import rearrange, repeat
from torch_geometric.nn import Sequential, ChebConv, BatchNorm, GATConv, Linear
from positional_encodings.torch_encodings import PositionalEncodingPermute1D, Summer
import torch.nn.functional as F


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, node_n, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.Sconv1d = nn.Conv1d(in_channels=node_n, out_channels=node_n, kernel_size=3, padding=1)
        self.Mconv1d = nn.Conv1d(in_channels=node_n, out_channels=node_n, kernel_size=5, padding=2)
        self.Lconv1d = nn.Conv1d(in_channels=node_n, out_channels=node_n, kernel_size=7, padding=3)
        self.BN = nn.BatchNorm1d(node_n)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # short conv1d
            path1 = self.Sconv1d(x)
            att1 = attn(path1)
            att1 = self.BN(att1)

            # medium conv1d
            path2 = self.Mconv1d(x)
            att2 = attn(path2) + x
            att2 = self.BN(att2)

            # large conv1d
            path3 = self.Lconv1d(x)
            att3 = attn(path3)
            att3 = self.BN(att3)

            # addition
            sum_path = att1 + att2 + att3

            x = sum_path + x
            x = ff(x) + x

        return self.norm(x)


class GRAPH1(nn.Module):
    def __init__(self, input_feature, hidden_feature,
                 batch_size, node_n, num_class, p_dropout, leaky_c):
        super(GRAPH1, self).__init__()

        self.input_feature = input_feature
        self.hidden_feature = hidden_feature
        self.node_n = node_n
        self.batch_size = batch_size

        self.emb = nn.Linear(input_feature, hidden_feature, bias=True)

        self.conv  = ChebConv(hidden_feature, hidden_feature, K=3)
        self.conv1 = ChebConv(hidden_feature, hidden_feature, K=3)
        self.conv2 = ChebConv(hidden_feature, hidden_feature, K=3)
        self.conv3 = ChebConv(hidden_feature, hidden_feature, K=3)
        self.conv4 = ChebConv(hidden_feature, hidden_feature, K=3)
        self.conv5 = ChebConv(hidden_feature, hidden_feature, K=3)
        self.conv6 = ChebConv(hidden_feature, hidden_feature, K=3)
        self.conv7 = ChebConv(hidden_feature, hidden_feature, K=3)
        self.conv8 = ChebConv(hidden_feature, hidden_feature, K=3)


        self.gat  = GATConv(hidden_feature,hidden_feature, heads=3)
        self.gat1 = GATConv(hidden_feature,hidden_feature, heads=3)
        self.gat2 = GATConv(hidden_feature,hidden_feature, heads=3)
        self.gat3 = GATConv(hidden_feature,hidden_feature, heads=3)
        self.gat4 = GATConv(hidden_feature, hidden_feature, heads=3)
        self.gat5 = GATConv(hidden_feature, hidden_feature, heads=3)
        self.gat6 = GATConv(hidden_feature, hidden_feature, heads=3)
        self.gat7 = GATConv(hidden_feature, hidden_feature, heads=3)
        self.gat8 = GATConv(hidden_feature, hidden_feature, heads=3)

        self.seq = Sequential('x, edge_index',
                           [
                               (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                               BatchNorm(3*hidden_feature),
                               Linear(hidden_feature*3, hidden_feature),
                           ])

        self.seq1 = Sequential('x, edge_index',
                               [
                                   (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                                   BatchNorm(3*hidden_feature),
                                   Linear(hidden_feature * 3, hidden_feature),
                               ])

        self.seq2 = Sequential('x, edge_index',
                               [
                                   (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                                   BatchNorm(3*hidden_feature),
                                   Linear(hidden_feature * 3, hidden_feature),

                               ])
        self.seq3 = Sequential('x, edge_index',
                               [
                                   (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                                   BatchNorm(3*hidden_feature),
                                   Linear(hidden_feature * 3, hidden_feature),

                               ])
        self.seq4 = Sequential('x, edge_index',
                               [
                                   (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                                   BatchNorm(3*hidden_feature),
                                   Linear(hidden_feature * 3, hidden_feature),

                               ])
        self.seq5 = Sequential('x, edge_index',
                               [
                                   (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                                   BatchNorm(3 * hidden_feature),
                                   Linear(hidden_feature * 3, hidden_feature),

                               ])
        self.seq6 = Sequential('x, edge_index',
                               [
                                   (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                                   BatchNorm(3 * hidden_feature),
                                   Linear(hidden_feature * 3, hidden_feature),

                               ])
        self.seq7 = Sequential('x, edge_index',
                               [
                                   (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                                   BatchNorm(3 * hidden_feature),
                                   Linear(hidden_feature * 3, hidden_feature),

                               ])
        self.seq8 = Sequential('x, edge_index',
                               [
                                   (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                                   BatchNorm(3 * hidden_feature),
                                   Linear(hidden_feature * 3, hidden_feature),

                               ])

        self.bn1d = BatchNorm(hidden_feature)
        self.BN1d = nn.BatchNorm1d(node_n * hidden_feature)

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(hidden_feature),
            nn.Linear(hidden_feature, hidden_feature),
            nn.LayerNorm(hidden_feature),)

        self.pos_embedding = nn.Parameter(torch.randn(1, node_n+1, hidden_feature))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_feature))


        # self.act_f = nn.Tanh()
        # self.act_f = nn.LeakyReLU(leaky_c)
        self.act_f = nn.ReLU()
        self.do = nn.Dropout(p_dropout)
        self.flat = nn.Flatten(0, 1)
        self.GAT_L = nn.Linear(hidden_feature, hidden_feature)
        self.cla = nn.Linear(node_n*hidden_feature, num_class)
        # torch.Size([16, 62, 64])


        # 位置编码
        self.p_enc_1d_model = PositionalEncodingPermute1D(node_n)
        self.p_enc_1d_model_sum = Summer(PositionalEncodingPermute1D(node_n))

        self.transformer = Transformer(dim=hidden_feature,
                                       node_n=node_n,
                                       depth=6, heads=6,
                                       dim_head=128,
                                       mlp_dim=hidden_feature,
                                       dropout=0.1)

        self.MLP = nn.Sequential(        # print(att1.dtype)
            nn.Linear(node_n * hidden_feature, node_n * hidden_feature // 3),
            nn.BatchNorm1d(node_n * hidden_feature // 3),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            #
            nn.Linear(node_n * hidden_feature // 3, num_class)
        )

    def forward(self, data):
        # batch_size = 20
        sample, adj1, att1, lab, batch = data.x, data.edgeG_index, data.edgeG_attr, data.y, data.batch
        sample = sample.view(self.batch_size, self.node_n, -1)
        y = self.emb(sample).relu()

        y = self.flat(y) # torch.Size([992, 64])

        # layer 1
        y = self.conv(y, adj1)
        y = self.bn1d(y)
        y_gat = self.seq(y, adj1)
        y = y.view(self.batch_size, self.node_n, -1)
        y = self.act_f(y)
        y_F = self.do(y) # torch.Size([16, 62, 64])
        # y_F = y_F.view(self.batch_size, -1)
        y = self.flat(y_F) # torch.Size([992, 64])

        # layer 2
        y1 = self.conv1(y, adj1)
        y1 = self.bn1d(y1)
        y1_gat = self.seq1(y1, adj1)
        y1= y1.view(self.batch_size, self.node_n, -1)
        y1 = self.act_f(y1)
        y1 = self.do(y1)
        y1 = self.flat(y1)

        # layer 3
        y2 = self.conv2(y1, adj1)
        y2 = self.bn1d(y2)
        y2_gat = self.seq2(y, adj1)
        # y2 = y2 * (y_gat + y1_gat + y2_gat)
        y2 = y2.view(self.batch_size, self.node_n, -1)
        # print(y2.shape)
        y2 = self.act_f(y2+y_F)
        y2_F = self.do(y2)
        y2 = self.flat(y2_F)

        # layer 4
        y3 = self.conv3(y2, adj1)
        y3 = self.bn1d(y3)
        y3_gat = self.seq3(y3, adj1)
        y3 = y3.view(self.batch_size, self.node_n, -1)
        y3 = self.act_f(y3)
        y3 = self.do(y3)
        y3 = self.flat(y3)

        # layer 5
        y4 = self.conv4(y3, adj1)
        y4 = self.bn1d(y4)
        y4_gat = self.seq4(y4, adj1)
        y4 = y4.view(self.batch_size, self.node_n, -1)
        y4 = self.act_f(y2_F+y4)
        y4_F = self.do(y4)
        y4 = self.flat(y4_F)

        # layer 6
        y5 = self.conv5(y4, adj1)
        y5 = self.bn1d(y5)
        y5_gat = self.seq5(y5, adj1)
        y5 = y5.view(self.batch_size, self.node_n, -1)
        y5 = self.act_f(y5)
        y5 = self.do(y5)
        y5 = self.flat(y5)

        # layer 7
        y6 = self.conv6(y5, adj1)
        y6 = self.bn1d(y6)
        y6_gat = self.seq6(y6, adj1)
        y6 = y6.view(self.batch_size, self.node_n, -1)
        y6 = self.act_f(y6)
        y6_F = self.do(y6+y4_F)
        y6 = self.flat(y6_F)

        # layer 8
        y7 = self.conv7(y6, adj1)
        y7 = self.bn1d(y7)
        y7_gat = self.seq7(y7, adj1)
        y7 = y7.view(self.batch_size, self.node_n, -1)
        y7 = self.act_f(y7)
        y7 = self.do(y7)
        y7 = self.flat(y7)

        # layer 9
        y8 = self.conv7(y7, adj1)
        y8 = self.bn1d(y8)
        y8_gat = self.seq8(y8, adj1)
        y8 = y8.view(self.batch_size, self.node_n, -1)
        y8 = self.act_f(y8)
        y8= self.do(y8+y6_F)
        # y8 = self.flat(y8) # torch.Size([992, 64])

        # GAT
        y_GAT = y_gat + y1_gat + y2_gat + y3_gat + y4_gat + y5_gat + y6_gat + y7_gat + y8_gat
        G_GAT = self.act_f(y_GAT.view(self.batch_size, self.node_n, -1))  # torch.Size([16, 62, 64])
        G_out = y8 * G_GAT  # torch.Size([16, 62, 64])

        G_out = G_out.view(self.batch_size, self.node_n, -1)  # torch.Size([16, 62 * 64])
        return  G_out

class GRAPH2(nn.Module):
    def __init__(self, input_feature, hidden_feature,
                 batch_size, node_n, num_class, p_dropout, leaky_c):
        super(GRAPH2, self).__init__()

        self.input_feature = input_feature
        self.hidden_feature = hidden_feature
        self.node_n = node_n
        self.batch_size = batch_size

        self.emb = nn.Linear(input_feature, hidden_feature, bias=True)

        self.conv  = ChebConv(hidden_feature, hidden_feature, K=3)
        self.conv1 = ChebConv(hidden_feature, hidden_feature, K=3)
        self.conv2 = ChebConv(hidden_feature, hidden_feature, K=3)
        self.conv3 = ChebConv(hidden_feature, hidden_feature, K=3)
        self.conv4 = ChebConv(hidden_feature, hidden_feature, K=3)
        self.conv5 = ChebConv(hidden_feature, hidden_feature, K=3)
        self.conv6 = ChebConv(hidden_feature, hidden_feature, K=3)
        self.conv7 = ChebConv(hidden_feature, hidden_feature, K=3)
        self.conv8 = ChebConv(hidden_feature, hidden_feature, K=3)


        self.gat  = GATConv(hidden_feature,hidden_feature, heads=3)
        self.gat1 = GATConv(hidden_feature,hidden_feature, heads=3)
        self.gat2 = GATConv(hidden_feature,hidden_feature, heads=3)
        self.gat3 = GATConv(hidden_feature,hidden_feature, heads=3)
        self.gat4 = GATConv(hidden_feature, hidden_feature, heads=3)
        self.gat5 = GATConv(hidden_feature, hidden_feature, heads=3)
        self.gat6 = GATConv(hidden_feature, hidden_feature, heads=3)
        self.gat7 = GATConv(hidden_feature, hidden_feature, heads=3)
        self.gat8 = GATConv(hidden_feature, hidden_feature, heads=3)

        self.seq = Sequential('x, edge_index',
                           [
                               (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                               BatchNorm(3*hidden_feature),
                               Linear(hidden_feature*3, hidden_feature),
                           ])

        self.seq1 = Sequential('x, edge_index',
                               [
                                   (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                                   BatchNorm(3*hidden_feature),
                                   Linear(hidden_feature * 3, hidden_feature),
                               ])

        self.seq2 = Sequential('x, edge_index',
                               [
                                   (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                                   BatchNorm(3*hidden_feature),
                                   Linear(hidden_feature * 3, hidden_feature),

                               ])
        self.seq3 = Sequential('x, edge_index',
                               [
                                   (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                                   BatchNorm(3*hidden_feature),
                                   Linear(hidden_feature * 3, hidden_feature),

                               ])
        self.seq4 = Sequential('x, edge_index',
                               [
                                   (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                                   BatchNorm(3*hidden_feature),
                                   Linear(hidden_feature * 3, hidden_feature),

                               ])
        self.seq5 = Sequential('x, edge_index',
                               [
                                   (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                                   BatchNorm(3 * hidden_feature),
                                   Linear(hidden_feature * 3, hidden_feature),

                               ])
        self.seq6 = Sequential('x, edge_index',
                               [
                                   (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                                   BatchNorm(3 * hidden_feature),
                                   Linear(hidden_feature * 3, hidden_feature),

                               ])
        self.seq7 = Sequential('x, edge_index',
                               [
                                   (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                                   BatchNorm(3 * hidden_feature),
                                   Linear(hidden_feature * 3, hidden_feature),

                               ])
        self.seq8 = Sequential('x, edge_index',
                               [
                                   (GATConv(hidden_feature, hidden_feature, 3), "x, edge_index -> x"),
                                   BatchNorm(3 * hidden_feature),
                                   Linear(hidden_feature * 3, hidden_feature),

                               ])

        self.bn1d = BatchNorm(hidden_feature)
        self.BN1d = nn.BatchNorm1d(node_n * hidden_feature)

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(hidden_feature),
            nn.Linear(hidden_feature, hidden_feature),
            nn.LayerNorm(hidden_feature),)

        self.pos_embedding = nn.Parameter(torch.randn(1, node_n+1, hidden_feature))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_feature))


        # self.act_f = nn.Tanh()
        # self.act_f = nn.LeakyReLU(leaky_c)
        self.act_f = nn.ReLU()
        self.do = nn.Dropout(p_dropout)
        self.flat = nn.Flatten(0, 1)
        self.GAT_L = nn.Linear(hidden_feature, hidden_feature)
        self.cla = nn.Linear(node_n*hidden_feature, num_class)
        # torch.Size([16, 62, 64])


        # 位置编码
        self.p_enc_1d_model = PositionalEncodingPermute1D(node_n)
        self.p_enc_1d_model_sum = Summer(PositionalEncodingPermute1D(node_n))

        self.transformer = Transformer(dim=hidden_feature,
                                       node_n=node_n,
                                       depth=6, heads=6,
                                       dim_head=128,
                                       mlp_dim=hidden_feature,
                                       dropout=0.1)

        self.MLP = nn.Sequential(        # print(att1.dtype)
            nn.Linear(node_n * hidden_feature, node_n * hidden_feature // 3),
            nn.BatchNorm1d(node_n * hidden_feature // 3),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            #
            nn.Linear(node_n * hidden_feature // 3, num_class)
        )

    def forward(self, data):
        # batch_size = 20
        sample, adj1, att1, lab, batch = data.x, data.edgeR_index, data.edgeR_attr, data.y, data.batch
        sample = sample.view(self.batch_size, self.node_n, -1)
        y = self.emb(sample).relu()

        y = self.flat(y) # torch.Size([992, 64])

        # layer 1
        y = self.conv(y, adj1)
        y = self.bn1d(y)
        y_gat = self.seq(y, adj1)
        y = y.view(self.batch_size, self.node_n, -1)
        y = self.act_f(y)
        y_F = self.do(y) # torch.Size([16, 62, 64])
        # y_F = y_F.view(self.batch_size, -1)
        y = self.flat(y_F) # torch.Size([992, 64])

        # layer 2
        y1 = self.conv1(y, adj1)
        y1 = self.bn1d(y1)
        y1_gat = self.seq1(y1, adj1)
        y1= y1.view(self.batch_size, self.node_n, -1)
        y1 = self.act_f(y1)
        y1 = self.do(y1)
        y1 = self.flat(y1)

        # layer 3
        y2 = self.conv2(y1, adj1)
        y2 = self.bn1d(y2)
        y2_gat = self.seq2(y, adj1)
        # y2 = y2 * (y_gat + y1_gat + y2_gat)
        y2 = y2.view(self.batch_size, self.node_n, -1)
        # print(y2.shape)
        y2 = self.act_f(y2+y_F)
        y2_F = self.do(y2)
        y2 = self.flat(y2_F)

        # layer 4
        y3 = self.conv3(y2, adj1)
        y3 = self.bn1d(y3)
        y3_gat = self.seq3(y3, adj1)
        y3 = y3.view(self.batch_size, self.node_n, -1)
        y3 = self.act_f(y3)
        y3 = self.do(y3)
        y3 = self.flat(y3)

        # layer 5
        y4 = self.conv4(y3, adj1)
        y4 = self.bn1d(y4)
        y4_gat = self.seq4(y4, adj1)
        y4 = y4.view(self.batch_size, self.node_n, -1)
        y4 = self.act_f(y2_F+y4)
        y4_F = self.do(y4)
        y4 = self.flat(y4_F)

        # layer 6
        y5 = self.conv5(y4, adj1)
        y5 = self.bn1d(y5)
        y5_gat = self.seq5(y5, adj1)
        y5 = y5.view(self.batch_size, self.node_n, -1)
        y5 = self.act_f(y5)
        y5 = self.do(y5)
        y5 = self.flat(y5)

        # layer 7
        y6 = self.conv6(y5, adj1)
        y6 = self.bn1d(y6)
        y6_gat = self.seq6(y6, adj1)
        y6 = y6.view(self.batch_size, self.node_n, -1)
        y6 = self.act_f(y6)
        y6_F = self.do(y6+y4_F)
        y6 = self.flat(y6_F)

        # layer 8
        y7 = self.conv7(y6, adj1)
        y7 = self.bn1d(y7)
        y7_gat = self.seq7(y7, adj1)
        y7 = y7.view(self.batch_size, self.node_n, -1)
        y7 = self.act_f(y7)
        y7 = self.do(y7)
        y7 = self.flat(y7)

        # layer 9
        y8 = self.conv7(y7, adj1)
        y8 = self.bn1d(y8)
        y8_gat = self.seq8(y8, adj1)
        y8 = y8.view(self.batch_size, self.node_n, -1)
        y8 = self.act_f(y8)
        y8= self.do(y8+y6_F)
        # y8 = self.flat(y8) # torch.Size([992, 64])

        # GAT
        y_GAT = y_gat + y1_gat + y2_gat + y3_gat + y4_gat + y5_gat + y6_gat + y7_gat + y8_gat
        G_GAT = self.act_f(y_GAT.view(self.batch_size, self.node_n, -1))  # torch.Size([16, 62, 64])
        G_out = y8 * G_GAT  # torch.Size([16, 62, 64])

        G_out = G_out.view(self.batch_size, self.node_n, -1)  # torch.Size([16, 62 * 64])
        return  G_out



class full_model(nn.Module):
    def __init__(self, input_feature, hidden_feature,
                 batch_size, node_n, num_class, p_dropout, leaky_c):
        super(full_model, self).__init__()
        self.batch_size = batch_size
        self.graph1 = GRAPH1(input_feature,
                               hidden_feature,
                               batch_size,
                               node_n,
                               num_class,
                               p_dropout,
                               leaky_c
                               )
        self.graph2 = GRAPH2(input_feature,
                               hidden_feature,
                               batch_size,
                               node_n,
                               num_class,
                               p_dropout,
                               leaky_c
                               )
        self.p_enc_1d_model = PositionalEncodingPermute1D(node_n)
        self.p_enc_1d_model_sum = Summer(PositionalEncodingPermute1D(node_n))

        self.transformer = Transformer(dim=hidden_feature,
                                       node_n=node_n,
                                       depth=6, heads=6,
                                       dim_head=128,
                                       mlp_dim=hidden_feature,
                                       dropout=0.1)

        self.MLP = nn.Sequential(  # print(att1.dtype)
            nn.Linear(node_n * hidden_feature, node_n * hidden_feature // 3),
            nn.BatchNorm1d(node_n * hidden_feature // 3),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            #
            nn.Linear(node_n * hidden_feature // 3, num_class)
        )

    def forward(self, data):
        G_out = self.graph1(data)
        R_out = self.graph2(data)
        graph_out = G_out + R_out # torch.Size([16, 62, 64])

        # 位置编码 1d-sincos
        P_out1 = self.p_enc_1d_model(graph_out)
        P_out = self.p_enc_1d_model_sum(P_out1)  # 16, 62, 64

        # MT
        T_out = self.transformer(P_out)
        T_out = T_out.view(self.batch_size, -1)  # torch.Size([16, 3968])
        #

        out = self.MLP(T_out)
        return out