import random
import torch
import torch.nn as nn


#input : 3 coordinates Per N Points, distance K
#output : kNN distance of N points (NxK) kNN indicies of N Points NxK
def find_knn(point_cloud, k):
    N = len(point_cloud)
    delta = point_cloud.view(N, 1, 3) - point_cloud.view(1, N, 3) #NxNx3이됨 by broadcasting
    dist = torch.sum(delta**2, dim=-1) #NxN

    knn_dist, knn_indicies = dist.topk(k=k, dim=-1, largest=False)

    return knn_dist, knn_indicies

#when query set and point set are not same
#input : dataset point Nx3, Query Point Mx3, K
#output : kNN distance MxK, kNN indicies MxK
def find_knn_general(query_points, dataset_points, k):
    M = len(query_points)
    N = len(dataset_points)

    delta = query_points.view(M, 1, 3) - dataset_points.view(1, N, 3) #MxNx3
    dist = torch.sum(delta**2, dim=-1) #MxN

    knn_dist, knn_indicies = dist.topk(k=k, dim=-1, largest=False) #M,K

    return knn_dist, knn_indicies
##
#input : dataset (Nx3), corresponding feature (NxC), query (Mx3), K
def interpolate_knn(query_points, dataset_points, dataset_features, k):
    M = len(query_points)
    N, C = dataset_features.shape

    knn_dist, knn_indices = find_knn_general(query_points, dataset_points, k)
    knn_dataset_features = dataset_features[knn_indices.view(-1)].view(M, k, C) #->(M*KxC)->MxKxC

    #calculate interpolation weights
    knn_dist_recip = 1. / (knn_dist + 1e-8) #MxK
    denom = knn_dist_recip.sum(dim=-1, keepdim=True) #Mx1
    weights = knn_dist_recip / denom #MxK

    #Linear Interpolation
    weighted_features = weights.view(M,k,1) * knn_dataset_features #MxKxC
    interpolated_features = weighted_features.sum(dim=1) #MxC

    return interpolated_features
##
#input : point Nx3, number of sample M
#output : sampled_indicies (M,)
def fps(points, num_samples):
    N = len(points)
    sampled_indicies = torch.zeros(num_samples, dtype=torch.long) #init
    distance = torch.ones(N,) * 1e10
    farthest_idx = random.randint(0, N)

    for i in range(num_samples):
        #sample farthest point
        sampled_indicies[i] = farthest_idx
        centroid = points[farthest_idx].view(1,3)
        #compute distance
        delta = points - centroid
        dist = torch.sum(delta**2, dim=-1) #N,

        mask = dist < distance #중복계산을 피하기 위해 -> fps는 가장 먼 거리부터 샘플링하므로 남은 점들은 현재 가장 먼 점보다 가까울 수 밖에 없다. 하나씩 줄어드는 것
        distance[mask] = dist[mask]
        #sample the next farthest
        farthest_idx = torch.max(distance, -1)[1] #maximum 값의 index

    return sampled_indicies

#input : dataset points (N,3) query points (M,3), Radius R
#output : indices, list (kNN과 다르게 Radius안에 있는 모든 점을 반환하므로 list로 반환)

def find_radius_general(query_points, dataset_points, r):
    M = len(query_points)
    N = len(dataset_points)

    delta = query_points.view(M, 1, 3) - dataset_points.view(1, N, 3) #(M, N, 3)
    dist = torch.sum(delta ** 2, dim=-1) #MxN

    mask = dist < r ** 2
    indicies = []
    for mask_ in mask:
        indicies.append(torch.nonzero(mask_, as_tuple=True)[0]) #index를 가져오기 위함

    return indicies

#input : point Nx3, feature NxCin, K
#output : output feature NxCout
class PointTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.linear_q = nn.Linear(in_channels, out_channels, bias=False)
        self.linear_k = nn.Linear(in_channels, out_channels, bias=False)
        self.linaer_v = nn.Linear(in_channels, out_channels, bias=False)

        self.mlp_attn = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )

        self.mlp_pos = nn.Sequential(
            nn.Linear(3, 3, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_channels, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, points, features):
        #Q, K, V
        N = len(points)
        f_q = self.linear_q(features) #NxCout
        f_k = self.linear_k(features)
        f_v = self.linaer_v(features)
        #kNN
        knn_dist, knn_indices = find_knn_general(points, points, self.k) #NxK
        knn_points = points[knn_indices.view(-1)].view(N, self.k, 3) #N*Kx1
        knn_k = f_k[knn_indices.view(-1)].view(N, self.k, self.out_channels)
        knn_v = f_v[knn_indices.view(-1)].view(N, self.k, self.out_channels)
        #position
        rel_pos = points.view(N, 1, 3) - knn_points #NxKx3
        rel_pos_enc = self.mlp_pos(rel_pos.view(-1, 3)).view(N, self.k, self.out_channels)#N*Kx3 since we use BN1d -> MLP에 의해 N*Kxoutchannels -> reshape
        #Similarity
        vec_sim = f_q.view(N, 1, self.out_channels) - knn_k + rel_pos_enc
        weights = self.mlp_attn(vec_sim.view(-1, self.out_channels)).view(N, self.k, self.out_channels)
        weights = self.softmax(weights) #softmax across k (dim=1), NxKxCout
        #weighted sum
        weighted_knn_v = weights * (knn_v + rel_pos_enc) #NxKxCout
        out_features = weighted_knn_v.sum(dim=1) #NxCout

        return out_features
##
class PointTransformerBlock(nn.Module):
    def __init__(self, channels, k):
        super().__init__()
        self.linear_in = nn.Linear(channels, channels)
        self.linear_out = nn.Linear(channels, channels)
        self.pt_layer = PointTransformerLayer(channels, channels, k)

    def forward(self, points, features):
        out_features = self.linear_in(features)
        out_features = self.pt_layer(points, out_features)
        out_features = self.linear_out(out_features)
        out_features += features

        return out_features

##
#input : input points Nx3, Features NxC, Sample M, K
#output : smapled Points Mx3, corresponding featrue MxC
class TransitionDown(nn.Module):
    def __init__(self, channels, num_samples, k):
        super().__init__()
        self.k = k
        self.num_samples = num_samples
        self.channels = channels

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, features):
        N = len(points)
        sampled_indicies = fps(points, self.num_samples) #numsamplex1
        sampled_points = points[sampled_indicies] #numsamplex3

        #kNN
        knn_dist, knn_indices = find_knn_general(sampled_points, points, self.k)

        #MLP
        knn_features = features[knn_indices.view(-1)]
        out_knn_features = self.mlp(knn_features).view(self.num_samples, self.k, -1) #MxKxC

        #Local MP
        out_features = out_knn_features.max(dim=1)[0]

        return sampled_points, out_features
##
# input : up_point Nx3, up_features NxC_up, down_points Mx3 down_features MxC_down
# output : out_features NxC_out
class TransitionUp(nn.Module):
    def __init__(self, up_channels, down_channels, out_channels):
        super().__init__()

        self.linear_up = nn.Linear(up_channels, out_channels)
        self.linear_down = nn.Linear(down_channels, out_channels)

    def forward(self, up_points, up_features, down_points, down_features):
        down_f = self.linear_down(down_features)
        interp_f = interpolate_knn(up_points, down_points, down_f, k=3)
        out_f = interp_f + self.linear_up(up_features)

        return out_f

class SimplePointTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, k):
        super().__init__()
        self.layer = PointTransformerLayer(in_channels, out_channels, k)
        self.down = TransitionDown(out_channels, ratio, k)
        self.up = TransitionUp(out_channels, out_channels, out_channels)

    def forward(self, points, features):
        skip_features = self.layer(points, features)
        down_points, out_features = self.down(points, skip_features)
        out_features = self.up(points, skip_features, down_points, out_features)

        return out_features

if __name__ == '__main__':
    N = 20
    M = 100
    R = 0.6
    K = 5
    ratio = 4
    C_in = 3
    C_out = 16

    points = torch.randn(N, 3)
    features = torch.randn(N, C_in)
    net = SimplePointTransformer(C_in, C_out, ratio, K)

    out_features = net(points, features)

    query_points = torch.randn(N, 3)
    dataset_points = torch.randn(M, 3)

    indicies = find_radius_general(query_points, dataset_points, R)
    print(len(indicies))
    print(indicies[0])

