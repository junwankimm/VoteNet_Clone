from util import *
from functions import *
import torch
import torch.nn as nn

#input : points Nx3, features NxC_in, output feature dimension C_out, num_votes, ratio, K
#output : output votes (num_votes, 3) output_features (num_votes, C_out)

class VotingModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_votes, ratio, k):
        super().__init__()
        self.num_votes = num_votes

        self.pfe = SimplePointTransformer(in_channels, out_channels, ratio, k)
        self.voter = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, 3 + out_channels) #3 for delta x, 나머지는 delta f
        )

    def forward(self, points, features):
        #point feature extract
        out_features = self.pfe(points, features)

        #sample seed points
        indicies = fps(points, self.num_votes)
        seed_points = points[indicies]
        seed_features = out_features[indicies]

        #voting
        residuals = self.voter(seed_features)
        vote_points = seed_points + residuals[:, :3]
        vote_features = seed_features + residuals[:, 3:]

        return vote_points, vote_features

#input : vote_points (N,3), vote_features(N, C_in), num_clusters, radius, nms_iou_threshold
#output : detected bboxes (M, 1+6) -> objectness 1 + coordinate 6
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_clusters, radius, nms_iou_threshold):
        super().__init__()
        self.num_clusters = num_clusters
        self.radius = radius
        self.nms_iou_threshold = nms_iou_threshold

        self.mlp1 = nn.Sequential(
            nn.Linear(3 + in_channels, in_channels), #voting된 center의 중심과 이웃점과의 relative distance가 concat
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Linear(in_channels, 7)

    def forward(self, vote_points, vote_features):
        #sample cluster centroids
        sampled_indicies = fps(vote_points, self.num_clusters)
        cluster_points = vote_points[sampled_indicies]

        #find cluster neighbors
        indicies = find_radius_general(cluster_points, vote_points, self.radius)
        grouped_features = []

        #grouping
        for group_center, group_indicies in zip(cluster_points, indicies):
            #calculate relative pos
            features_in_group = vote_features[group_indicies]
            relative_pos = group_center.unsqueeze(0) - vote_points[group_indicies] / self.radius
            features_with_pos = torch.cat([relative_pos, features_in_group], dim=-1)

            group_features = self.mlp1(features_with_pos).max(dim=0)[0]
            group_features = self.mlp2(group_features)
            grouped_features.append(group_features)

        grouped_features = torch.stack(grouped_features) #num_clusterxin_channels

        #predict bbox
        boxes = self.final(grouped_features)
        box_scores = boxes[:, 0].sigmoid()
        box_coordinates = boxes[:, 1:].sigmoid() #coord norm

        #nms
        final_boxes = nms(box_coordinates, box_scores, self.nms_iou_threshold)

        return final_boxes



if __name__ == '__main__':
    N = 50
    C_in = 8
    num_clusters = 16
    radius = 0.9
    iou_threshold = 0.4
    C_out = 16
    num_votes = 32
    ratio = 4
    K = 5

    vote_points = torch.randn(N, 3)
    vote_features = torch.randn(N, C_in)
    detection_h = DetectionHead(C_in, num_clusters, radius, iou_threshold)
    pred_boxes = detection_h(vote_points, vote_features)
    print(pred_boxes.shape)