import numpy as np
import torch
import torch.nn as nn

from geotransformer.modules.ops import pairwise_distance
from geotransformer.modules.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer
from geotransformer.modules.transformer.fusion_2 import AttentionFusion
from geotransformer.modules.transformer.projection import Projection
from geotransformer.modules.geotransformer.Img_Encoder import ImageEncoder
# class GeometricStructureEmbedding(nn.Module):
#     def __init__(self, hidden_dim, sigma_d, sigma_a, reduction_a='max'):
#         super(GeometricStructureEmbedding, self).__init__()
#         self.sigma_d = sigma_d
#         self.sigma_a = sigma_a
#         self.factor_a = 180.0 / (self.sigma_a * np.pi)

#         self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
#         self.proj_d = nn.Linear(hidden_dim, hidden_dim)
#         self.proj_a = nn.Linear(hidden_dim, hidden_dim)

#         self.reduction_a = reduction_a
#         if self.reduction_a not in ['max', 'mean', 'sum']:
#             raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        
        # Self attention for image features
        self.image_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        
        # Self attention for point cloud features
        self.pointcloud_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        
        # Cross attention between image and point cloud
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

        # Linear projection for final feature combination
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, image_feats, pointcloud_feats):
        # Image and pointcloud features self-attention
        image_feats, _ = self.image_attention(image_feats, image_feats, image_feats)
        pointcloud_feats, _ = self.pointcloud_attention(pointcloud_feats, pointcloud_feats, pointcloud_feats)
        
        # Cross attention between image and point cloud features
        cross_feats, _ = self.cross_attention(image_feats, pointcloud_feats, pointcloud_feats)
        
        # Concatenate the cross-modal features
        fused_feats = torch.cat([cross_feats, pointcloud_feats], dim=-1)
        
        # Project the final fused features
        fused_feats = self.fc(fused_feats)
        
        return fused_feats


class AdaptiveFusionModule(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(AdaptiveFusionModule, self).__init__()
        
        # Cross-modal attention module
        self.cross_modal_attention = CrossModalAttention(hidden_dim, num_heads, dropout)
        
        # Adaptive feature weighting module
        self.image_weighting = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.pointcloud_weighting = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Fusion refinement with an additional feed-forward network (optional)
        self.fusion_refinement = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, image_feats, pointcloud_feats):
        # Cross-modal attention fusion
        fused_feats = self.cross_modal_attention(image_feats, pointcloud_feats)
        
        # Calculate the weights for image and pointcloud features
        image_weight = torch.sigmoid(self.image_weighting(image_feats))
        pointcloud_weight = torch.sigmoid(self.pointcloud_weighting(pointcloud_feats))
        
        # Apply adaptive weighting to the fused features
        adaptive_fused_feats = image_weight * image_feats + pointcloud_weight * pointcloud_feats
        fused_feats += adaptive_fused_feats  # Add the adaptive weighted features to final fusion
        
        #在融合后的特征上增加了一个 fusion_refinement 网络
        refined_fused_feats = self.fusion_refinement(torch.cat([fused_feats, adaptive_fused_feats], dim=-1))
        return fused_feats


class SelfGeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d):
        """
        Initialize SelfGeometricStructureEmbedding.
        
        Args:
            hidden_dim (int): The dimension of the hidden feature.
            sigma_d (float): The scale for distance normalization.
        """
        super().__init__()
        
        # Initialize parameters
        self.sigma_d = sigma_d
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)

    def _compute_distance_indices(self, points):
        """
        Compute pairwise distance indices for the input point cloud.

        Args:
            points (torch.Tensor): The input point cloud of shape (B, N, 3).
        
        Returns:
            torch.Tensor: The normalized pairwise distance indices of shape (B, N, N).
        """
        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        return dist_map / self.sigma_d  # Normalize by sigma_d

    def forward(self, points):
        """
        Forward pass to generate the geometric embeddings for the input points.
        
        Args:
            points (torch.Tensor): The input point cloud of shape (B, N, 3).
        
        Returns:
            torch.Tensor: The generated geometric embeddings of shape (B, N, hidden_dim).
        """
        # Compute distance indices
        d_indices = self._compute_distance_indices(points)
        
        # Embed and project the distance indices
        d_embeddings = self.embedding(d_indices)
        return self.proj_d(d_embeddings)


    # @torch.no_grad()
    # def get_embedding_indices(self, points):
    #     r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

    #     Args:
    #         points: torch.Tensor (B, N, 3), input point cloud

    #     Returns:
    #         d_indices: torch.FloatTensor (B, N, N), distance embedding indices
    #         a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
    #     """

    #     dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
    #     d_indices = dist_map / self.sigma_d

    #     return d_indices

    # def forward(self, points):
    #     d_indices = self.get_embedding_indices(points)

    #     d_embeddings = self.embedding(d_indices)
    #     d_embeddings = self.proj_d(d_embeddings)

    #     return d_embeddings

# class CrossGeometricStructureEmbedding(nn.Module):
#     def __init__(self, hidden_dim, sigma_d, sigma_a, reduction_a='max'):
#         super(CrossGeometricStructureEmbedding, self).__init__()
#         self.sigma_d = sigma_d
#         self.sigma_a = sigma_a
#         self.factor_a = 180.0 / (self.sigma_a * np.pi)

#         self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
#         self.proj_a = nn.Linear(hidden_dim, hidden_dim)
#         self.proj_d = nn.Linear(hidden_dim, hidden_dim)

#         self.reduction_a = reduction_a
#         if self.reduction_a not in ['max', 'mean', 'sum']:
#             raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

#     @torch.no_grad()
#     def get_embedding_indices(self, points, anchor_points):
#         r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

#         Args:
#             points: torch.Tensor (B, N, 3), input point cloud
#             anchor_points:  torch.Tensor (B, K, 3), input point cloud

#         Returns:
#             d_indices: torch.FloatTensor (B, N, K), distance embedding indices
#             a_indices: torch.FloatTensor (B, N, K), angular embedding indices
#         """
#         dist_map = torch.sqrt(pairwise_distance(points, anchor_points))  # (B, N, k)
#         d_indices = dist_map / self.sigma_d

#         ref_vectors = points.unsqueeze(2) - anchor_points.unsqueeze(1)
#         # sta_indices = torch.tensor([1, 2, 0]).to('cuda')
#         sta_indices = torch.arange(1, anchor_points.shape[1] + 1).to('cuda')
#         sta_indices[anchor_points.shape[1] - 1] = 0
#         sta_indices = sta_indices[None, None, :,None].repeat(1, ref_vectors.shape[1],1,3)
#         anc_vectors = torch.gather(ref_vectors, dim=2, index=sta_indices)

#         sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)
#         cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)
#         angles = torch.atan2(sin_values, cos_values)
#         a_indices = angles * self.factor_a

#         return d_indices, a_indices

#     def forward(self, points, anchor_points, cor_score):
#         d_indices, a_indices = self.get_embedding_indices(points, anchor_points)

#         a_embeddings = self.embedding(a_indices)
#         a_embeddings = self.proj_a(a_embeddings)

#         d_embeddings = self.embedding(d_indices)#(B, N, k, d)
#         d_embeddings = self.proj_d(d_embeddings)#(B, N, k, d)
#         if self.reduction_a == 'max':
#             d_embeddings = d_embeddings.max(dim=2)[0] #(B, N, d)
#             a_embeddings = a_embeddings.max(dim=2)[0]
#         elif self.reduction_a == 'mean':
#             d_embeddings = d_embeddings.mean(dim=2) #(B, N, d)
#             a_embeddings = a_embeddings.mean(dim=2)
#         else:
#             d_embeddings = (cor_score[None, None, :,None] * d_embeddings).sum(2) #(B, N, d)
#             a_embeddings = (cor_score[None, None, :,None] * a_embeddings).sum(2)

#         embeddings = d_embeddings+ a_embeddings
#         return embeddings

class CrossGeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, reduction_a='max'):
        super().__init__()
        
        # Initialize parameters
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)

        # Embedding and projection layers
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)

        # Reduction mode validation
        self.reduction_a = reduction_a
        if reduction_a not in ['max', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points, anchor_points):
        """
        Compute pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points (torch.Tensor): (B, N, 3), input point cloud
            anchor_points (torch.Tensor): (B, K, 3), input anchor point cloud

        Returns:
            tuple: 
                - d_indices (torch.FloatTensor): (B, N, K), distance embedding indices
                - a_indices (torch.FloatTensor): (B, N, K), angular embedding indices
        """
        # Compute pairwise distance between points and anchor points
        dist_map = torch.sqrt(pairwise_distance(points, anchor_points))  # (B, N, K)
        d_indices = dist_map / self.sigma_d  # Normalize distances

        # Calculate angular embedding indices
        ref_vectors = points.unsqueeze(2) - anchor_points.unsqueeze(1)  # (B, N, K, 3)
        
        # Compute the angular components using cross and dot products
        sta_indices = torch.arange(1, anchor_points.shape[1] + 1).to('cuda')
        sta_indices[anchor_points.shape[1] - 1] = 0
        sta_indices = sta_indices[None, None, :, None].repeat(1, ref_vectors.shape[1], 1, 3)
        anc_vectors = torch.gather(ref_vectors, dim=2, index=sta_indices)

        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # Sin
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # Cos
        angles = torch.atan2(sin_values, cos_values)  # Angular values
        a_indices = angles * self.factor_a  # Scaling angles

        return d_indices, a_indices

    # def forward(self, points, anchor_points, cor_score):
    #     """
    #     Forward pass to generate the geometric structure embedding based on point distances and angles.

    #     Args:
    #         points (torch.Tensor): (B, N, 3), input point cloud
    #         anchor_points (torch.Tensor): (B, K, 3), anchor point cloud
    #         cor_score (torch.Tensor): Correspondence scores for weighted aggregation

    #     Returns:
    #         torch.Tensor: The fused geometric embeddings of shape (B, N, hidden_dim)
    #     """
    #     # Get distance and angular indices
    #     d_indices, a_indices = self.get_embedding_indices(points, anchor_points)

    #     # Embed and project distance and angular embeddings
    #     a_embeddings = self.embedding(a_indices)
    #     d_embeddings = self.embedding(d_indices)
        
    #     a_embeddings = self.proj_a(a_embeddings)
    #     d_embeddings = self.proj_d(d_embeddings)

    #     # Reduce embeddings based on the selected mode
    #     if self.reduction_a == 'max':
    #         a_embeddings = a_embeddings.max(dim=2)[0]
    #         d_embeddings = d_embeddings.max(dim=2)[0]
    #     elif self.reduction_a == 'mean':
    #         a_embeddings = a_embeddings.mean(dim=2)
    #         d_embeddings = d_embeddings.mean(dim=2)
    #     else:  # 'sum'
    #         a_embeddings = (cor_score[None, None, :, None] * a_embeddings).sum(2)
    #         d_embeddings = (cor_score[None, None, :, None] * d_embeddings).sum(2)

    #     # Combine the distance and angular embeddings
    #     embeddings = d_embeddings + a_embeddings

    #     return embeddings

    def forward(self, points, anchor_points, cor_score):
        """
        Forward pass to generate the geometric structure embedding based on point distances and angles.

        Args:
            points (torch.Tensor): (B, N, 3), input point cloud
            anchor_points (torch.Tensor): (B, K, 3), anchor point cloud
            cor_score (torch.Tensor): Correspondence scores for weighted aggregation

        Returns:
            torch.Tensor: The fused geometric embeddings of shape (B, N, hidden_dim)
        """
        # Get distance and angular indices
        d_indices, a_indices = self.get_embedding_indices(points, anchor_points)

        # Embed and project distance and angular embeddings
        a_embeddings = self.proj_a(self.embedding(a_indices))
        d_embeddings = self.proj_d(self.embedding(d_indices))

        # Create a dictionary for reduction operations
        reduction_operations = {
            'max': lambda x: x.max(dim=2)[0],
            'mean': lambda x: x.mean(dim=2),
            'sum': lambda x: (cor_score[None, None, :, None] * x).sum(2)
        }

        # Apply the selected reduction operation to the embeddings
        a_embeddings = reduction_operations.get(self.reduction_a, lambda x: x)(a_embeddings)
        d_embeddings = reduction_operations.get(self.reduction_a, lambda x: x)(d_embeddings)

        # Combine the distance and angular embeddings
        embeddings = d_embeddings + a_embeddings

        return embeddings


# class GeometricTransformer(nn.Module):
#     def __init__(
#         self,
#         hidden_dim,
#         num_heads,
#         blocks,
#         sigma_d,
#         sigma_a,
#         dropout=None,
#         activation_fn='ReLU',
#         reduction_a='max',
#     ):
#         r"""Geometric Transformer (GeoTransformer).

#         Args:
#             input_dim: input feature dimension
#             output_dim: output feature dimension
#             hidden_dim: hidden feature dimension
#             num_heads: number of head in transformer
#             blocks: list of 'self' or 'cross'
#             sigma_d: temperature of distance
#             sigma_a: temperature of angles
#             angle_k: number of nearest neighbors for angular embedding
#             activation_fn: activation function
#             reduction_a: reduction mode of angular embedding ['max', 'mean']
#         """
#         super(GeometricTransformer, self).__init__()

#         self.embedding_self = SelfGeometricStructureEmbedding(hidden_dim, sigma_d)
#         self.embedding_cross = CrossGeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, reduction_a=reduction_a)

#         # self.transformer = RPEConditionalTransformer(
#         #     blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn, return_attention_scores=True
#         # )
#         self.transformer = RPEConditionalTransformer(
#             blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn
#         )
#         self.out_proj = nn.Linear(hidden_dim, hidden_dim)

#     def forward(
#         self,
#         ref_points,
#         src_points,
#         ref_feats,
#         src_feats,
#         ref_anchor_indices, 
#         src_anchor_indices,
#         cor_score,
#         ref_masks=None,
#         src_masks=None,
#     ):
#         r"""Geometric Transformer

#         Args:
#             ref_points (Tensor): (B, N, 3)
#             src_points (Tensor): (B, M, 3)
#             ref_feats (Tensor): (B, N, C)
#             src_feats (Tensor): (B, M, C)
#             ref_masks (Optional[BoolTensor]): (B, N)
#             src_masks (Optional[BoolTensor]): (B, M)

#         Returns:
#             ref_feats: torch.Tensor (B, N, C)
#             src_feats: torch.Tensor (B, M, C)
#         """
#         ref_anchor_points = ref_points[:, ref_anchor_indices, :]
#         src_anchor_points = src_points[:, src_anchor_indices, :]

#         ref_embeddings_self = self.embedding_self(ref_points)
#         src_embeddings_self = self.embedding_self(src_points)

#         ref_embeddings_cross = self.embedding_cross(ref_points, ref_anchor_points, cor_score)
#         src_embeddings_cross = self.embedding_cross(src_points, src_anchor_points, cor_score)

#         ref_anchor_feats = ref_feats[:, ref_anchor_indices, :]
#         src_anchor_feats = src_feats[:, src_anchor_indices, :]

#         # ref_feats, src_feats, attention_scores = self.transformer(
#         #     ref_feats,
#         #     src_feats,
#         #     ref_embeddings_self,
#         #     src_embeddings_self,
#         #     ref_embeddings_cross,
#         #     src_embeddings_cross,
#         #     masks0=ref_masks,
#         #     masks1=src_masks,
#         # )
#         ref_feats, src_feats = self.transformer(
#             ref_feats,
#             src_feats,
#             ref_embeddings_self,
#             src_embeddings_self,
#             ref_embeddings_cross,
#             src_embeddings_cross,
#             masks0=ref_masks,
#             masks1=src_masks,
#         )

#         ref_feats = self.out_proj(ref_feats)
#         src_feats = self.out_proj(src_feats)

#         # return ref_feats, src_feats, attention_scores
#         return ref_feats, src_feats

class GeometricTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        blocks,
        sigma_d,
        sigma_a,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        """
        Geometric Transformer (GeoTransformer).

        Args:
            hidden_dim (int): Hidden feature dimension.
            num_heads (int): Number of attention heads.
            blocks (list): List of attention blocks ('self' or 'cross').
            sigma_d (float): Distance temperature for the geometric embedding.
            sigma_a (float): Angle temperature for the geometric embedding.
            dropout (float, optional): Dropout rate.
            activation_fn (str, optional): Activation function to use ('ReLU' by default).
            reduction_a (str, optional): Reduction mode for angular embedding ('max', 'mean', or 'sum').
        """
        super().__init__()

        # Initialize geometric structure embedding modules
        self.embedding_self = SelfGeometricStructureEmbedding(hidden_dim, sigma_d)
        self.embedding_cross = CrossGeometricStructureEmbedding(
            hidden_dim, sigma_d, sigma_a, reduction_a=reduction_a
        )

        # Initialize transformer module
        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_anchor_indices,
        src_anchor_indices,
        cor_score,
        ref_masks=None,
        src_masks=None,
    ):
        """
        Forward pass for the Geometric Transformer.

        Args:
            ref_points (Tensor): (B, N, 3), reference point cloud.
            src_points (Tensor): (B, M, 3), source point cloud.
            ref_feats (Tensor): (B, N, C), reference feature tensor.
            src_feats (Tensor): (B, M, C), source feature tensor.
            ref_masks (Tensor, optional): (B, N), mask for reference points.
            src_masks (Tensor, optional): (B, M), mask for source points.

        Returns:
            ref_feats (Tensor): (B, N, C), updated reference feature tensor.
            src_feats (Tensor): (B, M, C), updated source feature tensor.
        """
        # Extract anchor points based on provided indices
        ref_anchor_points = ref_points[:, ref_anchor_indices, :]
        src_anchor_points = src_points[:, src_anchor_indices, :]
        # Compute self-geometric embeddings
        ref_embeddings_self = self.embedding_self(ref_points)
        src_embeddings_self = self.embedding_self(src_points)

        # Compute cross-geometric embeddings with correspondences
        ref_embeddings_cross = self.embedding_cross(ref_points, ref_anchor_points, cor_score)
        src_embeddings_cross = self.embedding_cross(src_points, src_anchor_points, cor_score)

        # Extract anchor features
        ref_anchor_feats = ref_feats[:, ref_anchor_indices, :]
        src_anchor_feats = src_feats[:, src_anchor_indices, :]
        # Pass through the transformer
        ref_feats, src_feats = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings_self,
            src_embeddings_self,
            ref_embeddings_cross,
            src_embeddings_cross,
            masks0=ref_masks,
            masks1=src_masks,
        )

        # Output projection
        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats


class AggregateTransformer(nn.Module):
    def __init__(
        self,
        image_num,
        input_dim_c,
        input_dim_i,
        hidden_dim,
        num_heads,
        sigma_d_self,
        dropout=None,
        blocks=['self'],
        activation_fn='ReLU'
    ):
        """
        Args:
            image_num: Number of images to use (1 or 2).
            input_dim_c: Point cloud input feature dimension.
            input_dim_i: Image input feature dimension.
            hidden_dim: Hidden feature dimension.
            num_heads: Number of attention heads.
            sigma_d_self: Temperature of distance for geometric structure.
            dropout: Dropout rate.
            blocks: List of transformer blocks ('self' or 'cross').
            activation_fn: Activation function (e.g., 'ReLU').
        """
        super(AggregateTransformer, self).__init__()

        # Initialize modules
        self.image_num = image_num
        self.embedding_self = SelfGeometricStructureEmbedding(hidden_dim, sigma_d_self)
        self.in_proj = nn.Linear(input_dim_c, hidden_dim)
        self.transformer = RPEConditionalTransformer(blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn)
        self.Img_Encoder = ImageEncoder()
        self.fusion_transformer = AttentionFusion(
            image_num=image_num,
            image_dim=input_dim_i,
            latent_dim=input_dim_i,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=input_dim_i,
            latent_dim_head=input_dim_i,
        )
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.adaptive_fusion = AdaptiveFusionModule(hidden_dim, num_heads)


    def project(self, points_f, image, intrinsics, world2camera, node_knn_indices, node_knn_masks):
        """Projects the 3D points to 2D image space and computes corresponding masks and indices."""
        ifeats = self.Img_Encoder(image.unsqueeze(0).cuda()).squeeze(0).permute(1, 2, 0)  # (H, W, C)
        projection = Projection(intrinsics)
        xy_in_image_space, mask = projection.new_projection(points_f, ifeats, world2camera)
        mask = mask[node_knn_indices] & node_knn_masks
        inds3d = torch.nonzero(mask)
        inds2d = xy_in_image_space[node_knn_indices[inds3d[:, 0], inds3d[:, 1]]]
        return ifeats, inds2d, inds3d

    def project_points(points_f, image, intrinsics, rotation, knn_indices, knn_masks):
        return self.project(points_f, image, intrinsics, rotation.float().cuda(), knn_indices, knn_masks)

    def get_projected_features(points_f, data_dict, ref_or_src='ref'):
        image = data_dict[f'{ref_or_src}_image']
        intrinsics = data_dict[f'{ref_or_src}_intrinsics']
        rotation = data_dict[f'{ref_or_src}_rotation']
        knn_indices = data_dict[f'{ref_or_src}_node_knn_indices']
        knn_masks = data_dict[f'{ref_or_src}_node_knn_masks']
        
        return self.project_points(points_f, image, intrinsics, rotation, knn_indices, knn_masks)

    def fuse_features(ifeats, feats, inds2d, inds3d):
        fused_feats = self.fusion_transformer(ifeats, feats, inds2d, inds3d)
        return fused_feats.unsqueeze(0)

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_points_f, 
        src_points_f,
        ref_node_knn_indices, 
        src_node_knn_indices,
        ref_node_knn_masks, 
        src_node_knn_masks,
        data_dict
    ):

        ref_embeddings_self = self.embedding_self(ref_points)
        src_embeddings_self = self.embedding_self(src_points)  # 256

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats) # 1024 -> 256


        # get index between 2d and 3d
        #if self.image_num == 1:
        ref_ifeats, ref_inds2d, ref_inds3d = self.project(ref_points_f, 
                                                    data_dict['ref_image'],
                                                    data_dict['ref_intrinsics'],
                                                    data_dict['ref_rotation'].float().cuda(),
                                                    ref_node_knn_indices,
                                                    ref_node_knn_masks)
        src_ifeats, src_inds2d, src_inds3d = self.project(src_points_f, 
                                                    data_dict['src_image'],
                                                    data_dict['src_intrinsics'],
                                                    data_dict['src_rotation'].float().cuda(),
                                                    src_node_knn_indices,
                                                    src_node_knn_masks)
        # fusion image and point
        ref_pi_feats = self.fusion_transformer(ref_ifeats, ref_feats, ref_inds2d, ref_inds3d)
        ref_pi_feats = ref_pi_feats.unsqueeze(0)
        src_pi_feats = self.fusion_transformer(src_ifeats, src_feats, src_inds2d, src_inds3d)
        src_pi_feats = src_pi_feats.unsqueeze(0)

        # else:
        #     ref_ifeats, ref_inds2d, ref_inds3d = self.get_projected_features(ref_points_f, data_dict, 'ref')
        #     src_ifeats, src_inds2d, src_inds3d = self.get_projected_features(src_points_f, data_dict, 'src')

        #     ref_pi_feats = self.fuse_features(ref_ifeats, ref_feats, ref_inds2d, ref_inds3d)
        #     src_pi_feats = self.fuse_features(src_ifeats, src_feats, src_inds2d, src_inds3d)


        # elif self.image_num == 2:
        #     ref_ifeats_1, ref_inds2d_1, ref_inds3d_1 = self.project(ref_points_f, 
        #                                                             data_dict['ref_image_1'],
        #                                                             data_dict['ref_intrinsics'],
        #                                                             data_dict['ref_world2camera_1'],
        #                                                             ref_node_knn_indices,
        #                                                             ref_node_knn_masks)

        #     ref_ifeats_2, ref_inds2d_2, ref_inds3d_2 = self.project(ref_points_f, 
        #                                                             data_dict['ref_image_2'],
        #                                                             data_dict['ref_intrinsics'],
        #                                                             data_dict['ref_world2camera_2'],
        #                                                             ref_node_knn_indices,
        #                                                             ref_node_knn_masks)
            
        #     src_ifeats_1, src_inds2d_1, src_inds3d_1 = self.project(src_points_f, 
        #                                                             data_dict['src_image_1'],
        #                                                             data_dict['src_intrinsics'],
        #                                                             data_dict['src_world2camera_1'],
        #                                                             src_node_knn_indices,
        #                                                             src_node_knn_masks)

        #     src_ifeats_2, src_inds2d_2, src_inds3d_2 = self.project(src_points_f, 
        #                                                             data_dict['src_image_2'],
        #                                                             data_dict['src_intrinsics'],
        #                                                             data_dict['src_world2camera_2'],
        #                                                             src_node_knn_indices,
        #                                                             src_node_knn_masks)

        #     ref_ifeats = {'image_feats_1':ref_ifeats_1, 'image_feats_2':ref_ifeats_2}
        #     ref_inds2d = {'inds2d_1':ref_inds2d_1, 'inds2d_2':ref_inds2d_2}
        #     ref_inds3d = {'inds3d_1':ref_inds3d_1, 'inds3d_2':ref_inds3d_2}

        #     src_ifeats = {'image_feats_1':src_ifeats_1, 'image_feats_2':src_ifeats_2}
        #     src_inds2d = {'inds2d_1':src_inds2d_1, 'inds2d_2':src_inds2d_2}
        #     src_inds3d = {'inds3d_1':src_inds3d_1, 'inds3d_2':src_inds3d_2}

        #     ref_pi_feats = self.fusion_transformer(ref_ifeats, ref_feats, ref_inds2d, ref_inds3d)
        #     ref_pi_feats = ref_pi_feats.unsqueeze(0)
        #     src_pi_feats = self.fusion_transformer(src_ifeats, src_feats, src_inds2d, src_inds3d)
        #     src_pi_feats = src_pi_feats.unsqueeze(0)

        # fusion point-wise distance
        ref_pwd_feats, src_pwd_feats = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings_self,
            src_embeddings_self,
            embeddings0_cross=None,
            embeddings1_cross=None
        )

        ref_feats = torch.cat((ref_pi_feats, ref_pwd_feats), dim=2)
        src_feats = torch.cat((src_pi_feats, src_pwd_feats), dim=2)

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)


        return ref_feats, src_feats
