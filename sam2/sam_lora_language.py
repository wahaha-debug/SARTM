import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file
from icecream import ic
from sam2.modeling.sam2_base import SAM2Base
import torch.nn.init as init
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sam2.modules.ffm import FeatureFusionModule as FFM
from sam2.modules.ffm import FeatureRectifyModule as FRM
import clip

class PrototypeSegmentation:
    def __init__(self, num_classes, feature_dim):
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.global_prototypes = torch.zeros((num_classes, feature_dim), requires_grad=True).to(
            'cuda')

    def update_global_prototypes(self, current_prototypes):

        self.global_prototypes.data = 0.9 * self.global_prototypes.data + 0.1 * current_prototypes.data

    def compute_loss(self, features, labels):

        batch_prototypes = self.calculate_batch_prototypes(features, labels)

        self.update_global_prototypes(batch_prototypes)

        prototype_loss = self.prototype_loss(batch_prototypes)

        total_loss = prototype_loss
        return batch_prototypes, total_loss

    def calculate_batch_prototypes(self, features, labels):

        batch_prototypes = torch.zeros((self.num_classes, self.feature_dim), device=features.device)

        count = torch.zeros(self.num_classes, device=features.device)

        labels = labels.to(features.device)

        labels = labels.unsqueeze(1)

        labels_resized = F.interpolate(labels.float(), size=features.shape[2:], mode='nearest').long().squeeze(
            1)

        b, c, h, w = features.size()
        features = features.permute(0, 2, 3, 1).reshape(-1, c)

        labels_resized = labels_resized.view(-1)

        for i in range(self.num_classes):
            mask = (labels_resized == i)
            if mask.sum() > 0:
                class_features = features[mask]

                gap_features = class_features.view(1, c, -1, 1)
                gap = torch.nn.AdaptiveAvgPool2d((1, 1))
                batch_prototypes[i] = gap(gap_features).squeeze()
                count[i] = mask.sum()

        count = count.clamp(min=1)
        return batch_prototypes / count.unsqueeze(1)

    def prototype_loss(self, batch_prototypes):

        loss = F.kl_div(F.log_softmax(batch_prototypes, dim=-1), F.softmax(self.global_prototypes, dim=-1),
                        reduction='none')

        return loss


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            activation: nn.Module = nn.ReLU,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class _LoRA_qkv(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
            linear_a_q2: nn.Module = None,
            linear_b_q2: nn.Module = None,
            linear_a_v2: nn.Module = None,
            linear_b_v2: nn.Module = None,

            linear_a_q3: nn.Module = None,
            linear_b_q3: nn.Module = None,
            linear_a_v3: nn.Module = None,
            linear_b_v3: nn.Module = None,

    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v

        self.linear_a_q2 = linear_a_q2
        self.linear_b_q2 = linear_b_q2
        self.linear_a_v2 = linear_a_v2
        self.linear_b_v2 = linear_b_v2

        self.linear_a_q3 = linear_a_q3
        self.linear_b_q3 = linear_b_q3
        self.linear_a_v3 = linear_a_v3
        self.linear_b_v3 = linear_b_v3

        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v

        if self.linear_a_q2 and self.linear_b_q2:
            new_q2 = self.linear_b_q2(self.linear_a_q2(x))
            qkv[:, :, :, :self.dim] += new_q2

        if self.linear_a_v2 and self.linear_b_v2:
            new_v2 = self.linear_b_v2(self.linear_a_v2(x))
            qkv[:, :, :, -self.dim:] += new_v2

        if self.linear_a_q3 and self.linear_b_q3:
            new_q3 = self.linear_b_q3(self.linear_a_q3(x))
            qkv[:, :, :, :self.dim] += new_q3

        if self.linear_a_v3 and self.linear_b_v3:
            new_v3 = self.linear_b_v3(self.linear_a_v3(x))
            qkv[:, :, :, -self.dim:] += new_v3

        return qkv


class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.
    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, model_clip, sam_model: SAM2Base, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(
                len(sam_model.image_encoder.trunk.blocks)))

        self.w_As = []
        self.w_Bs = []

        self.w_As2 = []
        self.w_Bs2 = []

        self.w_As3 = []
        self.w_Bs3 = []

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features

            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            w_a_linear_q2 = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q2 = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v2 = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v2 = nn.Linear(r, self.dim, bias=False)
            self.w_As2.append(w_a_linear_q2)
            self.w_Bs2.append(w_b_linear_q2)
            self.w_As2.append(w_a_linear_v2)
            self.w_Bs2.append(w_b_linear_v2)

            w_a_linear_q3 = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q3 = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v3 = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v3 = nn.Linear(r, self.dim, bias=False)
            self.w_As3.append(w_a_linear_q3)
            self.w_Bs3.append(w_b_linear_q3)
            self.w_As3.append(w_a_linear_v3)
            self.w_Bs3.append(w_b_linear_v3)

            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                w_a_linear_q2,
                w_b_linear_q2,
                w_a_linear_v2,
                w_b_linear_v2,
                w_a_linear_q3,
                w_b_linear_q3,
                w_a_linear_v3,
                w_b_linear_v3
            )
        self.reset_parameters()
        self.sam = sam_model

        transformer_dim = self.sam.sam_mask_decoder.transformer_dim
        self.mlp_src = MLP(input_dim=transformer_dim, hidden_dim=transformer_dim // 8, output_dim=transformer_dim // 8,
                           num_layers=3)
        self.mlp_feat_s0 = MLP(input_dim=transformer_dim // 8, hidden_dim=transformer_dim // 8,
                               output_dim=transformer_dim // 8, num_layers=3)
        self.mlp_feat_s1 = MLP(input_dim=transformer_dim // 4, hidden_dim=transformer_dim // 8,
                               output_dim=transformer_dim // 8, num_layers=3)
        self.linear_fuse = nn.Conv2d(transformer_dim // 8 * 3, transformer_dim // 8, kernel_size=1)
        self.linear_pred = nn.Conv2d(transformer_dim // 8, 9, kernel_size=1)

        num_heads = [1, 2, 4]
        self.FRMs = nn.ModuleList([
            FRM(dim=32, reduction=1),
            FRM(dim=64, reduction=1),
            FRM(dim=256, reduction=1)])
        self.FFMs = nn.ModuleList([
            FFM(dim=32, reduction=1, num_heads=num_heads[0], norm_layer=nn.BatchNorm2d),
            FFM(dim=64, reduction=1, num_heads=num_heads[1], norm_layer=nn.BatchNorm2d),
            FFM(dim=256, reduction=1, num_heads=num_heads[2], norm_layer=nn.BatchNorm2d)])

        self.model_clip = model_clip
        self.class_names = ['Background', 'Hand-Drill', 'BackPack', 'Fire-Extinguisher', 'Survivor']

        self.prototype_segmentation = PrototypeSegmentation(5, 256)

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)

        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam,
                                                                     torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()

        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}

        save_file(merged_dict, filename)

    def reset_parameters(self):

        for w_a in self.w_As:
            init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))

        for w_b in self.w_Bs:
            init.zeros_(w_b.weight)

        for w_a2 in self.w_As2:
            init.kaiming_uniform_(w_a2.weight, a=math.sqrt(5))

        for w_b2 in self.w_Bs2:
            init.zeros_(w_b2.weight)

        for w_a3 in self.w_As3:
            init.kaiming_uniform_(w_a3.weight, a=math.sqrt(5))

        for w_b3 in self.w_Bs3:
            init.zeros_(w_b3.weight)

    def seg_fuse(self, src, feat_s0, feat_s1):
        '''
        功能：
        对输入的源特征（src）和其他两组特征（feat_s0 和 feat_s1）进行融合，生成分割输出。
        主要步骤：
        展平与变换：将输入特征展平为二维格式，通过MLP变换后恢复原状。
        上采样对齐：将所有特征上采样到统一的目标分辨率（1024x1024）。
        特征融合：通过通道拼接和1x1卷积对所有特征进行融合处理。
        输出生成：通过卷积生成嵌入结果，用于分割任务。
        输出：
        返回一个形状为 [batch_size, num_classes, 1024, 1024] 的张量，用于分割类别预测。
        '''

        b, c, _, _ = src.shape
        src_flat = src.view(b, c, -1).transpose(1, 2)
        feat_s0_flat = feat_s0.view(b, feat_s0.size(1), -1).transpose(1, 2)
        feat_s1_flat = feat_s1.view(b, feat_s1.size(1), -1).transpose(1, 2)

        src_transformed = self.mlp_src(src_flat)
        src_transformed = src_transformed.transpose(1, 2).view(b, -1, src.shape[2], src.shape[3])

        src_transformed = F.interpolate(src_transformed, size=[1280, 1280], mode='bilinear',
                                        align_corners=False)

        feat_s0_transformed = self.mlp_feat_s0(feat_s0_flat)
        feat_s0_transformed = feat_s0_transformed.transpose(1, 2).view(b, -1, feat_s0.size(2), feat_s0.size(
            3))
        feat_s0_transformed = F.interpolate(feat_s0_transformed, size=[1280, 1280], mode='bilinear',
                                            align_corners=False)

        feat_s1_transformed = self.mlp_feat_s1(feat_s1_flat)
        feat_s1_transformed = feat_s1_transformed.transpose(1, 2).view(b, -1, feat_s1.size(2), feat_s1.size(
            3))
        feat_s1_transformed = F.interpolate(feat_s1_transformed, size=[1280, 1280], mode='bilinear',
                                            align_corners=False)

        combined_features = torch.cat([src_transformed, feat_s0_transformed, feat_s1_transformed],
                                      dim=1)
        combined_features = self.dropout(combined_features)
        upscaled_embedding = self.linear_fuse(combined_features)
        upscaled_embedding = self.linear_pred(upscaled_embedding)

        return upscaled_embedding

    def fpn_fuse(self, feat, high_res_0, high_res_1):
        device = feat.device

        high_res_0_conv = nn.Conv2d(32, 256, kernel_size=1).to(device)(high_res_0)
        high_res_1_conv = nn.Conv2d(64, 256, kernel_size=1).to(device)(high_res_1)

        feat_up = F.interpolate(feat, size=high_res_1_conv.shape[2:], mode='bilinear', align_corners=False)
        fpn_merge_1 = feat_up + high_res_1_conv
        fpn_merge_1_up = F.interpolate(fpn_merge_1, size=high_res_0_conv.shape[2:], mode='bilinear',
                                       align_corners=False)
        final_merge = high_res_0_conv + fpn_merge_1_up
        final_output = nn.Conv2d(256, 5, kernel_size=3, padding=1).to(device)(final_merge)
        final_output = F.interpolate(final_output, size=(1280, 1280), mode='bilinear', align_corners=False)

        return final_output

    def process_tensor(self, feature, m, b):
        _, fc, fh, fw = feature.size()
        feature = feature.reshape(m, b, fc, fh, fw)
        cam, f = feature[0].squeeze(-1), feature[0].squeeze(-1)
        return cam, f

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature

    def forward(self, batched_input, label, multimask_output):
        batched_input = torch.stack([x for x in batched_input], dim=0)
        m, b, _, h, w = batched_input.shape
        batched_input = batched_input.reshape(2 * b, 3, h, w)
        image_embedding = self.sam.forward_image(batched_input)

        high_res_0 = image_embedding["backbone_fpn"][0]
        high_res_1 = image_embedding["backbone_fpn"][1]
        feat = image_embedding['vision_features']

        x0_cam, x0_f = self.process_tensor(high_res_0, m, b)
        x0_cam, x0_f = self.FRMs[0](x0_cam, x0_f)
        x0_fused = self.FFMs[0](x0_cam, x0_f)

        x1_cam, x1_f = self.process_tensor(high_res_1, m, b)
        x1_cam, x1_f = self.FRMs[1](x1_cam, x1_f)
        x1_fused = self.FFMs[1](x1_cam, x1_f)

        x_cam, x_f = self.process_tensor(feat, m, b)
        x_cam, x_f = self.FRMs[2](x_cam, x_f)
        x_fused = self.FFMs[2](x_cam, x_f)

        if self.training:
            text_labels = clip.tokenize(self.class_names).to("cuda")
            feature = self.get_text_feature(text_labels).to(feat.dtype)
            text_sim_matrix = F.cosine_similarity(feature.unsqueeze(0), feature.unsqueeze(1), dim=-1)
            batch_prototypes, prototype_loss = self.prototype_segmentation.compute_loss(x_fused, label)
            visual_sim_matrix = F.cosine_similarity(batch_prototypes.unsqueeze(0), batch_prototypes.unsqueeze(1),
                                                    dim=-1)

            text_sim_matrix = F.softmax(text_sim_matrix, dim=-1)
            visual_sim_matrix = F.log_softmax(visual_sim_matrix, dim=-1)

            kl_loss = F.kl_div(visual_sim_matrix, text_sim_matrix, reduction='none')
        else:
            kl_loss = torch.tensor(0.0).to(x_fused.device)
            prototype_loss = torch.tensor(0.0).to(x_fused.device)

        output = self.fpn_fuse(x_fused, x0_fused, x1_fused)
        multi_mask_output = self.sam._forward_sam_heads(
            image_embedding['vision_features'],
            high_res_features=image_embedding['backbone_fpn'][:2],
            multimask_output=multimask_output
        )
        m_output = multi_mask_output[1]
        _, fmc, fmh, fmw = output.size()
        m_output = m_output.reshape(m, b, fmc, fmh, fmw)
        m_output = torch.mean(m_output, dim=0)

        return m_output, output, prototype_loss, kl_loss
