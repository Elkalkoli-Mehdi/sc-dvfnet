import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SiameseEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        rn = models.resnet34(weights=weights)
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4

    def forward_single(self, x):
        c1 = self.stem(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5

    def forward(self, img_t1, img_t2):
        return self.forward_single(img_t1), self.forward_single(img_t2)


class FPN(nn.Module):
    def __init__(self, in_channels=(64, 128, 256, 512), out_ch=256):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(c * 2, out_ch, kernel_size=1) for c in in_channels
        ])
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ) for _ in in_channels
        ])

    def forward(self, feats_t1, feats_t2):
        combined = [torch.cat([f1, f2], dim=1) for f1, f2 in zip(feats_t1, feats_t2)]
        laterals = [l(c) for l, c in zip(self.lateral, combined)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        return [s(l) for s, l in zip(self.smooth, laterals)]


class SobelEdge(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        ex = F.conv2d(gray, self.kx, padding=1)
        ey = F.conv2d(gray, self.ky, padding=1)
        return torch.sqrt(ex**2 + ey**2 + 1e-6)


class SAT(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.sobel = SobelEdge()
        self.proj = nn.Sequential(nn.Conv2d(1, ch, 1), nn.Sigmoid())

    def forward(self, x, ref_img):
        edge = self.sobel(ref_img)
        edge = F.interpolate(edge, size=x.shape[2:], mode="bilinear", align_corners=False)
        att = self.proj(edge)
        return x + x * att


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SemanticBranch(nn.Module):
    def __init__(self, fpn_ch=256, n_classes=8, sem_ch=128):
        super().__init__()
        self.dec = DoubleConv(fpn_ch, sem_ch)
        self.sat = SAT(sem_ch)
        self.seg_head_t1 = nn.Conv2d(sem_ch, n_classes, 1)
        self.seg_head_t2 = nn.Conv2d(sem_ch, n_classes, 1)
        self.change_head = nn.Sequential(
            nn.Conv2d(sem_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, p2, ref_img):
        f = self.dec(p2)
        ref_r = F.interpolate(ref_img, size=f.shape[2:], mode="bilinear", align_corners=False)
        f = self.sat(f, ref_r)
        return self.seg_head_t1(f), self.seg_head_t2(f), self.change_head(f), f


class LinearCrossAttention(nn.Module):
    def __init__(self, geo_ch, sem_ch):
        super().__init__()
        self.proj_q = nn.Conv2d(geo_ch, sem_ch, 1)
        self.proj_k = nn.Conv2d(sem_ch, sem_ch, 1)
        self.proj_v = nn.Conv2d(sem_ch, sem_ch, 1)
        self.proj_out = nn.Conv2d(sem_ch, geo_ch, 1)
        self.norm = nn.GroupNorm(1, geo_ch)

    @staticmethod
    def _phi(x):
        return F.elu(x) + 1

    def forward(self, f_geo, f_sem):
        B, _, H, W = f_geo.shape
        f_sem_r = F.interpolate(f_sem, size=(H, W), mode="bilinear", align_corners=False)

        Q = self._phi(self.proj_q(f_geo)).view(B, -1, H * W)
        K = self._phi(self.proj_k(f_sem_r)).view(B, -1, H * W)
        V = self.proj_v(f_sem_r).view(B, -1, H * W)

        KV = torch.bmm(K, V.permute(0, 2, 1))
        QKV = torch.bmm(Q.permute(0, 2, 1), KV)
        K_sum = K.sum(dim=-1, keepdim=True)
        norm = (Q.permute(0, 2, 1) * K_sum.permute(0, 2, 1)).sum(dim=-1, keepdim=True) + 1e-6

        out = (QKV / norm).permute(0, 2, 1).view(B, -1, H, W)
        return self.norm(f_geo + self.proj_out(out))


class AffineDVFHead(nn.Module):
    def __init__(self, in_ch, img_size=(256, 256)):
        super().__init__()
        self.img_size = img_size
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, 6)
        nn.init.zeros_(self.fc.weight)
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

    def forward(self, x):
        B = x.shape[0]
        theta = self.fc(self.pool(x).view(B, -1)).view(B, 2, 3)
        grid = F.affine_grid(theta, (B, 1, *self.img_size), align_corners=False)
        ident = F.affine_grid(
            torch.eye(2, 3, device=x.device).unsqueeze(0).expand(B, -1, -1),
            (B, 1, *self.img_size),
            align_corners=False,
        )
        return (grid - ident).permute(0, 3, 1, 2)


class ProjectiveDVFHead(nn.Module):
    def __init__(self, in_ch, img_size=(256, 256)):
        super().__init__()
        self.img_size = img_size
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, 8)
        nn.init.zeros_(self.fc.weight)
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32))

    def forward(self, x):
        B = x.shape[0]
        h_par = self.fc(self.pool(x).view(B, -1))
        ones = torch.ones(B, 1, device=x.device)
        H_mat = torch.cat([h_par, ones], dim=1).view(B, 3, 3)

        H_, W_ = self.img_size
        ys = torch.linspace(-1, 1, H_, device=x.device)
        xs = torch.linspace(-1, 1, W_, device=x.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        ones_g = torch.ones_like(gx)

        coords = torch.stack([gx, gy, ones_g], -1).unsqueeze(0).expand(B, -1, -1, -1)
        cf = coords.reshape(B, -1, 3).permute(0, 2, 1)
        wc = torch.bmm(H_mat, cf)
        wc = (wc[:, :2] / (wc[:, 2:] + 1e-8)).permute(0, 2, 1).view(B, H_, W_, 2)
        ident = torch.stack([gx, gy], -1).unsqueeze(0).expand(B, -1, -1, -1)
        return (wc - ident).permute(0, 3, 1, 2)


class DenseDVFHead(nn.Module):
    def __init__(self, in_ch, img_size=(256, 256)):
        super().__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return F.interpolate(self.net(x), size=self.img_size, mode="bilinear", align_corners=False)


class ResidualConfidenceGate(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat):
        return self.gate(feat)


def warp(img, dvf):
    B, _, H, W = img.shape
    ys = torch.linspace(-1, 1, H, device=img.device)
    xs = torch.linspace(-1, 1, W, device=img.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    ident = torch.stack([gx, gy], -1).unsqueeze(0).expand(B, -1, -1, -1)
    grid = ident + dvf.permute(0, 2, 3, 1)
    return F.grid_sample(img, grid, align_corners=False, padding_mode="border")


def compose_dvf(dvf_a, dvf_b):
    return dvf_a + warp(dvf_b, dvf_a)


class SCDVFNet(nn.Module):
    def __init__(self, n_classes=8, img_size=(256, 256), fpn_ch=256, sem_ch=128, pretrained=True):
        super().__init__()
        self.img_size = img_size
        self.encoder = SiameseEncoder(pretrained=pretrained)
        self.fpn = FPN(in_channels=(64, 128, 256, 512), out_ch=fpn_ch)
        self.sem_branch = SemanticBranch(fpn_ch, n_classes, sem_ch)
        self.cross_attn = LinearCrossAttention(geo_ch=fpn_ch, sem_ch=sem_ch)
        self.affine_head = AffineDVFHead(fpn_ch, img_size)
        self.proj_head = ProjectiveDVFHead(fpn_ch, img_size)
        self.dense_head = DenseDVFHead(fpn_ch * 2, img_size)
        self.gate_p = ResidualConfidenceGate(fpn_ch)
        self.gate_d = ResidualConfidenceGate(fpn_ch)

    def forward(self, img_t1, img_t2):
        H, W = self.img_size

        feats_t1, feats_t2 = self.encoder(img_t1, img_t2)
        P2, P3, _, P5 = self.fpn(feats_t1, feats_t2)
        ref = (img_t1 + img_t2) / 2
        seg_t1, seg_t2, change_map, feat_sem = self.sem_branch(P2, ref)

        P5_att = self.cross_attn(P5, feat_sem)
        dvf_a = self.affine_head(P5_att)
        warped_t1_a = warp(img_t1, dvf_a)

        feats_wa, feats_t2_p = self.encoder(warped_t1_a, img_t2)
        P2_a, P3_a, _, P5_a = self.fpn(feats_wa, feats_t2_p)
        P5_a_att = self.cross_attn(P5_a, feat_sem)
        dvf_p_res = self.proj_head(P5_a_att)
        alpha_p = F.interpolate(
            self.gate_p(P5_a_att), size=(H, W), mode="bilinear", align_corners=False
        )
        dvf_p_gated = alpha_p * dvf_p_res
        dvf_ap = compose_dvf(dvf_a, dvf_p_gated)
        warped_t1_ap = warp(img_t1, dvf_ap)

        feats_wap, feats_t2_d = self.encoder(warped_t1_ap, img_t2)
        P2_ap, P3_ap, _, P5_ap = self.fpn(feats_wap, feats_t2_d)
        P5_ap_att = self.cross_attn(P5_ap, feat_sem)
        p23_ap = torch.cat([
            P2_ap,
            F.interpolate(P3_ap, size=P2_ap.shape[2:], mode="bilinear", align_corners=False)
        ], dim=1)
        dvf_d_res = self.dense_head(p23_ap)
        alpha_d = F.interpolate(
            self.gate_d(P5_ap_att), size=(H, W), mode="bilinear", align_corners=False
        )
        dvf_d_gated = alpha_d * dvf_d_res
        dvf_final = compose_dvf(dvf_ap, dvf_d_gated)
        warped_t1 = warp(img_t1, dvf_final)

        return {
            "warped_t1": warped_t1,
            "dvf_final": dvf_final,
            "dvf_affine": dvf_a,
            "dvf_proj_res": dvf_p_res,
            "dvf_proj_gated": dvf_p_gated,
            "dvf_dense_res": dvf_d_res,
            "dvf_dense_gated": dvf_d_gated,
            "gate_proj": alpha_p,
            "gate_dense": alpha_d,
            "seg_t1": seg_t1,
            "seg_t2": seg_t2,
            "change_map": change_map,
            "warped_t1_a": warped_t1_a,
            "warped_t1_ap": warped_t1_ap,
        }


def ncc_map(I, J, win=9):
    I_g = I.mean(dim=1, keepdim=True)
    J_g = J.mean(dim=1, keepdim=True)
    filt = torch.ones(1, 1, win, win, device=I.device) / (win ** 2)
    pad = win // 2
    u_I = F.conv2d(I_g, filt, padding=pad)
    u_J = F.conv2d(J_g, filt, padding=pad)
    u_II = F.conv2d(I_g * I_g, filt, padding=pad)
    u_JJ = F.conv2d(J_g * J_g, filt, padding=pad)
    u_IJ = F.conv2d(I_g * J_g, filt, padding=pad)
    cross = u_IJ - u_I * u_J
    var_I = u_II - u_I * u_I + 1e-5
    var_J = u_JJ - u_J * u_J + 1e-5
    return cross / torch.sqrt(var_I * var_J)


def smoothness_loss(dvf):
    dx = torch.abs(dvf[:, :, :, 1:] - dvf[:, :, :, :-1])
    dy = torch.abs(dvf[:, :, 1:, :] - dvf[:, :, :-1, :])
    return dx.mean() + dy.mean()


def jacobian_loss(dvf):
    dudx = dvf[:, 0, :, 1:] - dvf[:, 0, :, :-1]
    dvdy = dvf[:, 1, 1:, :] - dvf[:, 1, :-1, :]
    dudy = dvf[:, 0, 1:, :] - dvf[:, 0, :-1, :]
    dvdx = dvf[:, 1, :, 1:] - dvf[:, 1, :, :-1]
    H = min(dudx.shape[1], dvdy.shape[1])
    W = min(dudx.shape[2], dvdx.shape[2])
    det_J = ((1 + dudx[:, :H, :W]) * (1 + dvdy[:, :H, :W]) - dudy[:, :H, :W] * dvdx[:, :H, :W])
    return F.relu(-det_J).mean()


def change_sparsity_loss(change_map):
    return change_map.mean()


def gate_regularization_loss(gate_proj, gate_dense):
    def _var_penalty(alpha):
        return -(alpha.var())
    return _var_penalty(gate_proj) + _var_penalty(gate_dense)


class SCDVFNetLoss(nn.Module):
    def __init__(self, lam1=0.5, lam2=0.3, lam3=0.1, lam4=0.2, lam5=0.1, lam6=0.05, lam_gate=0.0):
        super().__init__()
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.lam4 = lam4
        self.lam5 = lam5
        self.lam6 = lam6
        self.lam_gate = lam_gate
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        warped = outputs["warped_t1"]
        img_t2 = targets["img_t2"]
        M = outputs["change_map"]

        M_r = F.interpolate(M, size=warped.shape[2:], mode="bilinear", align_corners=False)
        mask = (1 - M_r).detach()

        L_sim = ((1 - ncc_map(warped, img_t2)) * mask).mean()
        L_smooth = smoothness_loss(outputs["dvf_final"])

        def resize_gt(gt, target_size):
            if gt.shape[-2:] == target_size:
                return gt
            return F.interpolate(
                gt.float().unsqueeze(1), size=target_size, mode="nearest"
            ).squeeze(1).long()

        seg_size = outputs["seg_t1"].shape[2:]
        seg_gt_t1 = resize_gt(targets["seg_gt_t1"], seg_size)
        seg_gt_t2 = resize_gt(targets["seg_gt_t2"], seg_size)
        L_seg = (self.ce(outputs["seg_t1"], seg_gt_t1) + self.ce(outputs["seg_t2"], seg_gt_t2)) / 2

        L_inter = (
            ((1 - ncc_map(outputs["warped_t1_a"], img_t2)) * mask).mean()
            + ((1 - ncc_map(outputs["warped_t1_ap"], img_t2)) * mask).mean()
        )

        L_kp = torch.tensor(0.0, device=warped.device)
        if "keypoints_t1" in targets and targets["keypoints_t1"] is not None:
            kp_t1 = targets["keypoints_t1"]
            kp_t2 = targets["keypoints_t2"]
            dvf = outputs["dvf_final"]
            dvf_at = F.grid_sample(
                dvf, kp_t1.unsqueeze(1), align_corners=False
            ).squeeze(2).permute(0, 2, 1)
            L_kp = F.l1_loss(kp_t1 + dvf_at, kp_t2)

        L_jac = jacobian_loss(outputs["dvf_final"])
        L_sp = change_sparsity_loss(outputs["change_map"])

        L_gate = (
            gate_regularization_loss(outputs["gate_proj"], outputs["gate_dense"])
            if self.lam_gate > 0.0 else torch.tensor(0.0, device=warped.device)
        )

        L_total = (
            L_sim
            + self.lam1 * L_smooth
            + self.lam2 * L_seg
            + self.lam3 * L_inter
            + self.lam4 * L_kp
            + self.lam5 * L_jac
            + self.lam6 * L_sp
            + self.lam_gate * L_gate
        )

        return {
            "loss": L_total,
            "L_sim": L_sim,
            "L_smooth": L_smooth,
            "L_seg": L_seg,
            "L_intermediate": L_inter,
            "L_kp": L_kp,
            "L_jacobian": L_jac,
            "L_change_sparsity": L_sp,
            "L_gate": L_gate,
        }
