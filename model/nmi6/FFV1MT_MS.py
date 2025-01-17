import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model.nmi6.MT import FeatureTransformer
from networkfactory.convfactory import conv
from torch.cuda.amp import autocast as autocast
from utils.flow_utils import viz_img_seq, save_img_seq, plt_show_img_flow
from copy import deepcopy
from model.nmi6.V1 import V1SingleScale, V1
import matplotlib.pyplot as plt
from model.nmi6.highorderv1 import HigherOrderMotionDetector


def plt_attention(attention, h, w):
    col = len(attention) // 2
    fig = plt.figure(figsize=(10, 5))

    for i in range(len(attention)):
        viz = attention[i][0, :, :, h, w].detach().cpu().numpy()
        # viz = viz[7:-7, 7:-7]
        if i == 0:
            viz_all = viz
        else:
            viz_all = viz_all + viz

        ax1 = fig.add_subplot(2, col + 1, i + 1)
        img = ax1.imshow(viz, cmap="rainbow", interpolation="bilinear")
        plt.colorbar(img, ax=ax1)
        ax1.scatter(w, h, color='red')
        plt.title("Connectivity of Iteration %d" % (i + 1))
        ax1.set_xticks([])
        ax1.set_yticks([])

    ax1 = fig.add_subplot(2, col + 1, 2 * (col + 1))
    img = ax1.imshow(viz_all, cmap="rainbow", interpolation="bilinear")
    plt.colorbar(img, ax=ax1)
    ax1.scatter(w, h, color='red')
    plt.title("Mean Attention")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # remove all ticl

    plt.show()


# linear flow decoder
class FlowDecoder(nn.Module):
    # can reduce 25% of training time.
    def __init__(self, ch_in):
        super(FlowDecoder, self).__init__()
        self.conv1 = conv(ch_in, 256, kernel_size=1, isReLU=False)
        self.conv2 = conv(256, 128, kernel_size=1, isReLU=False)
        self.conv3 = conv(128, 64, kernel_size=1, isReLU=False)
        self.conv4 = conv(64, 32, kernel_size=1, isReLU=False)

        self.feat_dim = 32
        self.predict_flow = conv(64 + 32, 2, isReLU=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        flow = self.predict_flow(torch.cat([x4, x3], dim=1))
        return flow


def gate_loss(gates):
    loss_sum = 0.0
    gamma = 0.85
    length = len(gates)
    sec_obj = torch.ones_like(gates[0][0]).sum().unsqueeze(0) * 0.5

    for i in range(len(gates)):
        weight = gamma ** (length - i - 1)
        gate = gates[i]
        # make value close to 0 or 1
        loss = 0.5 - (gate - 0.5).abs().mean()  # force to 0 or 1
        loss_obj = (torch.sum(gate, dim=[1, 2, 3]) - sec_obj).abs().mean()
        loss_sum += loss * weight
        loss_sum += loss_obj * weight
    return loss_sum


# 3dcnn, v1, mt, upsample

class FFV1DNNV2(nn.Module):
    def __init__(self,
                 num_scales=8,
                 v1_1st_cells=256,
                 v1_2nd_cells=256,
                 upsample_factor=8,
                 mt_channels=256,
                 scale_factor=16,
                 num_layers=6,
                 num_frames=[11, 15]  # 30FPS, duration=0.5s, temporal kernel size=6+1
                 ):
        super(FFV1DNNV2, self).__init__()

        self.ffv2 = nn.Sequential(
            HigherOrderMotionDetector(output_dim=v1_2nd_cells, n_frames=num_frames[1], kernel_size=8),
            V1SingleScale(v1_1st_cells, norm_fn='instance', n_frames=num_frames[1], kernel_size=8,
                          kernel_radius=7))
        self.ffv1 = V1(spatial_num=v1_1st_cells // num_scales, scale_num=num_scales, scale_factor=scale_factor,
                       kernel_radius=7, num_ft=v1_1st_cells // num_scales,
                       kernel_size=6, average_time=True, n_frames=num_frames[0])

        self.v1_kz = 7
        self.scale_factor = scale_factor
        scale_each_level = np.exp(1 / (num_scales - 1) * np.log(1 / scale_factor))
        self.scale_num = num_scales
        self.scale_each_level = scale_each_level
        v1_channel = v1_1st_cells
        self.num_scales = num_scales
        self.MT_channel = mt_channels

        assert self.MT_channel == v1_channel
        self.upsample_factor = upsample_factor
        self.num_layers = num_layers
        # convex upsampling: concat feature0 and flow as input
        self.upsampler = nn.Sequential(nn.Conv2d(2 + v1_1st_cells, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, upsample_factor ** 2 * 9, 3, 1, 1))

        self.decoder = FlowDecoder(v1_1st_cells)
        self.MT = FeatureTransformer(d_model=256, num_layers=self.num_layers)
        self.fuse = nn.Sequential(nn.Conv2d(2 * v1_1st_cells, 2 * v1_1st_cells, 1, 1, 0, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(2 * v1_1st_cells, v1_1st_cells, 3, 1, 1, bias=False))

    # 2*2*8*scale`
    def upsample_flow(self, flow, feature, upsampler=None, bilinear=False, upsample_factor=4):
        if bilinear:
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor
        else:
            # convex upsampling
            concat = torch.cat((flow, feature), dim=1)
            mask = upsampler(concat)
            b, flow_channel, h, w = flow.shape
            mask = mask.view(b, 1, 9, upsample_factor, upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
            mask = torch.softmax(mask, dim=2)

            up_flow = F.unfold(upsample_factor * flow, [3, 3], padding=1)
            up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

            up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
            up_flow = up_flow.reshape(b, flow_channel, upsample_factor * h,
                                      upsample_factor * w)  # [B, 2, K*H, K*W]

        return up_flow

    @torch.no_grad()
    def data_preprocess(self, image_list):
        ' convert to 0-1 '
        if image_list[0].max() > 10:
            image_list = [img / 255.0 for img in image_list]  # [B, 3, H, W]  0-1
        # if is RGB, convert to gray
        if image_list[0].shape[1] == 1:
            image_list_rgb = deepcopy([img.repeat(1, 3, 1, 1) for img in image_list])
        else:
            image_list_rgb = deepcopy(image_list)
        if image_list[0].shape[1] == 3:
            # convert to gray using transform Gray = R*0.299 + G*0.587 + B*0.114
            image_list = [img[:, 0, :, :] * 0.299 + img[:, 1, :, :] * 0.587 + img[:, 2, :, :] * 0.114 for img in
                          image_list]
            image_list = [img.unsqueeze(1) for img in image_list]
        return image_list, image_list_rgb

    def forward(self, image_list, mix_enable=True, layer=None):
        if layer is not None:
            self.MT.num_layers = layer
            self.num_layers = layer
        results_dict = {}
        padding = self.v1_kz * self.scale_factor
        image_list, image_list_rgb = self.data_preprocess(image_list)
        image_list = image_list[2:-2]
        image_list_rgb = image_list_rgb
        assert len(image_list) == 11 and len(image_list_rgb) == 15
        B, _, H, W = image_list[0].shape
        MT_size = (H // 8, W // 8)
        with autocast(enabled=mix_enable):
            # with torch.no_grad(): #
            st_component1 = self.ffv1(image_list)

            st_component2 = self.ffv2(image_list_rgb)
            # v1 single
            flow_0 = [self.decoder(st_component1)]
            # extract high-order feature
            value1, attn1 = self.MT.forward_save_mem(st_component1)

            flow_1 = [self.decoder(feature) for feature in value1]
            flow_1_bi = [self.upsample_flow(flows, feature=None, bilinear=True, upsample_factor=8) for flows in
                         flow_0 + flow_1]
            flow_1_up = [self.upsample_flow(flows, upsampler=self.upsampler, feature=attn, upsample_factor=8)
                         for flows, attn in zip(flow_1, attn1)]
            st_component1 = st_component1.detach()

            # st_component1 = st_component1.detach()
            # concat v1 single and v1 high-order
            st_component = self.fuse(torch.cat([st_component1, st_component2], dim=1)).square()
            value_all, attn = self.MT.forward_save_mem(st_component)
            flows_all = [self.decoder(feature) for feature in value_all]
            flows_up = [self.upsample_flow(flows, upsampler=self.upsampler, feature=attn, upsample_factor=8) for
                        flows, attn in zip(flows_all, attn)]
            flow_bi = [self.upsample_flow(flows, feature=None, bilinear=True, upsample_factor=8) for flows in
                       flows_all]
            results_dict["flow_seq"] = flows_up
            results_dict["flow_seq_bi"] = flow_bi
            results_dict["flow_seq_1_bi"] = flow_1_bi
            results_dict["flow_seq_1"] = flow_1_up
            results_dict["flow_attn"] = attn
            results_dict["flow_attn_1"] = attn1
            if layer == 0:
                results_dict["flow_seq"] = flow_1_bi
                results_dict["flow_seq_1"] = flow_1_bi

        return results_dict

    def forward_test(self, image_list, mix_enable=True, layer=6):
        if layer is not None:
            self.MT.num_layers = layer
            self.num_layers = layer
        results_dict = {}
        padding = self.v1_kz * self.scale_factor
        with torch.no_grad():
            if image_list[0].max() > 10:
                image_list = [img / 255.0 for img in image_list]  # [B, 1, H, W]  0-1

        B, _, H, W = image_list[0].shape
        MT_size = (H // 8, W // 8)
        with autocast(enabled=mix_enable):
            st_component = self.ffv1(image_list)
            # viz_img_seq(image_scale, if_debug=True)
            if self.num_layers == 0:
                motion_feature = [st_component]
                flows = [self.decoder(feature) for feature in motion_feature]
                flows_up = [self.upsample_flow(flow, feature=None, bilinear=True, upsample_factor=8) for flow in flows]
                results_dict["flow_seq"] = [flows_up]
                return results_dict
            motion_feature, attn, _ = self.MT.forward_save_mem(st_component)
            flow_v1 = self.decoder(st_component)
            flows = [flow_v1] + [self.decoder(feature) for feature in motion_feature]
            flows_bi = [self.upsample_flow(flow, feature=None, bilinear=True, upsample_factor=8) for flow in flows]
            flows_up = [flows_bi[0]] + \
                       [self.upsample_flow(flows, upsampler=self.upsampler_1, feature=attn, upsample_factor=8) for
                        flows, attn in zip(flows[1:], attn)]
            assert len(flows_bi) == len(flows_up)
            results_dict["flow_seq"] = flows_up
            results_dict["flow_seq_bi"] = flows_bi
        return results_dict

    def forward_viz(self, image_list, mix_enable=True, layer=None, channel=2):
        if layer is not None:
            self.MT.num_layers = layer
            self.num_layers = layer
        results_dict = {}
        padding = self.v1_kz * self.scale_factor
        image_list_ori = deepcopy(image_list)
        image_list, image_list_rgb = self.data_preprocess(image_list)
        image_list = image_list[2:-2]
        image_list_rgb = image_list_rgb
        assert len(image_list) == 11 and len(image_list_rgb) == 15
        B, _, H, W = image_list[0].shape
        MT_size = (H // 8, W // 8)
        with autocast(enabled=mix_enable):
            # with torch.no_grad(): #
            st_component1 = self.ffv1(image_list)
            st_component2 = self.ffv2(image_list_rgb)
            self.ffv1.visualize_activation(st_component1)
            if channel == 2:
                st_component = self.fuse(torch.cat([st_component1, st_component2], dim=1)).square()

            else:
                st_component = st_component1
            upsampler = self.upsampler

            # viz_img_seq(image_scale, if_debug=True)
            motion_feature, attn, attn_viz = self.MT(st_component)
            flow_v1 = self.decoder(st_component1)

            flows = [flow_v1] + [self.decoder(feature) for feature in motion_feature]
            flows_bi = [self.upsample_flow(flow, feature=None, bilinear=True, upsample_factor=8) for flow in flows]
            flows_up = [flows_bi[0]] + \
                       [self.upsample_flow(flows, upsampler=upsampler, feature=attn, upsample_factor=8) for
                        flows, attn in zip(flows[1:], attn)]
            assert len(flows_bi) == len(flows_up)
            results_dict["flow_seq"] = flows_up

        plt_show_img_flow(image_list_ori, results_dict["flow_seq"])
        plt_attention(attn_viz, h=30, w=60)
        print("done")

        return results_dict

    def forward_activation(self, image_list, layer=None, keep_ori=False):
        if layer is not None:
            self.MT.num_layers = layer
        results_dict = {}
        image_list, image_list_rgb = self.data_preprocess(image_list)
        image_list = image_list[2:-2]
        image_list_rgb = image_list_rgb
        assert len(image_list) == 11 and len(image_list_rgb) == 15
        B, _, H, W = image_list[0].shape
        MT_size = (H // 8, W // 8)
        st_component = self.ffv1(image_list)
        # viz_img_seq(image_scale, if_debug=True)
        motion_feature, attn, _ = self.MT(st_component)
        motion_feature = [st_component] + motion_feature
        if keep_ori:
            results_dict["activation"] = motion_feature
        else:
            results_dict["activation"] = [
                torch.mean(motion_feature[:, :, 7:-7, 7:-7], dim=[2, 3],
                           keepdim=False).detach().cpu().squeeze().numpy()
                for motion_feature in motion_feature]
        return results_dict

    def num_parameters(self):
        return sum(
            [p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    @staticmethod
    def demo(file=None):
        print(torch.__version__)
        torch.backends.cudnn.benchmark = True
        print(torch.backends.cudnn.is_available())
        import time
        from utils import torch_utils as utils
        frame_list = [torch.randn([4, 3, 384, 512], device="cuda")] * 15
        model = FFV1DNNV2(num_scales=8, scale_factor=16, v1_1st_cells=256, v1_2nd_cells=256, upsample_factor=8,
                          num_layers=6, mt_channels=256, num_frames=[11, 15]).cuda()
        if file is not None:
            model = utils.restore_model(model, file)
        print(model.num_parameters())
        for i in range(100):
            start = time.time()
            output = model.forward(frame_list, layer=7, mix_enable=True)
            # print(output["flow_seq"][-1])
            torch.mean(output["flow_seq"][-1]).backward()
            # gate loss
            print(torch.any(torch.isnan(output["flow_seq"][-1])))
            end = time.time()
            print(end - start)
            print("#================================++#")


if __name__ == '__main__':
    FFV1DNNV2.demo(None)
