from torchvision.models.vgg import vgg16
from MTFAN import FAN, convertLayer, GeoDistill
from utils import *
from torch.autograd import Variable


def matching_loss(pred, gt, loss_type="L2", mask=None):
    if isinstance(pred, list):
        loss = 0
        for each_pred in pred:
            each_pred = each_pred.squeeze()
            s = each_pred.size()
            dis = each_pred - gt
            if len(s) == 4:
                dis = torch.abs(dis).sum(3).sum(2).sum(1)
            else:
                dis = torch.abs(dis).sum(2).sum(1)
            loss += dis
        return loss

    s = pred.size()
    if loss_type == "L2":
        dis = pred - gt
        dis = (dis ** 2).sum(2).sum(1)

    elif loss_type == "L1":
        dis = pred - gt
        if len(s) == 4:
            dis = torch.abs(dis).sum(3).sum(2).sum(1)
        else:
            # dis = torch.abs(dis).sum(2) * mask
            dis = torch.abs(dis).sum(2)
            # dis = dis.mean(1)
            dis = dis.sum(1)
    return dis.mean(0)


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=66, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        self.downsampling = nn.Sequential(*layers)

        layers = []
        curr_dim = curr_dim + c_dim
        layers.append(nn.BatchNorm2d(curr_dim, affine=True, track_running_stats=True))
        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, maps):
        '''

        :param x: Transformed input image [B, 3, H_ori, W_oir]
        :param maps: Obtained heatmaps [B, N, H, W]
        :return:
        '''
        x = self.downsampling(x)
        x = torch.cat((x, maps),1)
        return self.main(x)


class model():

    def __init__(self, sigma=0.5, temperature=0.5, gradclip=1, npts=10, option='incremental', size=128,
                 path_to_check='checkpoint_fansoft/fan_109.pth', args=None):
        self.npoints = npts
        self.gradclip = gradclip
        
        # - define FAN
        self.FAN = FAN(1, n_points=self.npoints)

        if not option == 'scratch':
            net_dict = self.FAN.state_dict()
            pretrained_dict = torch.load(path_to_check, map_location='cuda')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict)}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrained_dict[k].shape == net_dict[k].shape}
            net_dict.update(pretrained_dict)
            self.FAN.load_state_dict(net_dict, strict=True)

            if option == 'incremental':
                print('Option is incremental')
                self.FAN.apply(convertLayer)

        # - define Bottleneck
        self.BOT = GeoDistill(sigma=sigma, temperature=temperature, out_res=int(size/4))

        # - define GEN      
        self.GEN = Generator(conv_dim=32, c_dim=self.npoints)

        # # Load pretrained model
        if args.resume_folder:
            path_fan = '{}/model_{}.fan.pth'.format(args.resume_folder, args.resume_epoch)
            path_gen = '{}/model_{}.gen.pth'.format(args.resume_folder, args.resume_epoch)
            self._resume(path_fan, path_gen)

        # - multiple GPUs
        if torch.cuda.device_count() > 1:
            self.FAN = torch.nn.DataParallel(self.FAN)
            self.BOT = torch.nn.DataParallel(self.BOT)
            self.GEN = torch.nn.DataParallel(self.GEN)

        self.FAN.to('cuda').train()
        self.BOT.to('cuda').train()
        self.GEN.to('cuda').train()

        # - VGG for perceptual loss
        self.loss_network = LossNetwork(torch.nn.DataParallel(vgg16(pretrained=True)))\
            if torch.cuda.device_count() > 1 else LossNetwork(vgg16(pretrained=True))
        self.loss_network.eval()
        self.loss_network.to('cuda')
        self.loss = dict.fromkeys(['all_loss', 'rec', 'perceptual'])
        self.A = None
        
        # - define losses for reconstruction
        self.SelfLoss = torch.nn.MSELoss().to('cuda')
        self.PerceptualLoss = torch.nn.MSELoss().to('cuda')

        self.heatmap = HeatMap(32, 0.5).cuda()

    def _perceptual_loss(self, fake_im, real_im):
        vgg_fake = self.loss_network(fake_im)
        vgg_target = self.loss_network(real_im)
        perceptualLoss = 0
        for vgg_idx in range(0, 4):
            perceptualLoss += self.PerceptualLoss(vgg_fake[vgg_idx], vgg_target[vgg_idx].detach())
        return perceptualLoss

    def _resume(self, path_fan, path_gen):
        self.FAN.load_state_dict(torch.load(path_fan))
        self.GEN.load_state_dict(torch.load(path_gen))

    def _save(self, path_to_models, epoch):
        torch.save(self.FAN.state_dict(), path_to_models + str(epoch) + '.fan.pth')
        torch.save(self.GEN.state_dict(), path_to_models + str(epoch) + '.gen.pth')

    def _set_batch(self, data):
        self.A = {k: Variable(data[k], requires_grad=False)
            .to('cuda') for k in data.keys() if type(data[k]).__name__  == 'Tensor'}

    def forward(self, myoptimizers, order_idx):
        self.GEN.zero_grad()
        self.FAN.zero_grad()

        Im_3 = self.A['Im'][order_idx]
        H_3 = self.FAN(Im_3)  # [B, N, H, W] (N landmarks)
        H_3, Pts_3 = self.BOT(H_3)  # Change it to landmark locations

        H = self.FAN(self.A['Im'])   # [B, N, H, W] (N landmarks)
        H, Pts = self.BOT(H)  # Change it to landmark locations
        H_P = self.FAN(self.A['ImP'])  # [B, N, H, W] (N landmarks)
        H_P, Pts_P = self.BOT(H_P)  # Change it to landmark locations

        X_3 = 0.5 * (self.GEN(self.A['Im'], H_3)+1)
        X_P_3 = 0.5 * (self.GEN(self.A['ImP'], H_3)+1)

        X_P = 0.5 * (self.GEN(X_3, H_P) + 1)  # [B, 3, H_ori, W_ori]
        X = 0.5 * (self.GEN(X_P_3, H) + 1)  # [B, 3, H_ori, W_ori]

        self.loss['perceptual'] = self._perceptual_loss(X, self.A['Im']) + self._perceptual_loss(X_P, self.A['ImP'])
        self.loss['rec'] = self.SelfLoss(X, self.A['Im']) + self.SelfLoss(X_P, self.A['ImP'])
        self.loss['all_loss'] = self.loss['perceptual'] + self.loss['rec']
        self.loss['all_loss'].backward()

        if self.gradclip:
            torch.nn.utils.clip_grad_norm_(self.FAN.parameters(), 1, norm_type=2)
            torch.nn.utils.clip_grad_norm_(self.GEN.parameters(), 1, norm_type=2)

        return {'Heatmap': [H, H_P, H_3],
                'Reconstructed': [X, X_P, X_3, X_P_3],
                'Points': [Pts, Pts_P, Pts_3],
                'myoptimizers': myoptimizers}

    
    

