from __future__ import print_function, division
import glob, pickle
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from shutil import copy2
from model import model as model_ori
from databases import SuperDB
from utils import *
from Train_options import Options


def main():
    # parse args
    global args
    global plotter
    global random_orders
    global meters, l_iteration

    args = Options().args
    
    # copy all files from experiment
    cwd = os.getcwd()
    for ff in glob.glob("*.py"):
        copy2(os.path.join(cwd, ff), os.path.join(args.folder, 'code'))

    # initialise seeds
    torch.manual_seed(1000)
    torch.cuda.manual_seed(1000)
    np.random.seed(1000)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    # parameters
    sigma = float(args.s)
    temperature = float(args.t)
    gradclip = int(args.gc)
    npts = int(args.npts)
    bSize = int(args.bSize)
    angle = float(args.angle)
    flip = eval(str(args.flip))
    tight = int(args.tight)

    list_ex = np.arange(bSize)
    random_orders = shuffle_list(list_ex)
    mymodel = model_ori
    model = mymodel(sigma=sigma, temperature=temperature, gradclip=gradclip, npts=npts, option=args.option,
                    size=args.size, path_to_check=args.checkpoint, args=args)

    losskeys = list(model.loss.keys())
    plotter = None
    meterskey = ['batch_time', 'data_time'] 
    meters = dict([(key, AverageMeter()) for key in meterskey])
    meters['losses'] = dict([(key, AverageMeter()) for key in losskeys])
    l_iteration = float(0.0)
    
    # plot number of parameters
    params = sum([p.numel() for p in filter(lambda p: p.requires_grad, model.GEN.parameters())])
    print('GEN # trainable parameters: {}'.format(params))
    params = sum([p.numel() for p in filter(lambda p: p.requires_grad, model.FAN.parameters())])
    print('FAN # trainable parameters: {}'.format(params))

    # define data
    video_dataset = SuperDB(path=args.data_path, sigma=sigma, size=args.size, flip=flip, angle=angle, tight=tight,
                            db=args.db, npts=npts)
    videoloader = DataLoader(video_dataset, batch_size=bSize, shuffle=True,
                             num_workers=int(args.num_workers), pin_memory=True)
    print('Number of workers is {:d}, and bSize is {:d}'.format(int(args.num_workers), bSize))

    # define optimizers
    lr_fan = args.lr_fan
    lr_gan = args.lr_gan
    print('Using learning rate {} for FAN, and {} for GAN'.format(lr_fan, lr_gan))
    optimizerFAN = torch.optim.Adam(model.FAN.parameters(), lr=lr_fan, betas=(0, 0.9), weight_decay=5*1e-4)
    schedulerFAN = torch.optim.lr_scheduler.StepLR(optimizerFAN, step_size=args.step_size, gamma=args.gamma)
    optimizerGEN = torch.optim.Adam(model.GEN.parameters(), lr=lr_gan, betas=(0, 0.9), weight_decay=5*1e-4)
    schedulerGEN = torch.optim.lr_scheduler.StepLR(optimizerGEN, step_size=args.step_size, gamma=args.gamma)
    myoptimizers = {'FAN': optimizerFAN, 'GEN': optimizerGEN}

    # path to save models and images
    path_to_model = os.path.join(args.folder, args.file)

    # train
    for epoch in range(0, 1000):
        schedulerFAN.step()
        schedulerGEN.step()
        train_epoch(videoloader, model, myoptimizers, epoch, bSize)
        model._save(path_to_model, epoch)


def train_epoch(dataloader, model, myoptimizers, epoch, bSize):
    itervideo = iter(dataloader)
    global l_iteration
    log_epoch = {}
    end = time.time()
    for i in range(0, 2500):

        all_data = next(itervideo, None)
        if all_data is None:
            itervideo = iter(dataloader)
            all_data = next(itervideo, None)
        elif all_data['Im'].shape[0] < bSize:
            itervideo = iter(dataloader)
            all_data = next(itervideo, None)

        model._set_batch(all_data)

        if len(all_data['Im']) == 1:
            continue

        # order_idx = np.asarray(random.choice(random_orders))
        order_idx = random_orders

        # - forward
        output = model.forward(myoptimizers, order_idx)
        myoptimizers = output['myoptimizers']

        # - update parameters
        for _, v in myoptimizers.items():
            v.step()

        meters['losses']['all_loss'].update(model.loss['all_loss'].item(), bSize)
        meters['losses']['perceptual'].update(model.loss['perceptual'].item(), bSize)
        meters['losses']['rec'].update(model.loss['rec'].item(), bSize)
        l_iteration = l_iteration + 1

        if i % 1000 == 0:
            # - plot some images
            allimgs = None
            for (ii, imtmp) in enumerate(all_data['Im'].to('cpu').detach()):
                improc = (255*imtmp.permute(1, 2, 0).numpy()).astype(np.uint8).copy()
                x = 4 * output['Points'][0][ii]

                for m in range(0, x.shape[0]):
                    # cv2.circle(improc, (int(x[m, 0]), int(x[m, 1])), 3, colors[m], -1)
                    cv2.circle(improc, (int(x[m, 0]), int(x[m, 1])), 3, (255, 255, 255), -1)
                if allimgs is None:
                    allimgs = np.expand_dims(improc, axis=0)
                else:
                    allimgs = np.concatenate((allimgs, np.expand_dims(improc, axis=0)))

            save = torch.nn.functional.interpolate(torch.from_numpy(allimgs/255.0).permute(0, 3, 1, 2), scale_factor=1)
            save_image(save, args.folder + '/image_{}_{}.png'.format(epoch, i))

        log_epoch[i] = model.loss
        meters['batch_time'].update(time.time()-end)
        end = time.time()

        if i % args.print_freq == 0:
            mystr = 'Epoch [{}][{}/{}] '.format(epoch, i, len(dataloader))
            mystr += 'Time {:.2f} ({:.2f}) '.format(meters['batch_time'].val, meters['batch_time'].avg)
            mystr += ' '.join(['{:s} {:.5f} ({:.5f}) '.format(k, meters['losses'][k].val, meters['losses'][k].avg)
                               for k in meters['losses'].keys()])
            print(mystr)
            with open(args.folder + '/args_' + args.file[0:-8] + '.txt', 'a') as f:
                print(mystr, file=f)

    with open(args.folder + '/args_' + args.file[0:-8] + '_' + str(epoch) + '.pkl', 'wb') as f:
        pickle.dump(log_epoch, f)


if __name__ == '__main__':
    main()


