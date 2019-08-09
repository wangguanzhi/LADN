import argparse
import torch

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
        self.parser.add_argument('--landmark_file', type=str, default="../datasets/makeup/landmark.pk", help='path of the API landmark (to the pickle file)')
        self.parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'], help=' \
            This option set whether the network is running in training mode or in testing mode. \
            if opts.phase == train, train normally. \
            if opts.phase == test, only run one epoch of testing and exit. ')
        self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')
        self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
        self.parser.add_argument('--crop_size', type=int, default=256, help='cropped image size for training')
        self.parser.add_argument('--input_dim_a', type=int, default=3, help='# of input channels for domain A')
        self.parser.add_argument('--input_dim_b', type=int, default=3, help='# of input channels for domain B')
        self.parser.add_argument('--nThreads', type=int, default=8, help='# of threads for data loader')
        self.parser.add_argument('--no_extreme', action='store_true', help='whether to exclude the extreme makeup images in the after makeup domain')
        self.parser.add_argument('--extreme_only', action='store_true', help='whether to use only the extreme makeup images in the after makeup domain')

        self.parser.add_argument('--test_forward', action="store_true", help="\
            If opts.phase == test, this flag will be set. \
            if opts.test_forward is True, run over the testing set every opts.test_interval epochs. \
            if opts.test_forward is False, train the network only. ")
        self.parser.add_argument('--test_random', action='store_true', help='If set, randomly sample some pairs for test forward. Otherwise, iterate over all pair of before and after makeup exmaples')
        self.parser.add_argument('--test_size', type=int, default=100, help="size of the test dataset used for test_forward (effective whether test_random is set or not)")
        self.parser.add_argument('--test_interval', type=int, default=200, help="interval of the test forward (in # of epoch)")

        self.parser.add_argument('--interpolate_forward', action="store_true", help="whether to turn on interpolation mode")
        self.parser.add_argument('--interpolate_num', type=int, default=8, help="how many images for interpolation")

        # generator related
        self.parser.add_argument('--num_residule_block', type=int, default=9, help='number of residule blocks for fusion generator')

        # discriminator related
        self.parser.add_argument('--style_dis', action='store_true', help='whether to add style discriminator')
        self.parser.add_argument('--local_style_dis', action='store_true', help='whether to add local style discriminator')
        self.parser.add_argument('--local_laplacian_loss', action='store_true', help='whether to use local laplacian loss')
        self.parser.add_argument('--local_laplacian_loss_weight', type=float, default=1.0, help='weight for local laplacian loss')
        self.parser.add_argument('--local_smooth_loss', action='store_true', help='whether to use local smooth loss')
        self.parser.add_argument('--local_smooth_loss_weight', type=float, default=1.0, help='weight for local smooth loss')
        self.parser.add_argument('--dis_n_layer', type=int, default=5, help='number of layers for style discriminator')
        self.parser.add_argument('--n_local', type=int, default=3, help="Number of the local parts for local discriminators (see details in helpers.py)")

        # loss related
        self.parser.add_argument('--style_d_ls_weight', type=float, default=2.0, help='weight of the loss_D_Style')
        self.parser.add_argument('--style_g_ls_weight', type=float, default=2.0, help='weight of the loss_G_GAN_style')
        self.parser.add_argument('--recon_weight', type=float, default=8.0, help='weight of the reconstruction loss')

        # ouptput related
        self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
        self.parser.add_argument('--display_dir', type=str, default='../logs', help='path for saving display results')
        self.parser.add_argument('--result_dir', type=str, default='../results', help='path for saving result images and models')
        self.parser.add_argument('--display_freq', type=int, default=0, help='freq (iteration) of display\
            (default 0 means display one image per epoch)')
        self.parser.add_argument('--loss_save_freq', type=int, default=1, help="freq(iteration) of saving losses for tensorboardX display")
        self.parser.add_argument('--img_save_freq', type=int, default=5, help='freq (epoch) of saving images')
        self.parser.add_argument('--model_save_freq', type=int, default=50, help='freq (epoch) of saving models')
        self.parser.add_argument('--no_display_img', action='store_true', help='specified if no display')

        # training related
        self.parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
        self.parser.add_argument('--dis_norm', type=str, default='None', help='normalization layer in discriminator [None, Instance]')
        self.parser.add_argument('--dis_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
        self.parser.add_argument('--n_ep', type=int, default=1200, help='number of epochs') # 400 * d_iter
        self.parser.add_argument('--n_ep_decay', type=int, default=600, help='epoch start decay learning rate, set -1 if no decay') # 200 * d_iter
        self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--d_iter', type=int, default=3, help='# of iterations for updating content discriminator')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu: e.g. 0 ,use -1 for CPU')
        self.parser.add_argument('--backup_gpu', type=int, default=1, help='backup gpu: e.g. 1 ,use -1 for CPU, useful when one gpu does not have enough memory')


    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))

        # set gpu
        torch.cuda.set_device(self.opt.gpu)

        return self.opt
