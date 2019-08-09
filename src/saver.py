import os
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image

# tensor to PIL Image
def tensor2img(img):
    img = img[0].cpu().float().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    return img.astype(np.uint8)

# save a set of images
def save_imgs(imgs, names, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for img, name in zip(imgs, names):
        img = tensor2img(img)
        img = Image.fromarray(img)
        img.save(os.path.join(path, name + '.png'))

class Saver():
    def __init__(self, opts, dataset_size = None):
        self.display_dir = os.path.join(opts.display_dir, opts.name)
        self.model_dir = os.path.join(opts.result_dir, opts.name)
        self.image_dir = os.path.join(self.model_dir, 'images')
        self.test_image_dir = os.path.join(self.model_dir, 'test_images')

        if dataset_size is None or opts.display_freq != 0:
            self.display_freq = opts.display_freq
        else:
            self.display_freq = dataset_size + 1

        self.loss_save_freq = opts.loss_save_freq

        self.img_save_freq = opts.img_save_freq
        self.model_save_freq = opts.model_save_freq

        # make directory
        if not os.path.exists(self.display_dir):
            os.makedirs(self.display_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        if not os.path.exists(self.test_image_dir):
            os.makedirs(self.test_image_dir)

        # create tensorboard writer
        self.writer = SummaryWriter(log_dir=self.display_dir)

    # write losses and images to tensorboard
    def write_display(self, total_it, model):
        # write loss
        if (total_it + 1) % self.loss_save_freq == 0:
            members = [attr for attr in dir(model) if not callable(getattr(model, attr)) and not attr.startswith("__") and 'loss' in attr]
            for m in members:
                self.writer.add_scalar(m, getattr(model, m), total_it)

        # write img
        if (total_it + 1) % self.display_freq == 0:
            image_dis = torchvision.utils.make_grid(model.image_display, nrow=model.image_display.size(0)//2)/2 + 0.5
            # image_dis = np.transpose(image_dis.numpy(), (1, 2, 0)) * 255
            '''The above line will incur a bug in Image, probably because of the difference in version'''
            image_dis = image_dis.numpy() * 255
            image_dis = image_dis.astype('uint8')
            # print(type(image_dis), image_dis.shape)
            self.writer.add_image('Image', image_dis, total_it)

    # save result images
    def write_img(self, ep, model):
        if (ep + 1) % self.img_save_freq == 0:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_%05d.jpg' % (self.image_dir, ep)
            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)
        elif ep == -1:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_last.jpg' % (self.image_dir, ep)
            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

    def save_interpolate_img(self, ep, index, model, interpolate_num, index_a, index_b):
        path = '%s/interpolate_%05d_%05d_%05d' % (self.image_dir, index_a, index_b, ep)
        if not os.path.exists(path):
            os.mkdir(path)

        for i in range(interpolate_num+1):
            img = model.normalize_image(getattr(model, 'transfer'+str(i))).detach()[0:1, ::]
            img = tensor2img(img)
            img = Image.fromarray(img)
            img.save(os.path.join(path, 'transfer'+str(i)+'.jpg'))

        names = ['before','afterA','afterB']
        images = ['real_A_encoded','real_B1_encoded','real_B2_encoded']
        for i in range(len(names)):
            img = model.normalize_image(getattr(model, images[i])).detach()[0:1, ::]
            img = tensor2img(img)
            img = Image.fromarray(img)
            img.save(os.path.join(path, names[i]+'.jpg'))

    # save the individual result of test_forward during the progress of training
    def save_test_img(self, ep, index, model, index_a, index_b):
        names = ['source','transfer','random_makeup','source_recon','source_cycle_recon',\
                 'reference','demakeup','random_demakeup','reference_recon','reference_cycle_recon',\
                 'blend']
        images = ['real_A_encoded','fake_B_encoded','fake_B_random','fake_AA_encoded','fake_A_recon',\
                  'real_B_encoded','fake_A_encoded','fake_A_random','fake_BB_encoded','fake_B_recon',\
                  'real_C_encoded']

        path = '%s/test_%05d_%05d_%05d' % (self.test_image_dir, index_b, index_a, ep)
        if not os.path.exists(path):
            os.mkdir(path)

        for i in range(11):
            img = model.normalize_image(getattr(model, images[i])).detach()[0:1, ::]
            img = tensor2img(img)
            img = Image.fromarray(img)
            img.save(os.path.join(path, names[i]+'.jpg'))

    # save the result of test_forward during the progress of training
    def write_test_img(self, ep, index, model, index_a, index_b):
        assembled_images = model.assemble_outputs()
        img_filename = '%s/test_%05d_%05d_%05d.jpg' % (self.image_dir, index_b, index_a, ep)
        torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

    # save model
    def write_model(self, ep, total_it, model):
        if (ep + 1) % self.model_save_freq == 0:
            print('--- save the model @ ep %d ---' % (ep))
            model.save('%s/%05d.pth' % (self.model_dir, ep), ep, total_it)
        elif ep == -1:
            model.save('%s/last.pth' % self.model_dir, ep, total_it)
