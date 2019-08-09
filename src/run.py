import torch
from options import Options
from dataset import dataset_makeup
from model import LADN
from saver import Saver

'''
opts.phase
    This option set whether the network is running in training mode or in testing mode.
    if opts.phase == "train", train normally.
    if opts.phase == "test", only run one epoch of testing and exit.

opts.test_forward
    If opts.phase == "test", this flag will be set.
    if opts.test_forward is True, run over the testing set every opts.test_interval epochs.
    if opts.test_forward is False, train the network only.

opts.interpolate_forward
    It is independent of the above two options.
    If opts.interpolate_forward is True, interpolation will also be run in every test forward
'''

def main():
    # parse options
    parser = Options()
    opts = parser.parse()

    # If the overall mode is testing
    if opts.phase == "test":
        opts.test_forward = True

    # daita loader
    print('\n--- load dataset ---')
    dataset = dataset_makeup(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)

    # A separate dataset for test forward
    if opts.test_forward:
        print("creating dataset_test")
        dataset_test = dataset_makeup(opts, mode = 'test')
    # Another separate dataset for interpolation forwarding
    if opts.interpolate_forward:
        print("creating dataset_interpolate")
        dataset_interpolate = dataset_makeup(opts, mode = "interpolate")

    # model
    print('\n--- load model ---')
    model = LADN(opts)
    if opts.resume is None:
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)

    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d'%(ep0))

    # saver for display and output
    saver = Saver(opts, len(dataset))

    # Run only one epoch when testing
    if opts.phase == "test":
        opts.n_ep = ep0 + 1
        opts.test_interval = 1

    # train
    print('\n--- train ---')
    max_it = 500000
    for ep in range(ep0, opts.n_ep):
        # Run the testing set every test_interval epochs
        if opts.test_forward and (ep+1) % opts.test_interval == 0:
            print("starting forward for testing images")
            for i in range(len(dataset_test)):
                if (i+1) % 10 == 0:
                    print("forwarding %d/%d images for testing." % (i, len(dataset_test)))
                data = dataset_test[i]
                device = torch.device('cuda:{}'.format(opts.backup_gpu)) if opts.gpu>=0 else torch.device('cpu')
                images_a = data['img_A'].to(device).detach().unsqueeze(0)
                images_b = data['img_B'].to(device).detach().unsqueeze(0)
                images_c = data['img_C'].to(device).detach().unsqueeze(0)
                index_a = int(data['index_A'])
                index_b = int(data['index_B'])

                model.test_forward(images_a, images_b, images_c)
                saver.write_test_img(ep, i, model, index_a = index_a, index_b = index_b)
                saver.save_test_img(ep, i, model, index_a = index_a, index_b = index_b)

        if opts.interpolate_forward and (ep+1) % opts.test_interval == 0:
            print("starting forward for interpolated images")
            for i in range(len(dataset_interpolate)):
                if (i+1) % 10 == 0:
                    print("forwarding %d/%d images for interpolating." % (i, len(dataset_interpolate)))
                data = dataset_interpolate[i]
                device = torch.device('cuda:{}'.format(opts.backup_gpu)) if opts.gpu>=0 else torch.device('cpu')
                images_a = data['img_A'].to(device).detach().unsqueeze(0)
                images_b1 = data['img_B'].to(device).detach().unsqueeze(0)
                images_b2 = data['img_C'].to(device).detach().unsqueeze(0)
                images_b = torch.cat([images_b1, images_b2], dim=0)
                index_a = int(data['index_A'])
                index_b = int(data['index_B'])

                model.interpolate_forward(images_a, images_b1, images_b2)
                saver.save_interpolate_img(ep, i, model, opts.interpolate_num, index_a = index_a, index_b = index_b)

        if opts.phase == "train":
            for it, data in enumerate(train_loader):
                device = torch.device('cuda:{}'.format(opts.backup_gpu)) if opts.gpu>=0 else torch.device('cpu')
                images_a = data['img_A'].to(device).detach()
                images_b = data['img_B'].to(device).detach()
                images_c = data['img_C'].to(device).detach()

                if images_a.size(0) != opts.batch_size:
                    continue

                # update model
                if (it + 1) % opts.d_iter != 0 and not it == len(train_loader)-1:
                    model.update_D_content(images_a, images_b)
                    if opts.style_dis:
                        model.update_D_style(images_a, images_b, images_c)
                    if opts.local_style_dis:
                        model.update_D_local_style(data)
                    continue
                else:
                    model.update_D(data)
                    model.update_EG()

                # save to display file
                if not opts.no_display_img:
                    saver.write_display(total_it, model)

                print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
                total_it += 1
                if total_it >= max_it:
                    saver.write_img(-1, model)
                    saver.write_model(-1, model)
                    break

            # decay learning rate
            if opts.n_ep_decay > -1:
                model.update_lr()

            # save result image
            saver.write_img(ep, model)

            # Save network weights
            saver.write_model(ep, total_it, model)

    return

if __name__ == '__main__':
    main()
