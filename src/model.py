import networks
from networks import init_net
import torch
import torch.nn as nn
import numpy as np

class LADN(nn.Module):
    def __init__(self, opts):
        super(LADN, self).__init__()

        # parameters
        lr = 0.0001
        lr_dcontent = lr / 2.5
        self.batch_size = opts.batch_size
        self.device = torch.device('cuda:{}'.format(opts.gpu)) if opts.gpu>=0 else torch.device('cpu')
        self.backup_device = torch.device('cuda:{}'.format(opts.backup_gpu)) if opts.backup_gpu>=0 else torch.device('cpu')
        self.style_dis = opts.style_dis
        self.local_style_dis = opts.local_style_dis
        self.local_laplacian_loss = opts.local_laplacian_loss
        self.local_laplacian_loss_weight = opts.local_laplacian_loss_weight
        self.local_smooth_loss = opts.local_smooth_loss
        self.local_smooth_loss_weight = opts.local_smooth_loss_weight
        self.style_d_ls_weight = opts.style_d_ls_weight
        self.style_g_ls_weight = opts.style_g_ls_weight
        self.recon_weight = opts.recon_weight
        self.interpolate_num = opts.interpolate_num

        self.n_local = opts.n_local
        self.local_parts = ['eye', 'eye_', 'mouth', 'nose', 'cheek', 'cheek_', 'eyebrow', 'eyebrow_', 'uppernose', 'forehead', 'sidemouth', 'sidemouth_']
        self.local_parts_laplacian_weight = [4.0, 4.0, 2.0, 2.0, 4.0, 4.0, 3.0, 3.0, 2.0, 4.0, 2.0, 2.0]
        self.local_parts_smooth_weight = [0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0]

        counter = 0
        for i in range(self.n_local):
            local_part = self.local_parts[i]
            counter += 1
        self.valid_n_local = counter

        if self.style_dis:
            lr_dstyle = lr
        if self.local_style_dis:
            lr_dlocal_style = lr

        # discriminators
        self.disA = init_net(networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm), opts.backup_gpu, init_type='normal', gain=0.02)
        self.disB = init_net(networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm), opts.backup_gpu, init_type='normal', gain=0.02)
        self.disA2 = init_net(networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm), opts.backup_gpu, init_type='normal', gain=0.02)
        self.disB2 = init_net(networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm), opts.backup_gpu, init_type='normal', gain=0.02)

        self.disContent = init_net(networks.Dis_content(ndf=128), opts.backup_gpu, init_type='normal', gain=0.02)
        
        if self.style_dis:
            self.disStyle = init_net(networks.Dis_pair(opts.input_dim_a, opts.input_dim_b, opts.dis_n_layer), opts.backup_gpu, init_type='normal', gain=0.02)
        
        if self.local_style_dis:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                if '_' in local_part:
                    continue
                setattr(self, 'dis'+local_part.capitalize(), init_net(networks.Dis_pair(opts.input_dim_a, opts.input_dim_b, opts.dis_n_layer), opts.backup_gpu, init_type='normal', gain=0.02))

        # encoders
        self.enc_c = init_net(networks.E_content(opts.input_dim_a, opts.input_dim_b), opts.backup_gpu, init_type='normal', gain=0.02)
        self.enc_a = init_net(networks.E_attr(opts.input_dim_a, opts.input_dim_b), opts.backup_gpu, init_type='normal', gain=0.02)

        # generator
        self.gen = init_net(networks.G(opts.input_dim_a, opts.input_dim_b, num_residule_block=opts.num_residule_block), opts.gpu, init_type='normal', gain=0.02)

        # optimizers
        self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disA2_opt = torch.optim.Adam(self.disA2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disB2_opt = torch.optim.Adam(self.disB2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        
        if self.style_dis:
            self.disStyle_opt = torch.optim.Adam(self.disStyle.parameters(), lr=lr_dstyle, betas=(0.5, 0.999), weight_decay=0.0001)
        
        if self.local_style_dis:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                if '_' in local_part:
                    continue
                setattr(self, 'dis'+local_part.capitalize()+'_opt', torch.optim.Adam(getattr(self, 'dis'+local_part.capitalize()).parameters(), lr=lr_dlocal_style, betas=(0.5, 0.999), weight_decay=0.0001))
        # Setup the loss function for training
        self.criterionL1 = nn.L1Loss()


    def set_scheduler(self, opts, last_ep=-1):
        self.disA_sch = networks.get_scheduler(self.disA_opt, opts, last_ep)
        self.disB_sch = networks.get_scheduler(self.disB_opt, opts, last_ep)
        self.disA2_sch = networks.get_scheduler(self.disA2_opt, opts, last_ep)
        self.disB2_sch = networks.get_scheduler(self.disB2_opt, opts, last_ep)
        self.disContent_sch = networks.get_scheduler(self.disContent_opt, opts, last_ep)
        self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
        self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, opts, last_ep)
        self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)
        
        if self.style_dis:
            self.disStyle_sch = networks.get_scheduler(self.disStyle_opt, opts, last_ep)
        
        if self.local_style_dis:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                if '_' in local_part:
                    continue
                setattr(self, 'dis'+local_part.capitalize()+'_sch', networks.get_scheduler(getattr(self, 'dis'+local_part.capitalize()+'_opt'), opts, last_ep))


    def forward_content(self):
        half_size = self.batch_size//2
        self.real_A_encoded = self.input_A[0:half_size]
        self.real_B_encoded = self.input_B[0:half_size]
        # get encoded z_c
        self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)
        self.z_content_a = (self.z_content_a[0].to(self.device), self.z_content_a[1].to(self.device))
        self.z_content_b = (self.z_content_b[0].to(self.device), self.z_content_b[1].to(self.device))


    def backward_contentD(self):
        pred_fake = self.disContent.forward(self.z_content_a[1].detach().to(self.backup_device))
        pred_real = self.disContent.forward(self.z_content_b[1].detach().to(self.backup_device))

        loss_D = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all1 = torch.ones((out_real.size(0))).to(self.backup_device)
            all0 = torch.zeros((out_fake.size(0))).to(self.backup_device)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            loss_D += ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D


    def update_D_content(self, image_a, image_b):
        self.input_A = image_a
        self.input_B = image_b
        self.forward_content()
        self.disContent_opt.zero_grad()
        loss_D_Content = self.backward_contentD()
        self.disContent_loss = loss_D_Content.item()
        nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
        self.disContent_opt.step()


    def forward_style(self):
        half_size = self.batch_size//2
        self.real_A_encoded = self.input_A[0:half_size]
        self.real_B_encoded = self.input_B[0:half_size]
        self.real_C_encoded = self.input_C[0:half_size]

        # get encoded z_c
        self.z_content_a = self.enc_c.forward_a(self.real_A_encoded)
        self.z_content_a = (self.z_content_a[0].to(self.device), self.z_content_a[1].to(self.device))

        # get encoded z_a
        self.z_attr_b = self.enc_a.forward_b(self.real_B_encoded.to(self.backup_device))
        self.z_attr_b = self.z_attr_b.to(self.device)

        # first cross translation
        self.fake_B_encoded = self.gen.forward_b(*self.z_content_a, self.z_attr_b)


    def backward_styleD(self):
        pred_fake = self.disStyle.forward(self.real_B_encoded.to(self.backup_device), self.fake_B_encoded.detach().to(self.backup_device))
        pred_real = self.disStyle.forward(self.real_B_encoded.to(self.backup_device), self.real_C_encoded.to(self.backup_device))
        loss_D = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all1 = torch.ones((out_real.size(0))).to(self.backup_device)
            all0 = torch.zeros((out_fake.size(0))).to(self.backup_device)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            loss_D += (ad_true_loss + ad_fake_loss)
        loss_D = loss_D * self.style_d_ls_weight
        loss_D.backward()
        return loss_D


    def update_D_style(self, image_a, image_b, image_c):
        self.input_A = image_a
        self.input_B = image_b
        self.input_C = image_c
        self.forward_style()
        self.disStyle_opt.zero_grad()
        loss_D_Style = self.backward_styleD()
        self.disStyle_loss = loss_D_Style.item()
        nn.utils.clip_grad_norm_(self.disStyle.parameters(), 5)
        self.disStyle_opt.step()


    def forward_local_style(self):
        self.forward_style()
        half_size = self.batch_size//2
        self.rects_transfer_encoded = self.rects_A[0:half_size]
        self.rects_after_encoded = self.rects_B[0:half_size]
        self.rects_blend_encoded = self.rects_C[0:half_size]


    def backward_local_styleD(self, netD, rects_transfer, rects_after, rects_blend, name='', flip=False):
        N = self.real_B_encoded.size(0)
        C = self.real_B_encoded.size(1)
        H = rects_transfer[0][1]-rects_transfer[0][0]
        W = rects_transfer[0][3]-rects_transfer[0][2]

        transfer_crop = torch.empty((N,C,H,W)).to(self.backup_device)
        after_crop = torch.empty((N,C,H,W)).to(self.backup_device)
        blend_crop = torch.empty((N,C,H,W)).to(self.backup_device)

        for i in range(N):
            x1_t, x2_t, y1_t, y2_t = rects_transfer[i]
            x1_a, x2_a, y1_a, y2_a = rects_after[i]
            x1_b, x2_b, y1_b, y2_b = rects_blend[i]
            if not flip:
                transfer_crop[i] = self.fake_B_encoded[i,:,x1_t:x2_t,y1_t:y2_t].clone()
                after_crop[i] = self.real_B_encoded[i,:,x1_a:x2_a,y1_a:y2_a].clone()
                blend_crop[i] = self.real_C_encoded[i,:,x1_b:x2_b,y1_b:y2_b].clone()
            else:
                id = [i for i in range(W-1, -1, -1)]
                idx = torch.LongTensor(id).to(self.device)
                idx_backup = torch.LongTensor(id).to(self.backup_device)
                transfer_crop[i] = self.fake_B_encoded[i,:,x1_t:x2_t,y1_t:y2_t].index_select(2, idx).clone()
                after_crop[i] = self.real_B_encoded[i,:,x1_a:x2_a,y1_a:y2_a].index_select(2, idx_backup).clone()
                blend_crop[i] = self.real_C_encoded[i,:,x1_b:x2_b,y1_b:y2_b].index_select(2, idx_backup).clone()

        setattr(self, name+'_transfer', transfer_crop)
        setattr(self, name+'_after', after_crop)
        setattr(self, name+'_blend', blend_crop)

        pred_fake = netD.forward(after_crop.detach(), transfer_crop.detach())
        pred_real = netD.forward(after_crop.detach(), blend_crop.detach())
        loss_D = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all1 = torch.ones((out_real.size(0))).to(self.backup_device)
            all0 = torch.zeros((out_fake.size(0))).to(self.backup_device)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            loss_D += (ad_true_loss + ad_fake_loss)
        loss_D = loss_D * self.style_d_ls_weight / self.n_local
        loss_D.backward()
        return loss_D


    def update_D_local_style(self, data):
        self.input_A = data['img_A'].to(self.backup_device).detach()
        self.input_B = data['img_B'].to(self.backup_device).detach()
        self.input_C = data['img_C'].to(self.backup_device).detach()
        self.rects_A = data['rects_A'].to(self.backup_device).detach()
        self.rects_B = data['rects_B'].to(self.backup_device).detach()
        self.rects_C = data['rects_C'].to(self.backup_device).detach()

        self.forward_local_style()

        for i in range(self.n_local):
            local_part = self.local_parts[i]
            if '_' not in local_part:
                getattr(self, 'dis'+local_part.capitalize()+'_opt').zero_grad()
                loss_D_Style = self.backward_local_styleD(getattr(self, 'dis'+local_part.capitalize()), self.rects_transfer_encoded[:,i,:], self.rects_after_encoded[:,i,:], self.rects_blend_encoded[:,i,:], name=local_part)
                nn.utils.clip_grad_norm_(getattr(self, 'dis'+local_part.capitalize()).parameters(), 5)
                getattr(self, 'dis'+local_part.capitalize()+'_opt').step()
                setattr(self, 'dis'+local_part.capitalize()+'Style_loss', loss_D_Style.item())
            else:
                local_part = local_part.split('_')[0]
                getattr(self, 'dis'+local_part.capitalize()+'_opt').zero_grad()
                loss_D_Style_ = self.backward_local_styleD(getattr(self, 'dis'+local_part.capitalize()), self.rects_transfer_encoded[:,i,:], self.rects_after_encoded[:,i,:], self.rects_blend_encoded[:,i,:], name=local_part+'2', flip=True)
                nn.utils.clip_grad_norm_(getattr(self, 'dis'+local_part.capitalize()).parameters(), 5)
                getattr(self, 'dis'+local_part.capitalize()+'_opt').step()
                loss_D_Style = getattr(self, 'dis'+local_part.capitalize()+'Style_loss')
                setattr(self, 'dis'+local_part.capitalize()+'Style_loss', loss_D_Style+loss_D_Style_.item())


    def forward(self):
        # input images
        half_size = self.batch_size//2
        real_A = self.input_A
        real_B = self.input_B
        real_C = self.input_C
        self.real_A_encoded = real_A[0:half_size]
        self.real_A_random = real_A[half_size:]
        self.real_B_encoded = real_B[0:half_size]
        self.real_B_random = real_B[half_size:]
        self.real_C_encoded = real_C[0:half_size]

        if self.local_style_dis:
            self.rects_transfer_encoded = self.rects_A[0:half_size]
            self.rects_after_encoded = self.rects_B[0:half_size]
            self.rects_blend_encoded = self.rects_C[0:half_size]

        # get encoded z_c
        self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)
        self.z_content_a = (self.z_content_a[0].to(self.device), self.z_content_a[1].to(self.device))
        self.z_content_b = (self.z_content_b[0].to(self.device), self.z_content_b[1].to(self.device))

        # get encoded z_a
        self.z_attr_a, self.z_attr_b = self.enc_a.forward(self.real_A_encoded, self.real_B_encoded)
        self.z_attr_a = self.z_attr_a.to(self.device)
        self.z_attr_b = self.z_attr_b.to(self.device)

        # get random z_a
        self.z_random = torch.randn_like(self.z_attr_a).to(self.device)

        # first cross translation
        input_content_forA = (torch.cat((self.z_content_b[0], self.z_content_a[0], self.z_content_b[0]),0), \
                              torch.cat((self.z_content_b[1], self.z_content_a[1], self.z_content_b[1]),0))
        input_content_forB = (torch.cat((self.z_content_a[0], self.z_content_b[0], self.z_content_a[0]),0), \
                              torch.cat((self.z_content_a[1], self.z_content_b[1], self.z_content_a[1]),0))
        input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random),0)
        input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random),0)
        output_fakeA = self.gen.forward_a(*input_content_forA, input_attr_forA)
        output_fakeB = self.gen.forward_b(*input_content_forB, input_attr_forB)
        self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random = torch.split(output_fakeA, self.z_content_a[0].size(0), dim=0)
        self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fakeB, self.z_content_a[0].size(0), dim=0)

        # get reconstructed encoded z_c
        self.z_content_recon_b, self.z_content_recon_a = self.enc_c.forward(self.fake_A_encoded.to(self.backup_device), self.fake_B_encoded.to(self.backup_device))
        self.z_content_recon_b = (self.z_content_recon_b[0].to(self.device), self.z_content_recon_b[1].to(self.device))
        self.z_content_recon_a = (self.z_content_recon_a[0].to(self.device), self.z_content_recon_a[1].to(self.device))

        # get reconstructed encoded z_a
        self.z_attr_recon_a, self.z_attr_recon_b = self.enc_a.forward(self.fake_A_encoded.to(self.backup_device), self.fake_B_encoded.to(self.backup_device))
        self.z_attr_recon_a = self.z_attr_recon_a.to(self.device)
        self.z_attr_recon_b = self.z_attr_recon_b.to(self.device)

        # second cross translation
        self.fake_A_recon = self.gen.forward_a(*self.z_content_recon_a, self.z_attr_recon_a)
        self.fake_B_recon = self.gen.forward_b(*self.z_content_recon_b, self.z_attr_recon_b)

        # for display
        self.image_display = torch.cat((self.real_A_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(), \
                                        self.fake_B_random[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach().cpu(), self.fake_A_recon[0:1].detach().cpu(), \
                                        self.real_B_encoded[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(), \
                                        self.fake_A_random[0:1].detach().cpu(), self.fake_BB_encoded[0:1].detach().cpu(), self.fake_B_recon[0:1].detach().cpu()), dim=0)
        self.image_display = torch.cat((self.image_display, self.real_C_encoded[0:1].detach().cpu()), dim=0)

        # for latent regression
        self.z_attr_random_a, self.z_attr_random_b = self.enc_a.forward(self.fake_A_random.to(self.backup_device), self.fake_B_random.to(self.backup_device))
        self.z_attr_random_a = self.z_attr_random_a.to(self.device)
        self.z_attr_random_b = self.z_attr_random_b.to(self.device)


    def backward_D(self, netD, real, fake):
        pred_fake = netD.forward(fake.detach().to(self.backup_device))
        pred_real = netD.forward(real.to(self.backup_device))
        loss_D = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all0 = torch.zeros_like(out_fake).to(self.backup_device)
            all1 = torch.ones_like(out_real).to(self.backup_device)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            loss_D += ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D


    def update_D(self, data):
        images_a = data['img_A'].to(self.backup_device).detach()
        images_b = data['img_B'].to(self.backup_device).detach()
        images_c = data['img_C'].to(self.backup_device).detach()
        self.input_A = images_a
        self.input_B = images_b
        self.input_C = images_c

        if self.local_style_dis:
            self.rects_A = data['rects_A'].to(self.backup_device).detach()
            self.rects_B = data['rects_B'].to(self.backup_device).detach()
            self.rects_C = data['rects_C'].to(self.backup_device).detach()

        self.forward()

        # update disA
        self.disA_opt.zero_grad()
        loss_D1_A = self.backward_D(self.disA, self.real_A_encoded, self.fake_A_encoded)
        self.disA_loss = loss_D1_A.item()
        self.disA_opt.step()

        # update disA2
        self.disA2_opt.zero_grad()
        loss_D2_A = self.backward_D(self.disA2, self.real_A_random, self.fake_A_random)
        self.disA2_loss = loss_D2_A.item()
        self.disA2_opt.step()

        # update disB
        self.disB_opt.zero_grad()
        loss_D1_B = self.backward_D(self.disB, self.real_B_encoded, self.fake_B_encoded)
        self.disB_loss = loss_D1_B.item()
        self.disB_opt.step()

        # update disB2
        self.disB2_opt.zero_grad()
        loss_D2_B = self.backward_D(self.disB2, self.real_B_random, self.fake_B_random)
        self.disB2_loss = loss_D2_B.item()
        self.disB2_opt.step()

        # update disContent
        self.disContent_opt.zero_grad()
        loss_D_Content = self.backward_contentD()
        self.disContent_loss = loss_D_Content.item()
        nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
        self.disContent_opt.step()

        # update disStyle
        if self.style_dis:
            self.disStyle_opt.zero_grad()
            loss_D_Style = self.backward_styleD()
            self.disStyle_loss = loss_D_Style.item()
            nn.utils.clip_grad_norm_(self.disStyle.parameters(), 5)
            self.disStyle_opt.step()

        # update disLocalStyle
        if self.local_style_dis:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                if '_' not in local_part:
                    getattr(self, 'dis'+local_part.capitalize()+'_opt').zero_grad()
                    loss_D_Style = self.backward_local_styleD(getattr(self, 'dis'+local_part.capitalize()), self.rects_transfer_encoded[:,i,:], self.rects_after_encoded[:,i,:], self.rects_blend_encoded[:,i,:], name=local_part)
                    nn.utils.clip_grad_norm_(getattr(self, 'dis'+local_part.capitalize()).parameters(), 5)
                    getattr(self, 'dis'+local_part.capitalize()+'_opt').step()
                    setattr(self, 'dis'+local_part.capitalize()+'Style_loss', loss_D_Style.item())
                else:
                    local_part = local_part.split('_')[0]
                    getattr(self, 'dis'+local_part.capitalize()+'_opt').zero_grad()
                    loss_D_Style_ = self.backward_local_styleD(getattr(self, 'dis'+local_part.capitalize()), self.rects_transfer_encoded[:,i,:], self.rects_after_encoded[:,i,:], self.rects_blend_encoded[:,i,:], name=local_part+'2', flip=True)
                    nn.utils.clip_grad_norm_(getattr(self, 'dis'+local_part.capitalize()).parameters(), 5)
                    getattr(self, 'dis'+local_part.capitalize()+'_opt').step()
                    loss_D_Style = getattr(self, 'dis'+local_part.capitalize()+'Style_loss')
                    setattr(self, 'dis'+local_part.capitalize()+'Style_loss', loss_D_Style+loss_D_Style_.item())


    def backward_G_GAN_content(self, z_content):
        outs = self.disContent.forward(z_content[1].to(self.backup_device))
        loss_G = 0
        for out in outs:
            outputs_fake = torch.sigmoid(out)
            all_half = 0.5*torch.ones((outputs_fake.size(0))).to(self.backup_device)
            loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_half)
        return loss_G


    def backward_G_GAN(self, fake, netD):
        outs_fake = netD.forward(fake.to(self.backup_device))
        loss_G = 0
        for out_a in outs_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).to(self.backup_device)
            loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
        return loss_G


    def backward_G_GAN_style(self):
        outs = self.disStyle.forward(self.real_B_encoded.to(self.backup_device), self.fake_B_encoded.to(self.backup_device))
        loss_G = 0
        for out in outs:
            outputs_fake = torch.sigmoid(out)
            all_ones = torch.ones((outputs_fake.size(0))).to(self.backup_device)
            loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
        loss_G = loss_G * self.style_g_ls_weight
        return loss_G


    def backward_G_GAN_local_style(self, netD, rects_transfer, rects_after, flip=False):
        N = self.real_B_encoded.size(0)
        C = self.real_B_encoded.size(1)
        H = rects_transfer[0][1]-rects_transfer[0][0]
        W = rects_transfer[0][3]-rects_transfer[0][2]

        transfer_crop = torch.empty((N,C,H,W)).to(self.backup_device)
        after_crop = torch.empty((N,C,H,W)).to(self.backup_device)

        for i in range(N):
            x1_t, x2_t, y1_t, y2_t = rects_transfer[i]
            x1_a, x2_a, y1_a, y2_a = rects_after[i]
            if not flip:
                transfer_crop[i] = self.fake_B_encoded[i,:,x1_t:x2_t,y1_t:y2_t].clone()
                after_crop[i] = self.real_B_encoded[i,:,x1_a:x2_a,y1_a:y2_a].clone()
            else:
                id = [i for i in range(W-1, -1, -1)]
                idx = torch.LongTensor(id).to(self.device)
                idx_backup = torch.LongTensor(id).to(self.backup_device)
                transfer_crop[i] = self.fake_B_encoded[i,:,x1_t:x2_t,y1_t:y2_t].index_select(2, idx).clone()
                after_crop[i] = self.real_B_encoded[i,:,x1_a:x2_a,y1_a:y2_a].index_select(2, idx_backup).clone()

        outs_fake = netD.forward(after_crop.detach(), transfer_crop)
        loss_G = 0
        for out_a in outs_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).to(self.backup_device)
            loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
        loss_G = loss_G * self.style_g_ls_weight / self.n_local
        return loss_G


    def backward_EG(self):
        # content Ladv for generator
        loss_G_GAN_Acontent = self.backward_G_GAN_content(self.z_content_a).to(self.device)
        loss_G_GAN_Bcontent = self.backward_G_GAN_content(self.z_content_b).to(self.device)

        # Ladv for generator
        loss_G_GAN_A = self.backward_G_GAN(self.fake_A_encoded, self.disA).to(self.device)
        loss_G_GAN_B = self.backward_G_GAN(self.fake_B_encoded, self.disB).to(self.device)

        if self.style_dis:
            # style Ladv for generator
            loss_G_GAN_style = self.backward_G_GAN_style().to(self.device)
    
        if self.local_style_dis:
            # local style Ladv for generator
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                if '_' not in local_part:
                    loss_G_GAN_local_style = self.backward_G_GAN_local_style(getattr(self, 'dis'+local_part.capitalize()), self.rects_transfer_encoded[:,i,:], self.rects_after_encoded[:,i,:])
                    setattr(self, 'G_GAN_'+local_part+'_style', loss_G_GAN_local_style.to(self.device))
                else:
                    local_part = local_part.split('_')[0]
                    loss_G_GAN_local_style = self.backward_G_GAN_local_style(getattr(self, 'dis'+local_part.capitalize()), self.rects_transfer_encoded[:,i,:], self.rects_after_encoded[:,i,:], flip=True)
                    setattr(self, 'G_GAN_'+local_part+'2_style', loss_G_GAN_local_style.to(self.device))
        if self.local_laplacian_loss:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                rects_transfer = self.rects_transfer_encoded[:,i,:]
                rects_blend = self.rects_blend_encoded[:,i,:]
                rects_after = self.rects_after_encoded[:,i,:]

                N = self.real_B_encoded.size(0)
                C = self.real_B_encoded.size(1)
                H = rects_transfer[0][1]-rects_transfer[0][0]
                W = rects_transfer[0][3]-rects_transfer[0][2]
                kernel = torch.tensor(np.broadcast_to(np.array([[0,1,0],[1,-4,1],[0,1,0]]),(N,C,3,3)), dtype=torch.float32).to(self.backup_device)

                transfer_crop = torch.empty((N,C,H,W)).to(self.backup_device)
                blend_crop = torch.empty((N,C,H,W)).to(self.backup_device)
                after_recon_crop = torch.empty((N,C,H,W)).to(self.backup_device)
                after_crop = torch.empty((N,C,H,W)).to(self.backup_device)

                for n in range(N):
                    x1_t, x2_t, y1_t, y2_t = rects_transfer[n]
                    x1_b, x2_b, y1_b, y2_b = rects_blend[n]
                    x1_a, x2_a, y1_a, y2_a = rects_after[n]
                    transfer_crop[n] = self.fake_B_encoded[n,:,x1_t:x2_t,y1_t:y2_t].clone()
                    blend_crop[n] = self.real_C_encoded[n,:,x1_b:x2_b,y1_b:y2_b].clone()
                    after_recon_crop[n] = self.fake_B_recon[n,:,x1_a:x2_a,y1_a:y2_a].clone()
                    after_crop[n] = self.real_B_encoded[n,:,x1_a:x2_a,y1_a:y2_a].clone()

                transfer_crop_filter = nn.functional.conv2d(transfer_crop, kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
                blend_crop_filter = nn.functional.conv2d(blend_crop, kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
                after_recon_crop_filter = nn.functional.conv2d(after_recon_crop, kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
                after_crop_filter = nn.functional.conv2d(after_crop, kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)

                local_transfer_loss = self.criterionL1(transfer_crop_filter, blend_crop_filter) * 10
                local_recon_loss = self.criterionL1(after_recon_crop_filter, after_crop_filter) * 10
                local_laplacian_loss = (local_transfer_loss + local_recon_loss) * self.local_laplacian_loss_weight * self.local_parts_laplacian_weight[i] / self.valid_n_local
                setattr(self, 'G_GAN_'+local_part+'_local_laplacian', local_laplacian_loss.to(self.device))

        if self.local_smooth_loss:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                rects_after = self.rects_after_encoded[:,i,:]

                N = self.fake_A_encoded.size(0)
                C = self.fake_A_encoded.size(1)
                H = rects_after[0][1]-rects_after[0][0]
                W = rects_after[0][3]-rects_after[0][2]
                kernel = torch.tensor(np.broadcast_to(np.array([[0,1,0],[1,-4,1],[0,1,0]]),(N,C,3,3)), dtype=torch.float32).to(self.backup_device)

                demakeup_crop = torch.empty((N,C,H,W)).to(self.backup_device)

                for n in range(N):
                    x1_a, x2_a, y1_a, y2_a = rects_after[n]
                    demakeup_crop[n] = self.fake_A_encoded[n,:,x1_a:x2_a,y1_a:y2_a].clone()

                demakeup_crop_filter = nn.functional.conv2d(demakeup_crop, kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
                demakeup_gt = torch.zeros_like(demakeup_crop_filter).to(self.backup_device)
                local_smooth_loss = self.criterionL1(demakeup_crop_filter, demakeup_gt) * 10 * self.local_smooth_loss_weight * self.local_parts_smooth_weight[i] / self.valid_n_local
                setattr(self, 'G_GAN_'+local_part+'_local_smooth', local_smooth_loss.to(self.device))

        # KL loss - z_a
        loss_kl_za_a = self._l2_regularize(self.z_attr_a) * 0.01
        loss_kl_za_b = self._l2_regularize(self.z_attr_b) * 0.01

        # KL loss - z_c
        loss_kl_zc_a = self._l2_regularize(self.z_content_a[1]) * 0.01
        loss_kl_zc_b = self._l2_regularize(self.z_content_b[1]) * 0.01

        # cross cycle consistency loss
        loss_G_L1_A = self.criterionL1(self.fake_A_recon, self.real_A_encoded.to(self.device)) * 10 * self.recon_weight
        loss_G_L1_B = self.criterionL1(self.fake_B_recon, self.real_B_encoded.to(self.device)) * 10 * self.recon_weight
        loss_G_L1_AA = self.criterionL1(self.fake_AA_encoded, self.real_A_encoded.to(self.device)) * 10
        loss_G_L1_BB = self.criterionL1(self.fake_BB_encoded, self.real_B_encoded.to(self.device)) * 10

        loss_G = loss_G_GAN_A + loss_G_GAN_B + \
                 loss_G_GAN_Acontent + loss_G_GAN_Bcontent + \
                 loss_G_L1_AA + loss_G_L1_BB + \
                 loss_G_L1_A + loss_G_L1_B + \
                 loss_kl_zc_a + loss_kl_zc_b + \
                 loss_kl_za_a + loss_kl_za_b
        if self.style_dis:
            loss_G = loss_G + loss_G_GAN_style

        if self.local_style_dis:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                if '_' not in local_part:
                    loss_G = loss_G + getattr(self, 'G_GAN_'+local_part+'_style')
                else:
                    local_part = local_part.split('_')[0]
                    loss_G = loss_G + getattr(self, 'G_GAN_'+local_part+'2_style')

        if self.local_laplacian_loss:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                loss_G = loss_G + getattr(self, 'G_GAN_'+local_part+'_local_laplacian')

        if self.local_smooth_loss:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                loss_G = loss_G + getattr(self, 'G_GAN_'+local_part+'_local_smooth')

        loss_G.backward(retain_graph=True)

        self.gan_loss_a = loss_G_GAN_A.item()
        self.gan_loss_b = loss_G_GAN_B.item()
        self.gan_loss_acontent = loss_G_GAN_Acontent.item()
        self.gan_loss_bcontent = loss_G_GAN_Bcontent.item()
        self.kl_loss_za_a = loss_kl_za_a.item()
        self.kl_loss_za_b = loss_kl_za_b.item()
        self.kl_loss_zc_a = loss_kl_zc_a.item()
        self.kl_loss_zc_b = loss_kl_zc_b.item()
        self.l1_recon_A_loss = loss_G_L1_A.item()
        self.l1_recon_B_loss = loss_G_L1_B.item()
        self.l1_recon_AA_loss = loss_G_L1_AA.item()
        self.l1_recon_BB_loss = loss_G_L1_BB.item()
        self.G_loss = loss_G.item()
        if self.style_dis:
            self.gan_loss_style = loss_G_GAN_style.item()

        if self.local_style_dis:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                if '_' not in local_part:
                    setattr(self, 'gan_loss_'+local_part+'_style', getattr(self, 'G_GAN_'+local_part+'_style').item())
                else:
                    local_part = local_part.split('_')[0]
                    gan_loss_style = getattr(self, 'gan_loss_'+local_part+'_style')
                    gan_loss_style_ = getattr(self, 'G_GAN_'+local_part+'2_style').item()
                    setattr(self, 'gan_loss_'+local_part+'_style', gan_loss_style+gan_loss_style_)

        if self.local_laplacian_loss:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                setattr(self, 'gan_loss_'+local_part+'_local_laplacian', getattr(self, 'G_GAN_'+local_part+'_local_laplacian').item())

        if self.local_smooth_loss:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                setattr(self, 'gan_loss_'+local_part+'_local_smooth', getattr(self, 'G_GAN_'+local_part+'_local_smooth').item())


    def backward_G_alone(self):
        # Ladv for generator
        loss_G_GAN2_A = self.backward_G_GAN(self.fake_A_random, self.disA2).to(self.device)
        loss_G_GAN2_B = self.backward_G_GAN(self.fake_B_random, self.disB2).to(self.device)

        # latent regression loss
        loss_z_L1_a = torch.mean(torch.abs(self.z_attr_random_a - self.z_random)) * 10
        loss_z_L1_b = torch.mean(torch.abs(self.z_attr_random_b - self.z_random)) * 10

        loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_G_GAN2_A + loss_G_GAN2_B
        loss_z_L1.backward()
        self.l1_recon_z_loss_a = loss_z_L1_a.item()
        self.l1_recon_z_loss_b = loss_z_L1_b.item()
        self.gan2_loss_a = loss_G_GAN2_A.item()
        self.gan2_loss_b = loss_G_GAN2_B.item()


    def update_EG(self):
        # update G, Ec, Ea
        self.enc_c_opt.zero_grad()
        self.enc_a_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.backward_EG()
        self.enc_c_opt.step()
        self.enc_a_opt.step()
        self.gen_opt.step()

        # update G, Ec
        self.enc_c_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.backward_G_alone()
        self.enc_c_opt.step()
        self.gen_opt.step()



    # forward method for interpolation purpose
    def interpolate_forward(self, images_a, images_b1, images_b2):
        with torch.no_grad():
            # input images
            half_size = self.batch_size//2
            self.input_A = images_a
            self.input_B1 = images_b1
            self.input_B2 = images_b2
            real_A = self.input_A
            real_B1 = self.input_B1
            real_B2 = self.input_B2
            self.real_A_encoded = real_A[0:half_size]
            self.real_B1_encoded = real_B1[0:half_size]
            self.real_B2_encoded = real_B2[0:half_size]

            # get encoded z_c
            self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B1_encoded)
            self.z_content_a = (self.z_content_a[0].to(self.device), self.z_content_a[1].to(self.device))
            self.z_content_b = (self.z_content_b[0].to(self.device), self.z_content_b[1].to(self.device))

            # get encoded z_a
            self.z_attr_a, self.z_attr_b1 = self.enc_a.forward(self.real_A_encoded, self.real_B1_encoded)
            self.z_attr_a, self.z_attr_b2 = self.enc_a.forward(self.real_A_encoded, self.real_B2_encoded)
            self.z_attr_a = self.z_attr_a.to(self.device)
            self.z_attr_b1 = self.z_attr_b1.to(self.device)
            self.z_attr_b2 = self.z_attr_b2.to(self.device)

            # get random z_a
            self.z_random = torch.randn_like(self.z_attr_a).to(self.device)

            input_content_forB = (torch.cat((self.z_content_a[0], self.z_content_b[0], self.z_content_a[0]),0), \
                                  torch.cat((self.z_content_a[1], self.z_content_b[1], self.z_content_a[1]),0))
            delta_z_attr = (self.z_attr_b2 - self.z_attr_b1)/float(self.interpolate_num)

            for i in range(self.interpolate_num+1):
                z_attr = self.z_attr_b1 + float(i)*delta_z_attr
                input_attr_forB = torch.cat((z_attr, z_attr, self.z_random),0)
                output_fakeB = self.gen.forward_b(*input_content_forB, input_attr_forB)
                self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fakeB, self.z_content_a[0].size(0), dim=0)
                setattr(self, 'transfer'+str(i), self.fake_B_encoded)


    # forward method for testing purpose
    def test_forward(self, images_a, images_b, images_c):
        with torch.no_grad():
            # input images
            half_size = self.batch_size//2
            self.input_A = images_a
            self.input_B = images_b
            self.input_C = images_c
            real_A = self.input_A
            real_B = self.input_B
            real_C = self.input_C
            self.real_A_encoded = real_A[0:half_size]
            self.real_A_random = real_A[half_size:]
            self.real_B_encoded = real_B[0:half_size]
            self.real_B_random = real_B[half_size:]
            self.real_C_encoded = real_C[0:half_size]

            # get encoded z_c
            self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)
            self.z_content_a = (self.z_content_a[0].to(self.device), self.z_content_a[1].to(self.device))
            self.z_content_b = (self.z_content_b[0].to(self.device), self.z_content_b[1].to(self.device))

            # get encoded z_a
            self.z_attr_a, self.z_attr_b = self.enc_a.forward(self.real_A_encoded, self.real_B_encoded)
            self.z_attr_a = self.z_attr_a.to(self.device)
            self.z_attr_b = self.z_attr_b.to(self.device)

            # get random z_a
            self.z_random = torch.randn_like(self.z_attr_a).to(self.device)

            # first cross translation
            input_content_forA = (torch.cat((self.z_content_b[0], self.z_content_a[0], self.z_content_b[0]),0), \
                                  torch.cat((self.z_content_b[1], self.z_content_a[1], self.z_content_b[1]),0))
            input_content_forB = (torch.cat((self.z_content_a[0], self.z_content_b[0], self.z_content_a[0]),0), \
                                  torch.cat((self.z_content_a[1], self.z_content_b[1], self.z_content_a[1]),0))
            input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random),0)
            input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random),0)
            output_fakeA = self.gen.forward_a(*input_content_forA, input_attr_forA)
            output_fakeB = self.gen.forward_b(*input_content_forB, input_attr_forB)
            self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random = torch.split(output_fakeA, self.z_content_a[0].size(0), dim=0)
            self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fakeB, self.z_content_a[0].size(0), dim=0)

            # get reconstructed encoded z_c
            self.z_content_recon_b, self.z_content_recon_a = self.enc_c.forward(self.fake_A_encoded.to(self.backup_device), self.fake_B_encoded.to(self.backup_device))
            self.z_content_recon_b = (self.z_content_recon_b[0].to(self.device), self.z_content_recon_b[1].to(self.device))
            self.z_content_recon_a = (self.z_content_recon_a[0].to(self.device), self.z_content_recon_a[1].to(self.device))

            # get reconstructed encoded z_a
            self.z_attr_recon_a, self.z_attr_recon_b = self.enc_a.forward(self.fake_A_encoded.to(self.backup_device), self.fake_B_encoded.to(self.backup_device))
            self.z_attr_recon_a = self.z_attr_recon_a.to(self.device)
            self.z_attr_recon_b = self.z_attr_recon_b.to(self.device)

            # second cross translation
            self.fake_A_recon = self.gen.forward_a(*self.z_content_recon_a, self.z_attr_recon_a)
            self.fake_B_recon = self.gen.forward_b(*self.z_content_recon_b, self.z_attr_recon_b)


    def update_lr(self):
        self.disA_sch.step()
        self.disB_sch.step()
        self.disA2_sch.step()
        self.disB2_sch.step()
        self.disContent_sch.step()
        self.enc_c_sch.step()
        self.enc_a_sch.step()
        self.gen_sch.step()

        if self.style_dis:
            self.disStyle_sch.step()

        if self.local_style_dis:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                if '_' in local_part:
                    continue
                getattr(self, 'dis'+local_part.capitalize()+'_sch').step()


    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss


    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir, map_location=self.device)
        checkpoint_backup = torch.load(model_dir, map_location=self.backup_device)

        # weight
        if train:
            self.disA.load_state_dict(checkpoint_backup['disA'])
            self.disA2.load_state_dict(checkpoint_backup['disA2'])
            self.disB.load_state_dict(checkpoint_backup['disB'])
            self.disB2.load_state_dict(checkpoint_backup['disB2'])
            self.disContent.load_state_dict(checkpoint_backup['disContent'])

            if self.style_dis:
                self.disStyle.load_state_dict(checkpoint_backup['disStyle'])

            if self.local_style_dis:
                for i in range(self.n_local):
                    local_part = self.local_parts[i]
                    if '_' in local_part:
                        continue
                    getattr(self, 'dis'+local_part.capitalize()).load_state_dict(checkpoint_backup['dis'+local_part.capitalize()])

        self.enc_c.load_state_dict(checkpoint_backup['enc_c'])
        self.enc_a.load_state_dict(checkpoint_backup['enc_a'])
        self.gen.load_state_dict(checkpoint['gen'])
        
        # optimizer
        if train:
            self.disA_opt.load_state_dict(checkpoint_backup['disA_opt'])
            self.disA2_opt.load_state_dict(checkpoint_backup['disA2_opt'])
            self.disB_opt.load_state_dict(checkpoint_backup['disB_opt'])
            self.disB2_opt.load_state_dict(checkpoint_backup['disB2_opt'])
            self.disContent_opt.load_state_dict(checkpoint_backup['disContent_opt'])
            self.enc_c_opt.load_state_dict(checkpoint_backup['enc_c_opt'])
            self.enc_a_opt.load_state_dict(checkpoint_backup['enc_a_opt'])
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
            
            if self.style_dis:
                self.disStyle_opt.load_state_dict(checkpoint_backup['disStyle_opt'])
            
            if self.local_style_dis:
                for i in range(self.n_local):
                    local_part = self.local_parts[i]
                    if '_' in local_part:
                        continue
                    getattr(self, 'dis'+local_part.capitalize()+'_opt').load_state_dict(checkpoint_backup['dis'+local_part.capitalize()+'_opt'])
        checkpoint_ep = checkpoint['ep']
        checkpoint_total_it = checkpoint['total_it']
        del checkpoint
        del checkpoint_backup
        torch.cuda.empty_cache()
        return checkpoint_ep, checkpoint_total_it


    def save(self, filename, ep, total_it):
        state = {
                 'disA': self.disA.state_dict(),
                 'disA2': self.disA2.state_dict(),
                 'disB': self.disB.state_dict(),
                 'disB2': self.disB2.state_dict(),
                 'disContent': self.disContent.state_dict(),
                 'enc_c': self.enc_c.state_dict(),
                 'enc_a': self.enc_a.state_dict(),
                 'gen': self.gen.state_dict(),
                 'disA_opt': self.disA_opt.state_dict(),
                 'disA2_opt': self.disA2_opt.state_dict(),
                 'disB_opt': self.disB_opt.state_dict(),
                 'disB2_opt': self.disB2_opt.state_dict(),
                 'disContent_opt': self.disContent_opt.state_dict(),
                 'enc_c_opt': self.enc_c_opt.state_dict(),
                 'enc_a_opt': self.enc_a_opt.state_dict(),
                 'gen_opt': self.gen_opt.state_dict(),
                 'ep': ep,
                 'total_it': total_it
                }
        
        if self.style_dis:
            state['disStyle'] = self.disStyle.state_dict()
            state['disStyle_opt'] = self.disStyle_opt.state_dict()
        
        if self.local_style_dis:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                if '_' in local_part:
                    continue
                state['dis'+local_part.capitalize()] = getattr(self, 'dis'+local_part.capitalize()).state_dict()
                state['dis'+local_part.capitalize()+'_opt'] = getattr(self, 'dis'+local_part.capitalize()+'_opt').state_dict()
        torch.save(state, filename)
        return


    def assemble_outputs(self):
        images_a = self.normalize_image(self.real_A_encoded).detach()
        images_b = self.normalize_image(self.real_B_encoded).detach()
        images_a1 = self.normalize_image(self.fake_A_encoded).detach()
        images_a2 = self.normalize_image(self.fake_A_random).detach()
        images_a3 = self.normalize_image(self.fake_A_recon).detach()
        images_a4 = self.normalize_image(self.fake_AA_encoded).detach()
        images_b1 = self.normalize_image(self.fake_B_encoded).detach()
        images_b2 = self.normalize_image(self.fake_B_random).detach()
        images_b3 = self.normalize_image(self.fake_B_recon).detach()
        images_b4 = self.normalize_image(self.fake_BB_encoded).detach()
        images_c = self.normalize_image(self.real_C_encoded).detach()

        row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_a4[0:1, ::], images_a3[0:1, ::]),3)
        row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_b4[0:1, ::], images_b3[0:1, ::]),3)
        row3 = torch.cat((images_c[0:1, ::], torch.zeros_like(images_c[0:1, ::]), torch.zeros_like(images_c[0:1, ::]), torch.zeros_like(images_c[0:1, ::]), torch.zeros_like(images_c[0:1, ::])),3)
        return torch.cat((row1,row2,row3),2)


    def normalize_image(self, x):
        return x[:,0:3,:,:]
