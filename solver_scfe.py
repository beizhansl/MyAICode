import torch
from torch import nn
from torch.autograd import Variable, grad
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import os
import time
import datetime
import net
from loss import InpaintingLoss
from tqdm import tqdm as tqdm

device = torch.device('cuda')

class Solver_SCFEGAN(object):
    def __init__(self, data_loaders, config):
        # dataloader
        self.checkpoint = config.checkpoint
        # Hyper-parameteres
        self.g_lr = config.G_LR
        self.d_lr = config.D_LR
        self.ndis = config.ndis
        self.num_epochs = config.num_epochs  # set 200
        self.num_epochs_decay = config.num_epochs_decay #set 100
        self.batch_size = config.batch_size

        # Training settings
        self.snapshot_step = config.snapshot_step
        self.log_step = config.log_step
        self.vis_step = config.vis_step

        #training setting
        self.task_name = config.task_name

        # Data loader
        self.data_loader_train = data_loaders
        # Path
        self.vis_path = config.vis_path +  config.task_name
        self.snapshot_path = config.snapshot_path +  config.task_name
        self.result_path = config.vis_path +  config.task_name
        self.LAMBDA_DICT = {
            'valid': 1.0, 'hole': 2.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)
        #create model
        self.build_model()
        # Start with trained model
        if self.checkpoint:
            print('Loaded pretrained model from: ', self.checkpoint)
            self.load_checkpoint()

        writer_path = os.path.join('./', 'log', config.task_name)
        print('TensorBoard will be saved in: ', writer_path)
        self.writer = SummaryWriter(writer_path)
        if not os.path.isdir(os.path.join('./', 'log', config.task_name)):
            os.makedirs(os.path.join('./log', config.task_name))
        #for recording
        self.start_time = time.time()
        self.e = 0
        self.i = 0

        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def log_terminal(self,g_loss,d_loss,pos,neg):
        elapsed = time.time() - self.start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
            elapsed, self.e+1, self.num_epochs, self.i+1, self.iters_per_epoch)
        log += ", {}: {:.4f}".format('g_loss', g_loss)
        log += ", {}: {:.4f}".format('d_loss', d_loss)
        log += ", {}: {:.4f}".format('pos_loss', pos)
        log += ", {}: {:.4f}".format('neg_loss', neg)
        print('\n')
        print(log)

    def save_models(self):
        torch.save(self.G.state_dict(),
                   os.path.join(self.snapshot_path, '{}_{}_G.pth'.format(self.e + 1, self.i + 1)))

        torch.save(self.D.state_dict(),
                   os.path.join(self.snapshot_path, '{}_{}_D.pth'.format(self.e + 1, self.i + 1)))

    # custom weights initialization called on netG and netD
    def weights_init(self,model):
        # classname = m.__class__.__name__
        # if classname.find('Conv') != -1:
        #     nn.init.normal_(m.weight.data, 0.0, 0.02)
        # elif classname.find('BatchNorm') != -1:
        #     nn.init.normal_(m.weight.data, 1.0, 0.02)
        #     nn.init.constant_(m.bias.data, 0)
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear,nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)


    def load_checkpoint(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.snapshot_path, '{}_G.pth'.format(self.checkpoint))))
        self.D.load_state_dict(torch.load(os.path.join(
                self.snapshot_path, '{}_D.pth'.format(self.checkpoint))))
        print('loaded trained models (step: {})..!'.format(self.checkpoint))

    def build_model(self):
        # Define generators and discriminators
        self.G = net.Generator_scfe()
        self.D = net.Discriminator()
        #打印模型
        print(self.G)
        print(self.D)
        self.criterion= InpaintingLoss(net.VGG16FeatureExtractor()).to(device)

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr)
        self.g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.g_optimizer,mode='min',factor=0.5,patience=2,verbose=True)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(),self.d_lr)
        self.d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.d_optimizer, mode='min', factor=0.5,patience=2, verbose=True)
        # Weights initialization
        self.weights_init(self.G)
        self.weights_init(self.D)


        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()


    def train(self):
        """Train StarGAN within a single dataset."""
        # The number of iterations per epoch
        self.iters_per_epoch = len(self.data_loader_train)
        # Start with trained model if exists
        g_lr = self.g_lr
        d_lr = self.d_lr
        if self.checkpoint:
            start = int(self.checkpoint.split('_')[0])
        else:
            start = 0
        # Start training
        self.start_time = time.time()
        #num_epochs是对所有数据进行的轮次
        for self.e in tqdm(range(start, self.num_epochs)):
            d_loss = 0.0
            g_loss = 0.0
            for self.i, (img, mask, sketch, color, noise) in enumerate(tqdm(self.data_loader_train)):
                #mask是反的，这里需要返过来
                mask = 1 - mask
                # print("mask shape:",mask.shape)
                # print("img shape:", img.shape)
                # print("sketch shape:", sketch.shape)
                # print("color shape:", color.shape)
                # print("noise shape:", noise.shape)
                d_loss = 0.0
                g_loss = 0.0
                # Convert tensor to variable
                # ================== Train D ================== #
                # img is groud truth
                img1 = img * mask
                sketch = sketch * mask
                color = color * mask
                noise = noise * mask
                #后续的无论是G还是D都不是直接对原来的图进行的，是对cat后的图进行的哈
                img1 = self.to_var(img1,requires_grad=False)
                img = self.to_var(img, requires_grad=False)
                mask = self.to_var(mask, requires_grad=False)
                noise = self.to_var(noise, requires_grad=False)
                sketch = self.to_var(sketch, requires_grad=False)
                color = self.to_var(color, requires_grad=False)

                input = torch.cat(                          #这里totensor已经完成了CHW的转换
                    [img1,sketch,color,mask,noise],dim=1)
                input = self.to_var(input,requires_grad=True)
                #得到output和comp
                output = self.G(input)
                #这里不能对G进行更新，需要停住，comp的requires_grad为True
                comp = output * (1-mask) + img * mask

                #计算Dloss
                pos_imgs = torch.cat([img,sketch,color,mask], dim=1)        #c = 8
                neg_imgs = torch.cat([comp,sketch,color, mask], dim=1)
                pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)
                pos_imgs = self.to_var(pos_imgs, requires_grad=True)
                neg_imgs = self.to_var(neg_imgs, requires_grad=True)
                pos_neg_imgs = self.to_var(pos_neg_imgs, requires_grad=True).detach()
                # pred_pos_neg是真与假的扩展
                pred_pos_neg = self.D(pos_neg_imgs)
                pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
                pos_loss = torch.mean(1.-pred_pos)      #对真实图片的打分,越小越好
                neg_loss = torch.mean(1.+pred_neg)      #对虚假图片的打分
                # IE [1 − D(Igt)] + IE [1 + D(Icomp)]
                d_loss += pos_loss + neg_loss
                #还有一项GPLoss
                gp_loss = self.gradient_penalty(img,comp,mask,sketch,color)
                d_loss += gp_loss * 10          #theta = 10
                # D loss齐全,排查1次
                self.d_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)
                self.d_optimizer.step()

                #g_loss为什么会是0？  问题排查2
                # ================== Train G ================== #
                if (self.i + 1) % self.ndis == 0:
                    # output = self.G(input)      #output是输出的图像
                    # comp = output * (1-mask) + img * mask # output直接使用上面产生的即可
                    loss_dict = self.criterion(mask,output,img)
                    #per-pixel/percept/style/tv
                    for key,weight in self.LAMBDA_DICT.items():
                        value = weight * loss_dict[key]
                        g_loss += value
                    #LG_SN          neg_imgs就是comp
                    pred_neg = self.D(neg_imgs)
                    #  - E [D (Icomp)]*B(B=0.001)  -->  LG_SN
                    gsn_loss = -torch.mean(pred_neg)#负的
                    g_loss += gsn_loss * 0.001
                    #E(D(Igt)2) ,pos_imgs为gt
                    pos_neg = self.D(pos_imgs)
                    pos_neg_2 = torch.pow(pos_neg,2)
                    #IE[D(Igt)2]
                    Igt_loss = torch.mean(pos_neg_2)#正的
                    g_loss += Igt_loss * 0.001

                    self.g_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                # save the images
                if (self.i + 1) % self.vis_step == 0:
                    print("Saving middle output...")
                    self.vis_train([img,output])

                # if (self.i + 1) % self.snapshot_step == 0:

                if (self.i % 1000 == 0):
                    self.writer.add_scalar('loss_g',g_loss,self.i*self.e + 1)
                    self.writer.add_scalar('loss_d', d_loss, self.i*self.e + 1)
                    self.writer.add_images('comp', comp, self.i*self.e + 1)
                    self.writer.add_images('output', output, self.i * self.e + 1)
                # Output training stats
                if (self.i + 1) % self.log_step == 0:
                    self.log_terminal(g_loss.item(),d_loss.item(),pos_loss.item(),neg_loss.item())

            #使用策略更新学习率
            self.d_scheduler.step(d_loss)
            self.g_scheduler.step(g_loss)

            # Save model checkpoints
            self.save_models()

            # Decay learning rate
            if (self.e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print('Decay learning rate to g_lr: {}, d_lr:{}.'.format(g_lr, d_lr))

        self.writer.close()
    def vis_train(self, img_train_list):
        # saving training results
        img_train_list = torch.cat(img_train_list, dim=3)
        result_path_train = os.path.join(self.result_path)
        if not os.path.exists(result_path_train):
            os.mkdir(result_path_train)
        save_path = os.path.join(result_path_train, '{}_{}.jpg'.format(self.e, self.i))
        save_image(img_train_list.data, save_path,normalize=True)

    #这里需要考虑mask
    #(img,comp,mask,sketch,color)
    def gradient_penalty(self, real_data, generated_data,mask,sketch,color):
        batch_size = real_data.size()[0]
        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if torch.cuda.is_available():
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = torch.cat([interpolated, sketch, color, mask], dim=1)
        interpolated = Variable(interpolated, requires_grad=True)

        if torch.cuda.is_available():
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda() if torch.cuda.is_available() else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # Calculate the new gradient after mask
        gradients = gradients * mask
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        #self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return  ((gradients_norm - 1) ** 2).mean()