import torch
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks

import random
import torch.nn.functional as F

def nce_loss(image_feature, text_feature, N=9, select_index=None):
# image_feature: B * h
# text_feature : B * h
    # return: B * (N+1)
    batch_size = image_feature.shape[0]
    if select_index is not None:
        image_feature = image_feature[select_index]
        text_feature = text_feature[select_index]
        if N > len(select_index) - 1:
            N = len(select_index) - 1
    
    score = torch.mm(image_feature, text_feature.t()) #* torch.mean(i_norm) * torch.mean(t_norm)
    score = score.view(-1)
    negative_idx = list(range(len(score)))
    positive_idx = []
    for i in range(len(image_feature)):
        remove_idx = i*len(image_feature) + i
        negative_idx.remove(remove_idx)
        positive_idx.append(remove_idx)
    
    loss = 0
    pos_score = score[positive_idx]
    if 1:
        # K=2, M=B-1
        neg_score_starting = [i*(batch_size-1) for i in range(batch_size)]
        for i in range(batch_size-1):
            curr_neg_idx = [negative_idx[curr_neg_start + i] for curr_neg_start in neg_score_starting]
            neg_score = score[curr_neg_idx]
            loss += torch.log(1 + torch.exp(neg_score - pos_score))
        loss = loss.sum() / (batch_size * (batch_size - 1))
    else:
        # K=B-1
        for i in range(batch_size):
            curr_neg_idx = negative_idx[i*(batch_size-1):(i+1)*(batch_size-1)]
            neg_score = score[curr_neg_idx]
            loss -= torch.log(torch.exp(pos_score[i]) / (torch.exp(neg_score).sum() + torch.exp(pos_score[i])))
        loss = loss.sum() / (batch_size * (batch_size - 1))

    # K=b-1, M=1
    ## -1 since real index is removed
    #neg_batch_size = image_feature.shape[0] - 1
    #for i in range(len(image_feature)):
    #    remove_idx = i*len(image_feature) + i
    #    #random_idx = random.sample(range(0, len(negative_idx)), N)
    #    random_idx = random.sample(range(neg_batch_size * i, neg_batch_size * (i+1)), N)
    #    tmp = []
    #    for idx in random_idx:
    #        tmp.append(negative_idx[idx])
    #    tmp += [remove_idx]
    #    prob = F.log_softmax(score[tmp], dim=0).unsqueeze(0)
    #    #prob = score[tmp].unsqueeze(0)
    #    if ret is None:
    #        ret = prob
    #    else:
    #        ret = torch.cat([ret, prob], dim=0)
    #
    #target = Variable(torch.zeros(prob.shape).cuda())
    #target[:,-1] = 1
    #loss = -torch.sum(prob * target) / prob.shape[0] 
    return loss

def hard_negative_loss(image_features, text_features, margin=1, hard=False, labels=None):
    scores = torch.mm(image_features, text_features.T) 
    diagonal = scores.diag().view(image_features.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    cost_s = (margin + scores - d1).clamp(min=0)
    cost_im = (margin + scores - d2).clamp(min=0)
    
    # clear diagonals
    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    # Hard Negative
    if hard:
        #cost_s = cost_s.max(1)[0]
        #cost_im = cost_im.max(0)[0]

        k = 10
        if k > cost_s.shape[0]:
            k = cost_s.shape[0]

        cost_s, _ = torch.topk(cost_s, k, dim=1)
        cost_im, _ = torch.topk(cost_im, k, dim=0)
        cost_im = cost_im.t()
        
    return (cost_s + cost_im).mean()


class JointModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['pair_A', 'pair_B', 'compose']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['image_A'] #, 'image_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_G = torch.optim.SGD(self.netG.parameters(),lr=opt.lr, momentum=0.9, weight_decay=1e-6)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.query_dataset = False
        self.zero = torch.tensor([0]).cuda()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.image_A = input['A']
        #self.text_A = input['T_A']
        self.text_A = input['T_S']  # -> used for hub
        if 'B' in input.keys():
            self.image_B = input['B']
            self.text_B = input['T_B']
            self.query = input['Q']
            self.query_dataset = True
            self.labels = None
        else:
            self.labels = input['L']

        #print(self.text_A)
        #print(self.text_B)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.query_dataset:
            self.image_A_feature, self.image_B_feature, self.text_A_feature, self.text_B_feature, self.composed_feature = \
                    self.netG(self.image_A, self.image_B, self.text_A, self.text_B, self.query)  # G(A)
        else:
            self.image_A_feature, self.text_A_feature = self.netG(self.image_A, self.text_A)
        
        #print(self.image_A[0][0][0][:10])

        #print(self.image_A_feature[0][:10])
        #print(self.image_B_feature[0][:10])
        #print(self.text_A_feature[0][:10])
        #print(self.text_B_feature[0][:10])
        #print('****')
        
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # pair loss
        #self.loss_pair_A = hard_negative_loss(self.image_A_feature, self.text_A_feature, margin=0.2, hard=True, labels=self.labels) 
        self.loss_pair_A = hard_negative_loss(self.image_A_feature, self.text_A_feature, margin=0.05, hard=True, labels=self.labels) 
        #self.loss_pair_A = nce_loss(self.image_A_feature, self.text_A_feature, self.image_A_feature.shape[0]-1)

        if self.query_dataset:
            self.loss_pair_B = hard_negative_loss(self.image_B_feature, self.text_B_feature, margin=0.2, hard=True) 
            # compositional loss
            self.loss_compose = hard_negative_loss(self.image_B_feature, self.composed_feature, margin=0.2, hard=False)
        else:
            self.loss_pair_B, self.loss_compose = self.zero, self.zero

        self.loss_G = self.loss_pair_A + self.loss_pair_B + 0.1 * self.loss_compose
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        #self.set_requires_grad(self.netD, True)  # enable backprop for D
        #self.optimizer_D.zero_grad()     # set D's gradients to zero
        #self.backward_D()                # calculate gradients for D
        #self.optimizer_D.step()          # update D's weights
        # update G
        #self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
