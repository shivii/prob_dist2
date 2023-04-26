import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from models.utility import print_with_time as print
import torch.nn as nn
from models.get_kl_divergence import get_divergence
import models.get_kl_1channel as div
from math import log
##############TSNE changes






class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training real_labellosses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        ############TSNE changes
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'D_A_ad', 'D_B_ad', 'G_A_ad', 'G_B_ad']
        
        
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        """New coefficient term for reducing large loss terms"""
        #self.alpha_gan = 0.01
        self.alpha_js = 1/256

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # adversarial loss through kl divergence on local neighbourhood
            self.get_adv = div.JSD().to(self.device) 

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            

    def set_input(self, input): 
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def neutralise_zeros(self, tensor, dim):
        epsilon = 1e-20
        tensor = tensor + epsilon
        tensor = tensor * 1/(1 + dim*dim*epsilon)
        return tensor

    def backward_D_basic(self, netD, real, fake, opt):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

            Loss_D : log(D(X)) + log(1-D(G(Z))),
            where, X: real data, Z is noise, G(Z): generated samples

            Loss_D = log(D(X)) + 2.JS(p_r || p_g) - log(4)  

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        pred_real_sft = pred_real
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())

        # Combined loss and calculate gradients
        if opt.advloss == 0:
            print("In gaussian adv loss-----------------Dis")
            div_D = self.get_adv.adv_loss(pred_real, pred_fake) 
            D_ad = 1/self.neutralise_zeros(div_D * opt.disc_coeff, 30)
            loss_D = loss_D_real + D_ad
        else:
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) 
        
        #loss_D = (loss_D_real + loss_D_fake)
        #print("loss D is " , loss_D)
        loss_D.backward()
        return loss_D, D_ad

    def backward_D_A(self, opt):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A, self.loss_D_A_ad = self.backward_D_basic(self.netD_A, self.real_B, fake_B, opt)

    def backward_D_B(self, opt):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B, self.loss_D_B_ad = self.backward_D_basic(self.netD_B, self.real_A, fake_A, opt) 
        
    def initialise_pdf_losses(self, opt):
        pdf_list = opt.which_pdf.split(",")
        if "1" in pdf_list:
            self.loss_gauss_A = 0
            self.loss_gauss_B = 0
            self.loss_names.append("gauss_A")
            self.loss_names.append("gauss_B")
        if "2" in pdf_list:
            self.loss_hist_A = 0
            self.loss_hist_B = 0
            self.loss_names.append("hist_A")
            self.loss_names.append("hist_B")
        if "3" in pdf_list:
            self.loss_wt_hist_A = 0
            self.loss_wt_hist_B = 0
            self.loss_names.append("wt_hist_A")
            self.loss_names.append("wt_hist_B")
        if "4" in pdf_list:
            self.loss_img_pdf_A = 0
            self.loss_img_pdf_B = 0
            self.loss_names.append("img_pdf_A")
            self.loss_names.append("img_pdf_B")
        if "5" in pdf_list:
            self.loss_hist_pat_A = 0
            self.loss_hist_pat_B = 0
            self.loss_names.append("hist_pat_A")
            self.loss_names.append("hist_pat_B")
        if "6" in pdf_list:
            self.loss_log_A = 0
            self.loss_log_B = 0
            self.loss_names.append("log_A")
            self.loss_names.append("log_B")

    def get_log_loss(self, real_A, recreated_A, real_B, recreated_B, coeff):
        loss = nn.BCEWithLogitsLoss()
        self.loss_log_A = loss(real_A, recreated_A) * coeff
        self.loss_log_B = loss(real_B, recreated_B) * coeff
        sum = self.loss_log_A + self.loss_log_B
        return sum

    def compute_pdf_losses(self, opt):
        pdf_list = opt.which_pdf.split(",")
        sum = 0
        if "1" in pdf_list:
            coeff = 7
            self.loss_gauss_A = get_divergence(self.real_A, self.rec_A, pdf=1, klloss=opt.klloss) * coeff
            self.loss_gauss_B = get_divergence(self.real_B, self.rec_B, pdf=1, klloss=opt.klloss) * coeff
            sum = sum + self.loss_gauss_A + self.loss_gauss_B
        if "2" in pdf_list:
            coeff = 1
            self.loss_hist_A = get_divergence(self.real_A, self.rec_A, pdf=2, klloss=opt.klloss) * coeff
            self.loss_hist_B = get_divergence(self.real_B, self.rec_B, pdf=2, klloss=opt.klloss) * coeff
            sum = sum + self.loss_hist_A + self.loss_hist_B
        if "3" in pdf_list:
            coeff = 1
            self.loss_wt_hist_A = get_divergence(self.real_A, self.rec_A, pdf=3, klloss=opt.klloss) * coeff
            self.loss_wt_hist_B = get_divergence(self.real_B, self.rec_B, pdf=3, klloss=opt.klloss) * coeff
            sum = sum + self.loss_wt_hist_A + self.loss_wt_hist_B
        if "4" in pdf_list:
            coeff = 200
            self.loss_img_pdf_A = get_divergence(self.real_A, self.rec_A, pdf=4, klloss=opt.klloss) * coeff
            self.loss_img_pdf_B = get_divergence(self.real_B, self.rec_B, pdf=4, klloss=opt.klloss) * coeff
            sum = sum + self.loss_img_pdf_A + self.loss_img_pdf_B
        if "5" in pdf_list:
            coeff = 2
            self.loss_hist_pat_A = get_divergence(self.real_A, self.rec_A, pdf=5, klloss=opt.klloss) * coeff
            self.loss_hist_pat_B = get_divergence(self.real_B, self.rec_B, pdf=5, klloss=opt.klloss) * coeff
            sum = sum + self.loss_hist_pat_A + self.loss_hist_pat_B
        if "6" in pdf_list:
            coeff = 10
            total_log_loss = self.get_log_loss(self.real_A, self.rec_A, self.real_B, self.rec_B, coeff)
            sum = sum + total_log_loss
        
        return sum
        




    def backward_G(self, opt):
        """Calculate GAN loss for the discriminator

        Parameters:
            opt                 -- options
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
            js_div              -- Jensen shanon divergence for A to B and B to A

            Loss_D = log(D(X)) + 2.JS(p_r || p_g) - log(4)  

        Return the generator loss, cycle loss
        We also call loss_D.backward() to calculate the gradients.
        """

        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        """ Identity loss """
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        """Adversarial loss"""
        if opt.advloss == 0:
            "In gaussian adv loss-----------------Gen"
            # GAN loss D_A(G_A(A))
            div_G_A = self.get_adv.adv_loss(self.netD_A(self.fake_B), self.netD_A(self.real_A)) 
            self.loss_G_A_ad = div_G_A * opt.gen_coeff
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)  + self.loss_G_A_ad
            #print("generator loss fake_B", self.loss_G_A)
            # GAN loss D_B(G_B(B))
            div_G_B = self.get_adv.adv_loss(self.netD_B(self.fake_A), self.netD_B(self.real_B)) 
            self.loss_G_B_ad = div_G_B * opt.gen_coeff
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) + self.loss_G_B_ad
            #print("generator loss fake_A", self.loss_G_B)
        else:
            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) 
            #print("generator output fake_B", self.netD_A(self.fake_B).shape)
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) 
            #print("generator output fake_A", self.netD_A(self.fake_A).shape)     

        """print both GAN loss:"""
        #print("GAN loss:", self.loss_G_A, self.loss_G_B)


        """Cycle Loss"""
        if opt.cycleloss != 0:
            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        else:
            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = 0
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = 0
        
        """
        Divergence loss:

        if kl loss = 0: no divergence => div = 0
        if kl loss = 1: divergence = KL => div = kl_div
        if kl loss = 2: divergence = JS => div = js_div
        get_divergence(image1, image2, pdf, klloss=1, kernel=3, patch=8, sigma=1, agg="mean")
        pdf = 1:gaussian,2:hist,3:wt_hist,4:imagePDF,5:patch_imagePDF,6:combination of 2,4"
        """

        self.initialise_pdf_losses(opt)
        total_pdf_divergence = self.compute_pdf_losses(opt)
   
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + total_pdf_divergence
        
        self.loss_G.backward()
    


    def optimize_parameters(self, opt):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        #js_div_A_B = pdf_divergence(self.real_A, self.fake_B, opt.sigmaGen, opt.kernelGen)
        #js_div_B_A = pdf_divergence(self.real_B, self.fake_A, opt.sigmaGen, opt.kernelGen)

        #kl_div_A_B = get_divergence(self.fake_B, self.real_A, opt.sigmaGen, opt.kernelGen)
        #kl_div_B_A = get_divergence(self.fake_A, self.real_B, opt.sigmaGen, opt.kernelGen)
        #print("js_div_AB, BA:", js_div_A_B, js_div_B_A)

        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        
        self.backward_G(opt)             # calculate gradients for G_A and G_B
        #self.backward_G(opt, js_div_A_B, js_div_B_A)             # calculate gradients for G_A and G_B
        
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero

        self.backward_D_A(opt)      # calculate gradients for D_A
        self.backward_D_B(opt)      # calculate graidents for D_B

        #self.backward_D_A(opt, js_div_A_B)      # calculate gradients for D_A
        #self.backward_D_B(opt, js_div_B_A)      # calculate graidents for D_B

        #self.backward_D_A(opt, kl_div_A_B)      # calculate gradients for D_A
        #self.backward_D_B(opt, kl_div_B_A)      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
