import numpy as np

from tqdm import tqdm
import shutil
import random
import math
import gym

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils

from graphs.models.dqn import DQN
from graphs.models.replay_memory import ReplayMemory
from graphs.losses.loss import HuberLoss

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList, evaluate
from utils.misc import print_cuda_statistics

cudnn.benchmark = True

class DQNAgent:

    def __init__(self, config):
        self.config = config

        # define models (policy and target)
        self.policy_model = DQN(self.config)
        self.target_model = DQN(self.config)

        #define memory
        self.memory = ReplayMemory(self.config)

        # define dataloader
        self.batch_size = self.config.batch_size

        # define loss
        self.loss = HuberLoss()

        # define optimizer
        self.optim = torch.optim.RMSprop(self.policy_model.parameters())

        # define environment
        self.env = gym.make('CartPole-v0').unwrapped

        # initialize counter
        self.current_episode = 0
        self.current_iteration = 0
        self.episode_durations = []

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            print("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            print("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()

            self.policy_model = self.policy_model.cuda()
            self.loss = self.loss.cuda()
            self.device = "gpu"
        else:
            print("Program will run on *****CPU***** ")
            self.device = "cpu"

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='DQN')

    def load_checkpoint(self, file_name):
        filename = self.config.checkpoint_dir + file_name
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.policy_model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.fixed_noise = checkpoint['fixed_noise']
            self.manual_seed = checkpoint['manual_seed']

            print("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            print("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best = 0):
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.policy_model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'fixed_noise': self.fixed_noise,
            'manual_seed': self.manual_seed
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + file_name)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + file_name,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.config.eps_start + (self.config.eps_start - self.config.eps_end) * math.exp(-1. * self.current_iteration / self.config.eps_decay)
        self.current_iteration += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_model(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)

    def train(self):
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.save_checkpoint()

    def train_one_epoch(self):
        # initialize tqdm batch

        epoch_lossG = AverageMeter()
        epoch_lossD = AverageMeter()


        for curr_it, x in enumerate(tqdm_batch):
            #y = torch.full((self.batch_size,), self.real_label)
            x = x[0]
            y = torch.randn(x.size(0), )
            fake_noise = torch.randn(x.size(0), self.config.g_input_size, 1, 1)

            if self.cuda:
                x = x.cuda(async=self.config.async_loading)
                y = y.cuda(async=self.config.async_loading)
                fake_noise = fake_noise.cuda(async=self.config.async_loading)

            x = Variable(x)
            y = Variable(y)
            fake_noise = Variable(fake_noise)
            ####################
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # train with real
            self.netD.zero_grad()
            D_real_out = self.netD(x)
            y.fill_(self.real_label)
            loss_D_real = self.loss(D_real_out, y)
            loss_D_real.backward()
            #D_mean_real_out = D_real_out.mean().item()

            # train with fake
            G_fake_out = self.netG(fake_noise)
            y.fill_(self.fake_label)

            D_fake_out = self.netD(G_fake_out.detach())

            loss_D_fake = self.loss(D_fake_out, y)
            loss_D_fake.backward()
            #D_mean_fake_out = D_fake_out.mean().item()

            loss_D = loss_D_fake + loss_D_real
            self.optimD.step()

            ####################
            # Update G network: maximize log(D(G(z)))
            self.netG.zero_grad()
            y.fill_(self.real_label)
            D_out = self.netD(G_fake_out)
            loss_G = self.loss(D_out, y)
            loss_G.backward()

            #D_G_mean_out = D_out.mean().item()

            self.optimG.step()

            epoch_lossD.update(loss_D.data[0])
            epoch_lossG.update(loss_G.data[0])

            self.current_iteration += 1

            self.summary_writer.add_scalar("epoch/Generator_loss", epoch_lossG.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch/Discriminator_loss", epoch_lossD.val, self.current_iteration)

            if self.current_iteration % 1000 ==  0:
                self.summary_writer.add_image("train/Real_Image", x, self.current_iteration)
                gen_out = self.netG(self.fixed_noise)

                out_img = self.dataloader.plot_samples_per_epoch(gen_out.data, self.current_iteration)
                self.summary_writer.add_image('train/generated_image', out_img, self.current_iteration)
                self.summary_writer.add_image("Generated Images",out_img, self.current_iteration)

        tqdm_batch.close()

        print("Training at epoch-" + str(self.current_epoch) + " | " + "Discriminator loss: " + str(
            epoch_lossD.val) + " - Generator Loss-: " + str(epoch_lossG.val))


    def validate(self):
        pass

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.dataloader.finalize()
