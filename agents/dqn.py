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
from graphs.models.replay_memory import ReplayMemory, Transition
from graphs.losses.loss import HuberLoss

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList, evaluate
from utils.misc import print_cuda_statistics
from utils.extract_env_input import get_screen

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

            self.current_epoch = checkpoint['episode']
            self.current_iteration = checkpoint['iteration']
            self.policy_model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])

            print("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            print("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best = 0):
        state = {
            'episode': self.current_episode,
            'iteration': self.current_iteration,
            'state_dict': self.policy_model.state_dict(),
            'optimizer': self.optim.state_dict(),
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

    def optimized_policy_model(self):
        if self.memory.length() < self.batch_size:
            return
        # sample a batch
        transitions = self.memory.sample_batch(self.batch_size)

        # print("Transitions", transitions)
        one_batch = Transition(*zip(*transitions))
        # print("One batch", one_batch)

        # concatenate all tensors into one
        state_batch = torch.cat(one_batch.state)
        action_batch = torch.cat(one_batch.action)
        reward_batch = torch.cat(one_batch.reward)

        # debug here
        curr_state_values = self.policy_model(state_batch)
        curr_state_action_values = curr_state_values.gather(1, action_batch)

        # create a mask of non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,one_batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in one_batch.next_state if s is not None])

        # Get V(s_{t+1}) for all next states. By defition we set V(s)=0 if s is a terminal state.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()

        # Get the expected Q values
        expected_state_action_values = (next_state_values * self.config.gamma) + reward_batch

        # compute loss: temporal difference error
        loss = self.loss(curr_state_action_values, expected_state_action_values.unsqueeze(1))

        # optimizer step
        self.optim.step()
        loss.backward()
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

    def train(self):
        for episode in tqdm(range(self.current_episode, self.config.num_episodes)):
            self.current_episode = episode
            self.train_one_epoch()
            # update the target model???
            # The target network has its weights kept frozen most of the time, but is updated with the policy network’s weights every so often
            if self.current_episode % self.config.target_update:
                self.target_model.load_state_dict(self.policy_model.state_dict())

        print('Complete')
        self.env.render()
        self.env.close()

    def train_one_epoch(self):
        # reset environment
        self.env.reset()
        prev_frame = get_screen(self.env, self.config.screen_width)
        curr_frame = get_screen(self.env, self.config.screen_width)
        # get state
        curr_state = curr_frame - prev_frame

        while(1):
            # select action
            action = self.select_action(curr_state)
            # perform action and get reward
            _, reward, done, _ = self.env.step(action.item())
            if self.cuda:
                reward = torch.Tensor([reward]).cuda()
            else:
                reward = torch.Tensor([reward])

            prev_frame = curr_frame
            curr_frame = get_screen(self.env, self.config.screen_width)
            # assign next state
            if done:
                next_state = None
            else:
                next_state = curr_frame - prev_frame

            # add this transition into memory
            self.memory.push_transition(curr_state, action, next_state, reward)

            curr_state = next_state

            # Policy model optimization step #
            self.optimized_policy_model()

            # check if done
            if done:
                break

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