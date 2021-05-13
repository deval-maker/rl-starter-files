import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModelMA(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, n_agents, obs_dim, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.n_agents = n_agents

        # Define image embedding

        self.image_conv = nn.Sequential(
            nn.Conv2d(obs_dim, 32, (5, 5)),
            nn.ReLU(),
            # nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(128, 256, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024)
        )

        n = obs_space["image"][1]
        m = obs_space["image"][2]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )
        # self.actor = nn.Sequential(
        #     nn.Linear(self.embedding_size, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, action_space.n)
        # )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )        
        # self.critic = nn.Sequential(
        #     nn.Linear(self.embedding_size*2, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 1)
        # )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    # 8*2 x 13 x 13 x7, 8 x 2 x 13x 13 x 7
    def forward(self, agent_obs, env_obs, memory):
        
        # import ipdb; ipdb.set_trace()
        # agent obs - 256 x 13 x 13 x 7
        # env obs - 256 x 2 x 13 x 13 x 7 -> 512 x 13 x 13 x 7 -> 512 x 256 x 3 x 3 -> 256 x 2 x 256x 3 x3 -> 256 x 512 x 3 x 3 -> 256 x 512*3*3 -> 256
        # agent_obs -> 32 x 7 x 7 x 6
        x_actor = agent_obs.image.transpose(1, 3).transpose(2, 3)
        x_actor = self.image_conv(x_actor)
        # x_actor = x_actor.reshape(x_actor.shape[0], -1)

        if self.use_memory: #memory-> 10*2 x 2048
            # memory = memory.flatten(start_dim=0, end_dim=1)
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x_actor, hidden)
            x_actor = hidden[0]
            memory = torch.cat(hidden, dim=1)

        # 32 x 7 x 7 x 6 -> 32
        x_actor = self.actor(x_actor)
        dist = Categorical(logits=F.log_softmax(x_actor, dim=1))

        # else:
        #     embedding = x

        # if self.use_text:
        #     embed_text = self._get_embed_text(obs.text)
        #     embedding = torch.cat((embedding, embed_text), dim=1)

        # 16x2x7x7x6 -> 16x7x7x6x2 -> 16x7x7x12 # 8x2x
        # x_critic = env_obs.image.permute(0, 2, 3, 4, 1)
        # x_critic = torch.flatten(x_critic, start_dim=3, end_dim=4)
        # x_critic = x_critic.transpose(1, 3).transpose(2, 3)
        # x_critic = self.image_conv_critic(x_critic)
        # x_critic = x_critic.reshape(x_critic.shape[0], -1)

        #256x2x13x13x7 -> 512 * 13 * 13 * 7 -> 512 * 7 * 13 * 13 -> 512 * 256 * 3 * 3 -> 256 * 2 * 256 * 3 * 3
        # 256 * 512 * 3 * 3 -> 256 * 512*3*3
        x_critic = env_obs.image.flatten(start_dim=0, end_dim=1)
        x_critic = x_critic.permute(0,3,1,2)
        x_critic = self.image_conv(x_critic)

        x_critic = x_critic.reshape(-1, self.n_agents, *x_critic.shape[1:])

        x_critic = torch.sum(x_critic, dim=1)
        
        # 16 x 7 x 7 x 12 -> 16
        x_critic = self.critic(x_critic)
        value = x_critic.squeeze(1)

        # memory = memory.reshape(-1, self.n_agents, memory.shape[1])

        # dist -> 32, value -> 16
        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
