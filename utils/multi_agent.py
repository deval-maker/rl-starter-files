import torch

import utils
from model_ma import ACModelMA


class MultiAgent:
    """A class for multi-agent inference.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir, n_agents, obs_dim,
                 device=None, argmax=False, num_envs=1, use_memory=False, use_text=False):
        
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        self.acmodel = ACModelMA(obs_space, action_space, n_agents, obs_dim, 
                                 use_memory=use_memory, use_text=use_text)
        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs
        self.num_agents = n_agents

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs*self.num_agents, self.acmodel.memory_size, device=self.device)

        self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        self.acmodel.to(self.device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def get_actions(self, obss):

        agent_obs, env_obs = self.preprocess_obss(obss, device=self.device)

        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(agent_obs, env_obs, self.memories)
            else:
                dist, _ = self.acmodel(agent_obs, env_obs)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=False)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
