import torch
import gymnasium as gym

class TorchToNumpyWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._convert(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        return (
            self._convert(obs),
            self._convert_reward(reward),
            self._convert_done(done),
            truncated,
            self._convert_info(info),
        )

    def _convert(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x

    def _convert_reward(self, r):
        if torch.is_tensor(r):
            return r.detach().cpu().numpy()
        return r

    def _convert_done(self, d):
        if torch.is_tensor(d):
            return d.detach().cpu().numpy()
        return d

    def _convert_info(self, info):
        if isinstance(info, dict):
            return {
                k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in info.items()
            }
        return info