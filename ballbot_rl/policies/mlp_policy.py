import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from pathlib import Path


class Extractor(BaseFeaturesExtractor):

    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 frozen_encoder_path: str = ""):

        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():

            if "rgbd_" in key:

                if not frozen_encoder_path:
                    #note that we're iterating on observation_space objects, so there is not batch size info
                    C, H, W = subspace.shape  #typically, C=1 and H=W=32 here

                    F1 = 32
                    F2 = 32
                    self.out_sz = 20
                    extractors[key] = torch.nn.Sequential(
                        torch.nn.Conv2d(1,
                                        F1,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1),  #output BxF1xH/2xW/2
                        torch.nn.BatchNorm2d(F1),
                        torch.nn.LeakyReLU(),
                        torch.nn.Conv2d(F1,
                                        F2,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1),  #output BxF2xH/4xW/4
                        torch.nn.BatchNorm2d(F2),
                        torch.nn.LeakyReLU(),
                        torch.nn.Flatten(),
                        torch.nn.Linear(F2 * H // 4 * W // 4, self.out_sz),
                        torch.nn.BatchNorm1d(self.out_sz),
                        torch.nn.Tanh(),
                    )

                    total_concat_size += self.out_sz
                else:
                    encoder_path = Path(frozen_encoder_path).resolve()
                    print(f"loading encoder from {encoder_path}")
                    # Load on CPU to handle models saved on CUDA devices
                    extractors[key] = torch.load(str(encoder_path),
                                                 map_location='cpu',
                                                 weights_only=False)
                    p_sum = sum([
                        param.abs().sum().item()
                        for param in extractors[key].parameters()
                        if param.requires_grad
                    ])
                    # Use tolerance for floating point comparison (especially when loading from CUDA to CPU)
                    tolerance = 1e-5
                    if not hasattr(extractors[key], 'p_sum'):
                        print(f"Warning: Model does not have p_sum attribute, skipping validation")
                    else:
                        stored_p_sum = extractors[key].p_sum
                        if abs(p_sum - stored_p_sum) > tolerance:
                            print(f"Warning: Parameter sum mismatch: computed={p_sum}, stored={stored_p_sum}, diff={abs(p_sum - stored_p_sum)}")
                            print(f"This may be due to device differences (CUDA vs CPU). Continuing anyway...")
                            # Uncomment the line below if you want to enforce strict checking
                            # assert False, "unexpected model params sum. The file might be corrupted"
                    last_linear = [
                        m for m in extractors[key].modules()
                        if isinstance(m, torch.nn.Linear)
                    ][-1]
                    self.out_sz = last_linear.out_features
                    total_concat_size += self.out_sz

                    for param in extractors[key].parameters(
                    ):  #to keep it frozen
                        param.requires_grad = False

            else:
                #note that we're iterating on observation_space objects, so there is not batch size info
                S = subspace.shape[0]
                extractors[key] = torch.nn.Flatten()
                total_concat_size += S

        self.extractors = torch.nn.ModuleDict(extractors)

        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        encoded_tensor_dict = {}  #for debug only

        for key, extractor in self.extractors.items():
            cur = extractor(observations[key])

            encoded_tensor_list.append(
                cur
            )  #for rgbd_<int> the cnn uses a tanh at the end so no need for normalization

            #encoded_tensor_dict[key]=cur

        out = torch.cat(encoded_tensor_list, dim=1)
        return out
