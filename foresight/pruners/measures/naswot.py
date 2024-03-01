# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
import numpy as np

from . import measure

@measure('naswot', bn=True)
def compute_naswot(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    net.zero_grad()
    
    net.K = np.zeros((inputs.shape[0], inputs.shape[0]))
    def counting_forward_hook(module, inp, out):
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.view(inp.size(0), -1)
        x = inp.float()
        K = x @ x.t()
        net.K += K.detach().cpu().numpy()
        
    for name, module in net.named_modules():
        if 'ReLU' in str(type(module)):
            module.register_forward_hook(counting_forward_hook)

    _ = net.forward(inputs)
    s, ld = np.linalg.slogdet(net.K)
    return ld
