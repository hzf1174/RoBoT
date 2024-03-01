import argparse
import json
import numpy as np
from bayes_opt import BayesianOptimization

# for REINFORCE
import torch
import torch.nn as nn
from torch.distributions import Categorical

def parse_arguments():
    parser = argparse.ArgumentParser(description="Search on TransNAS-Bench-101")
    parser.add_argument('benchmark', choices=['micro', 'macro'])
    parser.add_argument('task', choices=['class_scene', 'class_object', 'room_layout', 'jigsaw', 'segmentsemantic', 'normal', 'autoencoder'])
    parser.add_argument('--search_budget', default=100, type=int)
    parser.add_argument('--training_free_metrics', nargs='+',
                        default=['grad_norm', 'snip', 'grasp', 'fisher', 'synflow', 'jacov'],
                        help='List of training-free metrics')
    parser.add_argument('--method', choices=['RoBoT', 'RS', 'REA', 'REINFORCE'], default='RoBoT', help='Method for NAS')
    parser.add_argument('--seed_start', default=10, type=int)
    parser.add_argument('--seed_end', default=20, type=int)
    args = parser.parse_args()
    return args


def RS():
    opt_test_accs_list_rs = []
    for seed in range(args.seed_start, args.seed_end):
        np.random.seed(seed)
        opt_val_acc = min(val_accs) - 1
        opt_test_acc = min(test_accs) - 1
        opt_test_accs = []
        searched = []
        for i in range(running_rounds):
            while True:
                arch = np.random.randint(num_arch)
                if arch not in searched:
                    searched.append(arch)
                    val_acc = val_accs[arch]
                    if val_acc > opt_val_acc:
                        opt_val_acc = val_acc
                        opt_test_acc = test_accs[arch]
                    opt_test_accs.append(opt_test_acc)
                    break
        opt_test_accs_list_rs.append(opt_test_accs)

    mean = np.mean(opt_test_accs_list_rs, axis=0)[-1]
    std = np.std(opt_test_accs_list_rs, axis=0)[-1]
    print("Test Performance:" + str(mean) + "±" + str(std))
    return


def REA():
    opt_test_accs_list_rea = []
    max_node = 4

    import ast
    id_to_spec = {}
    spec_to_id = {}

    for spec in data_stats[xtask]:
        spec_list = ast.literal_eval(spec)
        new_spec = []
        for op in spec_list:
            new_spec.append(str(op))
        id_to_spec[data_stats[xtask][spec]['id']] = new_spec
        new_spec = str(new_spec)
        spec_to_id[new_spec] = data_stats[xtask][spec]['id']

    def mutate_spec(old_id, searched):
        old_spec = id_to_spec[old_id]
        while True:
            idx_to_change = np.random.randint(len(old_spec))
            entry_to_change = old_spec[idx_to_change]
            if args.benchmark == "macro":
                possible_entries = [x for x in range(1, 1 + max_node) if x != entry_to_change]
            else:
                possible_entries = [x for x in range(max_node) if x != entry_to_change]
            new_entry = np.random.choice(possible_entries)
            new_spec = list(old_spec)
            new_spec[idx_to_change] = str(new_entry)
            new_spec = str(new_spec)
            if new_spec not in spec_to_id:
                continue
            new_id = spec_to_id[new_spec]
            if new_id not in searched:
                break
        return new_id

    search_size = running_rounds
    pot_size = int(running_rounds / 3)
    for seed in range(args.seed_start, args.seed_end):
        np.random.seed(seed)
        opt_val_acc = min(val_accs) - 1
        opt_test_acc = min(test_accs) - 1
        opt_test_accs = []
        searched = []
        init = np.random.randint(num_arch, size=pot_size)
        pool = []  # (valid accs, arch number)
        for arch in init:
            val_acc = val_accs[arch]
            if val_acc > opt_val_acc:
                opt_val_acc = val_acc
                opt_test_acc = test_accs[arch]
            opt_test_accs.append(opt_test_acc)
            searched.append(arch)
            pool.append((val_acc, arch))

        while len(searched) != search_size:
            pool = sorted(pool, key=lambda i: -i[0])
            _, best_arch = pool[0]
            pool.pop(0)
            new_arch = mutate_spec(best_arch, searched)

            if new_arch not in searched:
                val_acc = val_accs[new_arch]
                if val_acc > opt_val_acc:
                    opt_val_acc = val_acc
                    opt_test_acc = test_accs[new_arch]
                opt_test_accs.append(opt_test_acc)
                searched.append(new_arch)
                pool.append((val_acc, new_arch))
        opt_test_accs_list_rea.append(opt_test_accs)

    mean = np.mean(opt_test_accs_list_rea, axis=0)[-1]
    std = np.std(opt_test_accs_list_rea, axis=0)[-1]
    print("Test Performance:" + str(mean) + "±" + str(std))
    return

def REINFORCE_micro():
    import ast
    id_to_spec = {}
    spec_to_id = {}

    for spec in data_stats[xtask]:
        spec_list = ast.literal_eval(spec)
        new_spec = []
        for op in spec_list:
            new_spec.append(str(op))
        id_to_spec[data_stats[xtask][spec]['id']] = new_spec
        new_spec = str(new_spec)
        spec_to_id[new_spec] = data_stats[xtask][spec]['id']

    class ExponentialMovingAverage(object):
        """Class that maintains an exponential moving average."""

        def __init__(self, momentum):
            self._numerator = 0
            self._denominator = 0
            self._momentum = momentum

        def update(self, value):
            self._numerator = (
                    self._momentum * self._numerator + (1 - self._momentum) * value
            )
            self._denominator = self._momentum * self._denominator + (1 - self._momentum)

        def value(self):
            """Return the current value of the moving average"""
            return self._numerator / self._denominator

    def select_action(policy):
        probs = nn.functional.softmax(policy, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        return m.log_prob(action), action.cpu().tolist()

    learning_rate = 0.01
    search_size = running_rounds
    opt_test_accs_list_rl = []
    max_node = 4
    for seed in range(args.seed_start, args.seed_end):
        torch.manual_seed(seed)
        policy = nn.Parameter(1e-3 * torch.randn(6, max_node))
        baseline = ExponentialMovingAverage(0.9)
        optimizer = torch.optim.Adam([policy], lr=learning_rate)

        opt_val_acc = min(val_accs) - 1
        opt_test_acc = min(test_accs) - 1
        opt_test_accs = []
        searched = []

        while len(searched) != search_size:
            log_prob, action = select_action(policy)
            for i in range(len(action)):
                action[i] = str(action[i])
            action = str(action)
            arch = spec_to_id[action]
            val_acc = val_accs[arch]

            if arch not in searched:
                if val_acc > opt_val_acc:
                    opt_val_acc = val_acc
                    opt_test_acc = test_accs[arch]
                opt_test_accs.append(opt_test_acc)
                searched.append(arch)

            baseline.update(val_acc)
            # calculate loss
            policy_loss = (-log_prob * (val_acc - baseline.value())).sum()
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
        opt_test_accs_list_rl.append(opt_test_accs)

    mean = np.mean(opt_test_accs_list_rl, axis=0)[-1]
    std = np.std(opt_test_accs_list_rl, axis=0)[-1]
    print("Test Performance:" + str(mean) + "±" + str(std))
    return


def REINFORCE_macro():
    import ast
    id_to_spec = {}
    spec_to_id = {}

    for spec in data_stats[xtask]:
        spec_list = ast.literal_eval(spec)
        new_spec = []
        for op in spec_list:
            new_spec.append(str(op))
        id_to_spec[data_stats[xtask][spec]['id']] = new_spec
        new_spec = str(new_spec)
        spec_to_id[new_spec] = data_stats[xtask][spec]['id']

    from itertools import combinations

    def combination_mapping(n, k):
        elements = list(range(1, n + 1))
        comb_list = list(combinations(elements, k))

        reverse_comb_map = []
        for index, comb in enumerate(comb_list):
            reverse_comb_map.append(comb)

        return reverse_comb_map
    layer_actions_to_arch = {}
    for i in range(3):
        layer = i + 4
        long_mapping = []
        for choice in range(4):
            reverse_mapping = combination_mapping(layer, choice + 1)
            long_mapping = long_mapping + reverse_mapping
        layer_actions_to_arch[i] = long_mapping

    def transfer_actions_arch(layer, actions):
        layer = layer + 4
        return_list = []
        for i in range(layer):
            return_list.append(1)
        for i in range(2):
            action = actions[i]
            mapping = layer_actions_to_arch[layer - 4][action]
            for position in mapping:
                if i == 0:
                    return_list[position - 1] += 1
                else:
                    return_list[position - 1] += 2
        for i in range(layer):
            return_list[i] = str(return_list[i])
        return_list = str(return_list)
        return spec_to_id[return_list]

    class ExponentialMovingAverage(object):
        """Class that maintains an exponential moving average."""

        def __init__(self, momentum):
            self._numerator = 0
            self._denominator = 0
            self._momentum = momentum

        def update(self, value):
            self._numerator = (
                    self._momentum * self._numerator + (1 - self._momentum) * value
            )
            self._denominator = self._momentum * self._denominator + (1 - self._momentum)

        def value(self):
            """Return the current value of the moving average"""
            return self._numerator / self._denominator

    class LayeredPolicy(nn.Module):
        def __init__(self, layer_shapes):
            super(LayeredPolicy, self).__init__()

            self.num_layers = len(layer_shapes)

            # Create learnable parameters for each layer and parameter shape
            self.params = nn.ParameterList(
                [nn.ParameterList([nn.Parameter(1e-3 * torch.randn(shape)) for shape in layer]) for layer in
                 layer_shapes])
            self.layer_probs = nn.Parameter(1e-3 * torch.randn(self.num_layers))

        def forward(self):
            layer_softmax_probs = nn.functional.softmax(self.layer_probs, dim=0)
            param_softmax_probs = [[nn.functional.softmax(param, dim=0) for param in layer] for layer in self.params]
            return layer_softmax_probs, param_softmax_probs

    learning_rate = 0.01
    search_size = running_rounds
    opt_test_accs_list_rl = []
    for seed in range(50):
        layer_shapes = [[(14,), (15,)], [(25,), (30,)], [(41,), (56,)]]
        policy = LayeredPolicy(layer_shapes)
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        baseline = ExponentialMovingAverage(0.9)

        opt_val_acc = min(val_accs) - 1
        opt_test_acc = min(test_accs) - 1
        opt_test_accs = []
        searched = []

        while len(searched) != search_size:
            # Sample a layer
            layer_probs, param_probs = policy.forward()
            layer_dist = Categorical(layer_probs)
            chosen_layer = layer_dist.sample().item()

            # Sample an action for each parameter in the chosen layer
            chosen_param_probs = param_probs[chosen_layer]
            actions = [Categorical(param_prob).sample().item() for param_prob in chosen_param_probs]
            arch = transfer_actions_arch(chosen_layer, actions)
            val_acc = val_accs[arch]

            if arch not in searched:
                if val_acc > opt_val_acc:
                    opt_val_acc = val_acc
                    opt_test_acc = test_accs[arch]
                opt_test_accs.append(opt_test_acc)
                searched.append(arch)
            # Compute the loss
            baseline.update(val_acc)
            loss = -layer_dist.log_prob(torch.tensor(chosen_layer)) * val_acc
            for param_prob, action in zip(chosen_param_probs, actions):
                loss += -Categorical(param_prob).log_prob(torch.tensor(action)) * val_acc

            # Update the policy using the loss (use your preferred optimizer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        opt_test_accs_list_rl.append(opt_test_accs)

    mean = np.mean(opt_test_accs_list_rl, axis=0)[-1]
    std = np.std(opt_test_accs_list_rl, axis=0)[-1]
    print("Test Performance:" + str(mean) + "±" + str(std))
    return

def search(weights, datas, num_arch, metric_names):
    data_metrics, val_accs = datas

    global opt_archs, opt_weights, opt_target

    score_list = []
    for arch in list(range(num_arch)):
        score = 0
        for metric in metric_names:
            weight = weights[metric]
            if not np.isnan(data_metrics[metric][arch]):
                score += weight * data_metrics[metric][arch]
        score_list.append(score)

    score_list_order = np.flip(np.argsort(score_list))

    if val_accs[score_list_order[0]] >= opt_target:
        opt_target = val_accs[score_list_order[0]]
        opt_weights = weights
    if score_list_order[0] not in opt_archs:
        opt_archs = opt_archs + [score_list_order[0]]

    return val_accs[score_list_order[0]]

if __name__ == '__main__':
    args = parse_arguments()

    if args.benchmark == "micro":
        file = "data/zc_transbench101_micro.json"
    else:
        file = "data/zc_transbench101_macro.json"

    xtask = args.task
    metric_names = args.training_free_metrics

    if xtask in ['segmentsemantic', 'normal', 'autoencoder']:
        if "synflow" in metric_names:
            metric_names.remove("synflow")

    with open(file, "r") as f:
        data_stats = json.load(f)

    num_arch = len(data_stats[xtask])
    # training-free metrics, validation accs and dummy test accs
    val_accs = []
    data_metrics = {}
    for metric in metric_names:
        data_metrics[metric] = []
    for data_key in data_stats[xtask]:
        data = data_stats[xtask][data_key]
        val_accs.append(data['val_accuracy'])

        for metric in metric_names:
            data_metrics[metric].append(data[metric]['score'])
    test_accs = val_accs
    test_order = np.flip(np.argsort(test_accs))
    test_rank = np.argsort(test_order)

    running_rounds = args.search_budget
    # search
    if args.method == "RS":
        RS()
    elif args.method == "REA":
        REA()
    elif args.method == "REINFORCE":
        if args.benchmark == "micro":
            REINFORCE_micro()
        else:
            REINFORCE_macro()
    else:  # main method RoBoT
        # training-free metrics normalization
        for metric_name in metric_names:
            #print(metric_name, max(data_metrics[metric_name]), min(data_metrics[metric_name]))
            if max(data_metrics[metric_name]) - min(data_metrics[metric_name]) == 0:
                data_metrics[metric_name] = [float(i) for i in data_metrics[metric_name]]
            else:
                data_metrics[metric_name] = [(float(i) - min(data_metrics[metric_name])) /
                                             (max(data_metrics[metric_name]) - min(data_metrics[metric_name]))
                                             for i in data_metrics[metric_name]]

        highest_test_accs_list = []
        for seed in range(args.seed_start, args.seed_end):
            print("Running the "+str(seed - args.seed_start + 1)+"th round")
            datas = [data_metrics, val_accs]

            # the domain for search
            pbounds = {}
            for metric_name in metric_names:
                pbounds[metric_name] = (-1, 1)

            val_order = np.flip(np.argsort(val_accs))
            val_ranks = np.argsort(val_order)
            opt_archs = []
            opt_weights = []
            opt_target = min(val_accs) - 1
            optimizer = BayesianOptimization(
                f=lambda **kwargs: search(kwargs, datas, num_arch, metric_names),
                pbounds=pbounds,
                verbose=2,
                random_state=seed
            )

            # exploration part
            optimizer.maximize(
                init_points=0,
                n_iter=running_rounds - 1,
                acq='ucb',
            )

            print("T0 for this round:"+str(len(opt_archs)))

            # exploitation part
            opt_score_list = []
            for arch in list(range(num_arch)):
                score = 0
                for metric in metric_names:
                    weight = opt_weights[metric]
                    if not np.isnan(data_metrics[metric][arch]):
                        score += weight * data_metrics[metric][arch]
                opt_score_list.append(score)
            score_list_order = np.flip(np.argsort(opt_score_list))

            for i in range(len(score_list_order)):
                if len(opt_archs) == running_rounds:
                    break
                arch = score_list_order[i]
                if arch not in opt_archs:
                    opt_archs = opt_archs + [arch]

            val_accs_opt_archs = [val_accs[arch] for arch in opt_archs]

            highest_test_accs = []
            opt_acc = min(val_accs) - 1
            opt_arch = 0
            for i in range(len(val_accs_opt_archs)):
                acc = val_accs_opt_archs[i]
                arch = opt_archs[i]
                if acc > opt_acc:
                    opt_acc = acc
                    opt_arch = arch
                highest_test_accs.append(test_accs[opt_arch])
            print("Test performance of selected architecture for this round:"+ str(highest_test_accs[-1]))
            highest_test_accs_list.append(highest_test_accs)
        mean = np.mean(highest_test_accs_list, axis=0)[-1]
        std = np.std(highest_test_accs_list, axis=0)[-1]
        print("Test Performance:" + str(mean) + "±" + str(std))
