import argparse
import pickle
import numpy as np
from bayes_opt import BayesianOptimization

# for REINFORCE
import torch
import torch.nn as nn
from torch.distributions import Categorical

def parse_arguments():
    parser = argparse.ArgumentParser(description="Search on NAS-Bench-201")
    parser.add_argument('task', choices=['C10', 'C100', 'IN-16'])
    parser.add_argument('--search_budget', default=20, type=int)
    parser.add_argument('--training_free_metrics', nargs='+',
                        default=['grad_norm', 'snip', 'grasp', 'fisher', 'synflow', 'jacob_cov'],
                        help='List of training-free metrics')
    parser.add_argument('--method', choices=['RoBoT', 'RS', 'REA', 'REINFORCE'], default='RoBoT', help='Method for NAS')
    parser.add_argument('--seed_start', default=10, type=int)
    parser.add_argument('--seed_end', default=20, type=int)
    args = parser.parse_args()
    return args


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


def RS():
    opt_test_accs_list_rs = []
    search_costs = []
    for seed in range(args.seed_start, args.seed_end):
        np.random.seed(seed)
        opt_val_acc = min(val_accs) - 1
        opt_test_acc = min(test_accs) - 1
        opt_test_accs = []
        searched = []
        cost = 0
        for i in range(running_rounds):
            while True:
                arch = np.random.randint(num_arch)
                if arch not in searched:
                    cost += costs[arch]
                    searched.append(arch)
                    val_acc = val_accs[arch]
                    if val_acc > opt_val_acc:
                        opt_val_acc = val_acc
                        opt_test_acc = test_accs[arch]
                    opt_test_accs.append(opt_test_acc)
                    break
        search_costs.append(cost)
        opt_test_accs_list_rs.append(opt_test_accs)

    mean = np.mean(opt_test_accs_list_rs, axis=0)[-1]
    std = np.std(opt_test_accs_list_rs, axis=0)[-1]
    average_cost = np.mean(search_costs)
    print("Test Accs:" + str(mean) + "±" + str(std) + " Search Costs:" + str(average_cost))
    return


def REA():
    opt_test_accs_list_rea = []
    search_costs = []
    max_node = 5

    _opname_to_index = {
        'none': '0',
        'skip_connect': '1',
        'nor_conv_1x1': '2',
        'nor_conv_3x3': '3',
        'avg_pool_3x3': '4'
    }

    def get_spec_from_arch_str(arch_str):
        nodes = arch_str.split('+')
        nodes = [node[1:-1].split('|') for node in nodes]
        nodes = [[op_and_input.split('~')[0] for op_and_input in node] for node in nodes]

        spec = [_opname_to_index[op] for node in nodes for op in node]
        return spec

    id_to_spec = {}
    for i in range(len(data_stats)):
        arch_str = data_stats[i]['arch']
        id_to_spec[i] = get_spec_from_arch_str(arch_str)

    spec_to_id = {}
    for i, spec in id_to_spec.items():
        spec_to_id[str(spec)] = i

    def mutate_spec(old_id, searched):
        old_spec = id_to_spec[old_id]
        while True:
            idx_to_change = np.random.randint(len(old_spec))
            entry_to_change = old_spec[idx_to_change]
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
        cost = 0
        for arch in searched:
            cost += costs[arch]
        search_costs.append(cost)
        opt_test_accs_list_rea.append(opt_test_accs)

    mean = np.mean(opt_test_accs_list_rea, axis=0)[-1]
    std = np.std(opt_test_accs_list_rea, axis=0)[-1]
    average_cost = np.mean(search_costs)
    print("Test Accs:" + str(mean) + "±" + str(std) + " Search Costs:" + str(average_cost))
    return

def REINFORCE():
    _opname_to_index = {
        'none': '0',
        'skip_connect': '1',
        'nor_conv_1x1': '2',
        'nor_conv_3x3': '3',
        'avg_pool_3x3': '4'
    }

    def get_spec_from_arch_str(arch_str):
        nodes = arch_str.split('+')
        nodes = [node[1:-1].split('|') for node in nodes]
        nodes = [[op_and_input.split('~')[0] for op_and_input in node] for node in nodes]

        spec = [_opname_to_index[op] for node in nodes for op in node]
        return spec

    id_to_spec = {}
    for i in range(len(data_stats)):
        arch_str = data_stats[i]['arch']
        id_to_spec[i] = get_spec_from_arch_str(arch_str)

    spec_to_id = {}
    for i, spec in id_to_spec.items():
        spec_to_id[str(spec)] = i

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
    search_costs = []
    max_node = 5
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
        cost = 0
        for arch in searched:
            cost += costs[arch]
        search_costs.append(cost)
        opt_test_accs_list_rl.append(opt_test_accs)

    mean = np.mean(opt_test_accs_list_rl, axis=0)[-1]
    std = np.std(opt_test_accs_list_rl, axis=0)[-1]
    average_cost = np.mean(search_costs)
    print("Test Accs:" + str(mean) + "±" + str(std) + " Search Costs:" + str(average_cost))
    return


if __name__ == '__main__':
    args = parse_arguments()

    if args.task == "C10":
        training_free_metrics_file = "data/nb2_cf10_seed42_dlrandom_dlinfo1_initwnone_initbnone.p"
        test_acc_file = "data/nb2_cf10_test_accuracy.p"
    elif args.task == "C100":
        training_free_metrics_file = "data/nb2_cf100_seed42_dlrandom_dlinfo1_initwnone_initbnone.p"
        test_acc_file = "data/nb2_cf100_test_accuracy.p"
    else:
        training_free_metrics_file = "data/nb2_im120_seed42_dlrandom_dlinfo1_initwnone_initbnone.p"
        test_acc_file = "data/nb2_im120_test_accuracy.p"

    data_stats = []
    # training-free metrics. can collect validation acc as well
    f = open(training_free_metrics_file, 'rb')
    while True:
        try:
            data_stats.append(pickle.load(f))
        except EOFError:
            break
    f.close()

    num_arch = len(data_stats)
    metric_names = args.training_free_metrics
    data_metrics = {}
    for metric in metric_names:
        data_metrics[metric] = []
    for i in range(len(data_stats)):
        for metric in metric_names:
            data_metrics[metric].append(data_stats[i]['logmeasures'][metric])

    # validation accs and costs
    file = "data/nb2_cf10_hp12_info.p"
    with open(file, 'rb') as f:
        info = pickle.load(f)

    costs = []
    val_accs = []
    for key in info:
        costs.append(key['cost'])
        val_accs.append(key['valacc'])

    # test accs
    with open(test_acc_file, 'rb') as f:
        test_accs = pickle.load(f)
    test_order = np.flip(np.argsort(test_accs))
    test_rank = np.argsort(test_order)

    running_rounds = args.search_budget
    # search
    if args.method == "RS":
        RS()
    elif args.method == "REA":
        REA()
    elif args.method == "REINFORCE":
        REINFORCE()
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
        costs_list = []
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
            cost = 0
            for i in range(len(val_accs_opt_archs)):
                acc = val_accs_opt_archs[i]
                arch = opt_archs[i]
                cost += costs[arch]
                if acc > opt_acc:
                    opt_acc = acc
                    opt_arch = arch
                highest_test_accs.append(test_accs[opt_arch])
            print("Test accs of selected architecture for this round:"+ str(highest_test_accs[-1]))
            print("Search costs for this round:"+ str(cost))
            highest_test_accs_list.append(highest_test_accs)
            costs_list.append(cost)
        mean = np.mean(highest_test_accs_list, axis=0)[-1]
        std = np.std(highest_test_accs_list, axis=0)[-1]
        average_cost = np.mean(costs_list)
        print("Test Accs:" + str(mean) + "±" + str(std) + " Search Costs:" + str(average_cost))


