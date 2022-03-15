import os
import collections
from this import d
import tqdm
import multiprocessing
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from generator import Generator
from discriminator import Discriminator
import utils
from evaluation import link_prediction as lp




def construct_trees(nodes, graph):
    """use BFS algorithm to construct the BFS-trees

    Args:
        nodes: the list of nodes in the graph
    Returns:
        trees: dict, root_node_id -> tree, where tree is a dict: node_id -> list: [father, child_0, child_1, ...]
    """

    trees = {}
    for root in tqdm.tqdm(nodes):
        trees[root] = {}
        trees[root][root] = [root]
        used_nodes = set()
        queue = collections.deque([root])
        while len(queue) > 0:
            cur_node = queue.popleft()
            used_nodes.add(cur_node)
            for sub_node in graph[cur_node]:
                if sub_node not in used_nodes:
                    trees[root][cur_node].append(sub_node)
                    trees[root][sub_node] = [cur_node]
                    queue.append(sub_node)
                    used_nodes.add(sub_node)

    return trees


def construct_trees_with_mp(nodes, n_node):
    """use the multiprocessing to speed up trees construction

    Args:
        nodes: the list of nodes in the graph
    """

    cores = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(cores)
    new_nodes = []
    n_node_per_core = n_node // cores
    for i in range(cores):
        if i != cores - 1:
            new_nodes.append(nodes[i * n_node_per_core: (i + 1) * n_node_per_core])
        else:
            new_nodes.append(nodes[i * n_node_per_core:])

    trees = {}
    trees_result = pool.map(construct_trees, new_nodes)
    for tree in trees_result:
        trees.update(tree)

    return trees


def sampling(root, tree, sample_num, for_d, all_score): 
    """ sample nodes from BFS-tree

    Args:
        root: int, root node
        tree: dict, BFS-tree
        sample_num: the number of required samples
        for_d: bool, whether the samples are used for the generator or the discriminator
    Returns:
        samples: list, the indices of the sampled nodes
        paths: list, paths from the root to the sampled nodes
    """

    samples = []
    paths = []
    n = 0
    while len(samples) < sample_num:
        current_node = root
        previous_node = -1
        paths.append([])
        is_root = True
        paths[n].append(current_node)

        while True:
            node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
            is_root = False
            if len(node_neighbor) == 0:  # the tree only has a root
                return None, None
            if for_d:  # skip 1-hop nodes (positive samples)
                if node_neighbor == [root]:
                    # in current version, None is returned for simplicity
                    return None, None
                if root in node_neighbor:
                    node_neighbor.remove(root)

            relevance_probability = all_score[current_node, node_neighbor]
            relevance_probability = utils.softmax(relevance_probability.detach().cpu().numpy())
            next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]  # select next node
            paths[n].append(next_node)
            if next_node == previous_node:  # terminating condition
                samples.append(current_node)
                break

            previous_node = current_node
            current_node = next_node

        n = n + 1

    return samples, paths


def get_node_pairs_from_path(path):
    """
    given a path from root to a sampled node, generate all the node pairs within the given windows size
    e.g., path = [1, 0, 2, 4, 2], window_size = 2 -->
    node pairs= [[1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]]
    :param path: a path from root to the sampled node
    :return pairs: a list of node pairs
    """

    path = path[:-1]
    pairs = []
    for i in range(len(path)):
        center_node = path[i]

        for j in range(max(i - config.window_size, 0), min(i + config.window_size + 1, len(path))):
            if i == j:
                continue
            node = path[j]
            pairs.append([center_node, node])

    return pairs


def prepare_data_for_d(root_nodes, graph, trees, all_score):
    """generate positive and negative samples for the discriminator, and record them in the txt file"""

    center_nodes = []
    neighbor_nodes = []
    labels = []
    for i in root_nodes:
        if np.random.rand() < config.update_ratio:
            pos = graph[i]
            neg, _ = sampling(i, trees[i], len(pos), for_d=True, all_score=all_score)
            if len(pos) != 0 and neg is not None:
                # positive samples
                center_nodes.extend([i] * len(pos))
                neighbor_nodes.extend(pos)
                labels.extend([1] * len(pos))

                # negative samples
                center_nodes.extend([i] * len(pos))
                neighbor_nodes.extend(neg)
                labels.extend([0] * len(neg))

    return center_nodes, neighbor_nodes, labels


def prepare_data_for_g(discriminator, root_nodes, trees, all_score):
    """sample nodes for the generator"""

    paths = []

    for i in root_nodes:
        if np.random.rand() < config.update_ratio:
            sample, paths_from_i = sampling(i, trees[i], config.n_sample_gen, for_d=False, all_score=all_score)
            if paths_from_i is not None:
                paths.extend(paths_from_i)

    node_pairs = list(map(get_node_pairs_from_path, paths))
    node_1 = []
    node_2 = []
    for i in range(len(node_pairs)):
        for pair in node_pairs[i]:
            node_1.append(pair[0])
            node_2.append(pair[1])

    score, _, _, _ = discriminator(np.array(node_1), np.array(node_2))
    reward= discriminator.get_reward(score)

    return node_1, node_2, reward


def write_embeddings_to_file(generator, discriminator, n_node):
    """write embeddings of the generator and the discriminator to files"""

    modes = [generator, discriminator]
    for i in range(2):
        embedding_matrix = modes[i].embedding_matrix
        index = np.array(range(n_node)).reshape(-1, 1)
        embedding_matrix = np.hstack([index, embedding_matrix.detach().cpu().numpy()])
        embedding_list = embedding_matrix.tolist()
        embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                            for emb in embedding_list]

        with open(config.emb_filenames[i], "w+") as f:
            lines = [str(n_node) + "\t" + str(config.n_emb) + "\n"] + embedding_str
            f.writelines(lines)


def evaluation(n_node):
    results = []
    if config.app == "link_prediction":
        for i in range(2):
            lpe = lp.LinkPredictEval(
                config.emb_filenames[i], config.test_filename, config.test_neg_filename, n_node, config.n_emb)
            result = lpe.eval_link_prediction()
            results.append(config.modes[i] + ":" + str(result) + "\n")

    with open(config.result_filename, mode="a+") as f:
        f.writelines(results)


def train(generator, discriminator, n_node, graph, root_nodes, trees, device):
    write_embeddings_to_file(generator, discriminator, n_node)
    evaluation(n_node)

    # Initialize all_score, loss
    all_score = torch.matmul(generator.embedding_matrix, generator.embedding_matrix.t()) + generator.bias_vector
    criterion1 = nn.BCEWithLogitsLoss()
    #criterion2 = nn.MSELoss(reduction='sum')

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.lr_dis)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config.lr_gen)


    print("Start Training...")
    for epoch in range(config.n_epochs):
        print("epoch %d" % epoch)

        # save the model
        if epoch > 0 and epoch % config.save_steps == 0:
            torch.save(generator.state_dict(), config.model_log+'generator_checkpoint.pt')
            torch.save(discriminator.state_dict(), config.model_log+'discriminator_checkpoint.pt')

        # D-steps
        center_nodes = []
        neighbor_nodes = []
        labels = []
        for d_epoch in range(config.n_epochs_dis):
            # generate new nodes for the discriminator for every dis_interval iterations
            if d_epoch % config.dis_interval == 0:
                center_nodes, neighbor_nodes, labels = prepare_data_for_d(root_nodes, graph, trees, all_score)

            # training
            train_size = len(center_nodes)
            start_list = list(range(0, train_size, config.batch_size_dis))
            np.random.shuffle(start_list)
            for start in start_list:
                end = start + config.batch_size_dis

                d_optimizer.zero_grad()
                score, node_embedding, node_neighbor_embedding, bias = discriminator(np.array(center_nodes[start:end]),
                np.array(neighbor_nodes[start:end]))
                label = np.array(labels[start:end])
                label = torch.from_numpy(label).type(torch.float64)
                label = label.to(device)

                d_loss = torch.sum(criterion1(score, label)) + config.lambda_dis * (torch.sum(node_embedding**2)/2+
                torch.sum(node_neighbor_embedding**2)/2+torch.sum(bias**2)/2)

                d_loss.backward()
                d_optimizer.step()

            print(f"[Total Epoch {epoch}/{config.n_epochs}] [D Epoch {d_epoch}/{config.n_epochs_dis}] [D loss: {d_loss.item():.4f}]")
            
            if d_epoch == config.n_epochs_dis - 1:
                print(f"Discrimination finished(Epoch {epoch}).")


        # G-steps
        node_1 = []
        node_2 = []
        reward = []
        for g_epoch in range(config.n_epochs_gen):
            all_score = generator.get_all_score()
            if g_epoch % config.gen_interval == 0:
                node_1, node_2, reward = prepare_data_for_g(discriminator, root_nodes, trees, all_score)
                reward = reward.detach() # Prevent the gradient flowing in the discriminator

            # training
            train_size = len(node_1)
            start_list = list(range(0, train_size, config.batch_size_gen))
            np.random.shuffle(start_list)
            for start in start_list:
                end = start + config.batch_size_gen

                g_optimizer.zero_grad()
                node_embedding, node_neighbor_embedding, prob = generator(np.array(node_1[start:end]), np.array(node_2[start:end]))
                reward_p = reward[start:end]

                g_loss = torch.mean(torch.log(prob)*reward_p) + config.lambda_gen * (torch.sum(node_embedding**2)/2+
                torch.sum(node_neighbor_embedding**2)/2)

                g_loss.backward()
                g_optimizer.step()

            print(f"[Total Epoch {epoch}/{config.n_epochs}] [G Epoch {g_epoch}/{config.n_epochs_gen}] [G loss: {g_loss.item():.4f}]")


            if g_epoch == config.n_epochs_gen - 1:
                print(f"Generation finished (Epoch {epoch}).")


        write_embeddings_to_file(generator, discriminator, n_node)
        evaluation(n_node)
    
    print('Training completes')


def main():
    device = f'cuda:{config.cuda}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print('Current cuda device:', torch.cuda.current_device())


    print("reading graphs...")
    n_node, graph = utils.read_edges(config.train_filename, config.test_filename)
    root_nodes = [i for i in range(n_node)]

    print("reading initial embeddings...")
    node_embed_init_d = utils.read_embeddings(filename=config.pretrain_emb_filename_d,
                                                    n_node=n_node,
                                                    n_embed=config.n_emb)
    node_embed_init_g = utils.read_embeddings(filename=config.pretrain_emb_filename_g,
                                                    n_node=n_node,
                                                    n_embed=config.n_emb)

    # construct or read BFS-trees
    trees = None
    if os.path.isfile(config.cache_filename):
        print("reading BFS-trees from cache...")
        pickle_file = open(config.cache_filename, 'rb')
        trees = pickle.load(pickle_file)
        pickle_file.close()
    else:
        print("constructing BFS-trees...")
        pickle_file = open(config.cache_filename, 'wb')
        if config.multi_processing:
            construct_trees_with_mp(root_nodes, n_node)
        else:
            trees = construct_trees(root_nodes, graph)

        pickle.dump(trees, pickle_file)
        pickle_file.close()

    print("building GAN model...")
    discriminator = Discriminator(n_node, node_embed_init_d).to(device)
    generator = Generator(n_node, node_embed_init_g).to(device)

    train(generator, discriminator, n_node, graph, root_nodes, trees, device)
    



if __name__ == '__main__':
    main()