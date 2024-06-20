import math, operator, pdb 
import dgl, torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import dgl.data
from dgl.data import AmazonCoBuy, Coauthor
from dgl.nodeflow import NodeFlow

import sys, os, random
sys.path.append(os.getcwd())
from gcc.datasets import data_util





####################################################################################################################################
def Inject_backdoor(raw_graph):
    """ e.g. DGLGraph(
                num_nodes=137969, num_edges=739384,
                ndata_schemes={}
                edata_schemes={}
            )
    """
    _graph_nodes_num = raw_graph.number_of_nodes()
    _graph_edges_num = raw_graph.number_of_edges()
    netx = raw_graph.to_networkx()
    raw_graph.readonly(False)
    subgraph_list      = []
    subgraph_nums      = 1000
    inject_frac        = 0.05
    trigger_scale      = 0.2
    prob4method        = 0.8
    nodes_num_each_sub = int(_graph_nodes_num / subgraph_nums) # 130000 / 1000
    trigger_size       = int(trigger_scale * nodes_num_each_sub) # 130 * 0.2
    
    assert prob4method > np.log(trigger_size) / trigger_size
    G_generated = nx.erdos_renyi_graph(trigger_size, prob4method)
    nx.write_edgelist(
        G_generated, 
        'gcc/datasets/tmp_data/subgraph_gen/ER_' + 
        'TrigSz_' + str(trigger_size) + '_Prob_' + str(prob4method) + 
        '.edgelist'
    )
    # test_graph_file      = open(
    #     'gcc/datasets/tmp_data/test_graphs/ER_' + str(inject_frac) + 
    #     '_TrigSz_' + str(trigger_size) + '_Prob_' + str(prob4method) + 
    #     '.backdoor_graphs',
    #     'w'
    # )
    train_graph_file     = open(
        'gcc/datasets/tmp_data/backdoor_graphs/ER_' + str(inject_frac) + 
        '_TrigSz_' + str(trigger_size) + '_Prob_' + str(prob4method) + 
        '.backdoor_graphs',
        'w'
    )
    train_graph_nodefile = open(
        'gcc/datasets/tmp_data/backdoor_graphs/ER_' + str(inject_frac) + 
        '_TrigSz_' + str(trigger_size) + '_Prob_' + str(prob4method) + 
        '.backdoor_graphnodes',
        'w'
    )
    num_backdoor_graphs = int(subgraph_nums * inject_frac) # 1000 * 0.002 = 2
    print('[Numbers of Backdoor subGraphs (%d %%): ' % (inject_frac * 100), num_backdoor_graphs, "]")
    # pdb.set_trace()
    
    # G.nodes 返回 nodeview 类型的迭代器
    subgraph_list = [ \
        list(netx)[x : x + nodes_num_each_sub] \
        for x in range(0, len(netx.nodes), nodes_num_each_sub) \
    ]
    # enumerate
    subgraph_dict = {}
    subgraph_idx  = []
    for i in range(len(subgraph_list)):
        subgraph_dict[i] = subgraph_list[i]
        subgraph_idx.append(i)     
    # pdb.set_trace()    
    
    # 1000个挑两个
    random_backdoor_idx = random.sample(
        subgraph_idx,
        k=num_backdoor_graphs
    )
    train_graph_file.write(" ".join(str(idx) for idx in random_backdoor_idx))
    train_graph_file.close()
    
    edges = [list(_pair) for _pair in netx.edges()]
    edges.extend([[i, j] for j, i in edges]) 
    # print(edges)
    old_edge_mat = torch.LongTensor(np.asarray(edges).transpose())
    # print(old_edge_mat)
    
    for idx in random_backdoor_idx:
        if trigger_size >= len(subgraph_dict[idx]):
            random_select_nodes = np.random.choice(subgraph_dict[idx], trigger_size)
        else:
            random_select_nodes = np.random.choice(subgraph_dict[idx], trigger_size, replace=False)
        train_graph_nodefile.write(" ".join(str(idx) for idx in random_select_nodes))
        train_graph_nodefile.write("\n")
        
        for i in random_select_nodes:
            for j in random_select_nodes:
                if [i, j] in edges:
                    edges.remove([i, j])
                    # raw_graph.remove_edge(i, j)
                    raw_graph.remove_edges([i, j])
        
        for e in G_generated.edges: 
            # ------ the indexes in e begin with 0 -------- #
            # the idxex in random are the same as raw graph #
            edges.append([random_select_nodes[e[0]], random_select_nodes[e[1]]])
            edges.append([random_select_nodes[e[1]], random_select_nodes[e[0]]])
            netx.add_edge(e[0], e[1])
            raw_graph.add_edge(e[0], e[1])
    
    train_graph_nodefile.close()    
        
    # plt.figure(figsize=(20,6))
    # plt.title("Networkx",fontsize=20)
    # nx.draw(netx, with_labels=True)
    # plt.savefig("a.png")
    
    # backdoor_graph = dgl.DGLGraph()
    # backdoor_graph.from_networkx(netx)
    backdoor_graph = raw_graph
    
    netx_ = raw_graph.to_networkx()
    raw_graph.readonly(False)
    edges_ = [list(_pair) for _pair in netx_.edges()]
    edges_.extend([[i, j] for j, i in edges_]) 
    cur_edge_mat = torch.LongTensor(np.asarray(edges_).transpose())
    # print(old_edge_mat.size, cur_edge_mat.size)
    return backdoor_graph


def inj():
    graphs, _ = dgl.data.utils.load_graphs("./data/small.bin",)
    nodes_how_many = [g.number_of_nodes() for g in graphs]
    length = nodes_how_many[np.argmin(nodes_how_many)]
    for g in graphs:
        if length == g.number_of_nodes():
            print(length)
            graph = Inject_backdoor(g)
            break


####################################################################################################################################
def test_inject_trigger(raw_graph, graph_labels):
    # .......................... same as train ............................. # TODO:
    _graph_nodes_num   = 137969
    subgraph_nums      = 1000
    nodes_num_each_g   = int(_graph_nodes_num / subgraph_nums) # 137969 / 1000
    
    inject_frac        = 0.05
    trigger_scale      = 0.2
    prob4method        = 0.8

    trigger_size       = int(trigger_scale * nodes_num_each_g) # 138 * 0.2
    # .......................................................................
    G_generated = nx.read_edgelist(
        'gcc/datasets/tmp_data/subgraph_gen/ER_' + 
        'TrigSz_' + str(trigger_size) + 
        '_Prob_'  + str(prob4method)  + 
        '.edgelist'
    )
    edges_trigger = G_generated.edges()
    # print(list(G_generated.edges()))
    
    # - - - - - - - - test graph with index - - - - - - - - - 
    test_graph_file = open(        
        'gcc/datasets/tmp_data/test_graphs/ER_' + str(inject_frac) + 
        '_TrigSz_' + str(trigger_size) + 
        '_Prob_'   + str(prob4method)  + 
        '.backdoor_graphs', "w"
    )
    test_graph_num      = len(raw_graph)
    num_backdoor_graphs = int(test_graph_num * inject_frac) # 5000 * 0.05
    print('[Numbers of Backdoor testGraphs (%d %%): ' % (inject_frac * 100), num_backdoor_graphs, "]")
    
    # --------------- if target label = 0 ----------------- # TODO:
    target_label = 0
    test_graphs_with_target_label_indexes    = []
    test_graphs_without_target_label_indexes = []

    for graph_idx in range(len(graph_labels)):
        # 不是目标类即需要注入trigger构成所谓测试集
        if graph_labels[graph_idx] != target_label:   
            test_graphs_without_target_label_indexes.append(graph_idx)
        else:
            test_graphs_with_target_label_indexes.append(graph_idx)
    print(
        ' +--Numbers of Test Graphs (target label):',
        len(test_graphs_with_target_label_indexes), '\n'
        ' +--Numbers of Test Graphs (other labels):',
        len(test_graphs_without_target_label_indexes)
    )
    # print(test_graphs_without_target_label_indexes)
    random_test_backdoor_idx = random.sample(
        test_graphs_without_target_label_indexes,
        int(len(test_graphs_without_target_label_indexes) * inject_frac)
        #k=num_backdoor_graphs
    )
    print(int(len(test_graphs_without_target_label_indexes) * inject_frac))
    test_graph_file.write(" ".join(str(idx) for idx in random_test_backdoor_idx))
    test_graph_file.close()
    # pdb.set_trace() 
       
    # -- 可以对测试图中所有不是目标类的图注入 trigger -- #
    for idx in random_test_backdoor_idx:
        # print(idx)
        num_nodes = len(list(raw_graph[idx].nodes()))
        
        if trigger_size >= num_nodes:
            rand_select_nodes = np.random.choice(num_nodes, trigger_size)
        else:
            rand_select_nodes = np.random.choice(num_nodes, trigger_size, replace=False)
            
        # print(num_nodes, rand_select_nodes)
        
        netx_each_graph = raw_graph[idx].to_networkx()
        
        old_edges = [list(_pair) for _pair in netx_each_graph.edges()]
        old_edges.extend([[i, j] for j, i in old_edges]) 
        now_edges = []
        for i, j in old_edges:
            if [i, j] not in now_edges:
                now_edges.append([i, j])
                     
        # pdb.set_trace()
        
        #---- 去除 trigger 在图中的原有边（i, j 同时在旧的边集合中） ----#
        for i in rand_select_nodes:
            for j in rand_select_nodes:
                if [i, j] in now_edges:
                    now_edges.remove([i, j])
                    raw_graph[idx].remove_edges([i, j])
        
        # pdb.set_trace()
        #---- 把 generator 生成的 trigger 的边放在图中对应的位置 ----#
        # ------- generaotr 的边的关系可以和训练图共用 ---------#        
        for e in G_generated.edges: 
            e = list(map(lambda x: int(x), e)) 
            # ------ the indexes in e begin with 0 -------- #
            # the idxex in random are the same as raw graph #
            now_edges.append([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
            now_edges.append([rand_select_nodes[e[1]], rand_select_nodes[e[0]]])
            raw_graph[idx].add_edge(rand_select_nodes[e[0]], rand_select_nodes[e[1]])
            # raw_graph[idx].add_edge(rand_select_nodes[e[1]], rand_select_nodes[e[0]])
        # exit()
        
    ''' 
        return raw_graph has been changed
               graph_label, of test_graphs would be the same as before
    '''
    return raw_graph, graph_labels


def test_inj():
    train_dataset = GraphClassificationDataset(dataset="collab")
    # pdb.set_trace()

####################################################################################################################################











def worker_init_fn(worker_id): #@ modify each copy’s behavior.
    worker_info = torch.utils.data.get_worker_info()
    
    dataset = worker_info.dataset
    # print("---- in work_init_fn() ", dataset, "------") #e.g. <__main__.LoadBalanceGraphDataset object at 0x7fef40d831d0>
    dataset.graphs, _ = dgl.data.utils.load_graphs(
        dataset.dgl_graphs_file, dataset.jobs[worker_id]
    )
    # TODO:
    nodes_how_many = [g.number_of_nodes() for g in dataset.graphs]
    # dataset.length = nodes_how_many[np.argmin(nodes_how_many)] + nodes_how_many[np.argsort(nodes_how_many)[1]]
    # _graphs = []
    # for g in dataset.graphs:
    #     if nodes_how_many[np.argmin(nodes_how_many)] == g.number_of_nodes() \
    #       or \
    #     nodes_how_many[np.argsort(nodes_how_many)[1]] == g.number_of_nodes():
    #         _graphs.append(g)
    # dataset.graphs = _graphs
    dataset.length = nodes_how_many[np.argmin(nodes_how_many)]
    for g in dataset.graphs:
        if dataset.length == g.number_of_nodes():
            dataset.graph = g
            dataset.graph = Inject_backdoor(g)
            print("++++++++++++ Dataset Modified ++++++++++")
            break
    
    # print(dataset.graphs) #e.g. [DGLGraph(num_nodes=4843953, num_edges=85691368, ndata_schemes={} edata_schemes={}), ..., ...]
    # dataset.length = sum([g.number_of_nodes() for g in dataset.graphs]) 
    # print(dataset.length) #e.g. 9832958
    np.random.seed(worker_info.seed % (2 ** 32))


#region------------------------------------------------------
#-------------------- 多数据集 预训练 -------------------
# -----------------------------------------------------
class LoadBalanceGraphDataset(torch.utils.data.IterableDataset): #@ https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    def __init__( self,
        rw_hops=64,                    restart_prob=0.8,                                positional_embedding_size=32,          step_dist=[1.0, 0.0, 0.0],
        num_workers=1,                 dgl_graphs_file = "./data/small.bin",            num_samples=10000,                     num_copies=1,
        graph_transform=None,          aug="rwr",                                       num_neighbors=5,        
    ):
        #X print(dgl_graphs_file)
        #X dgl.data.utils.load_labels(dgl_graphs_file)
        super(LoadBalanceGraphDataset).__init__()
        self.rw_hops = rw_hops
        self.num_neighbors = num_neighbors
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.num_samples = num_samples
        assert sum(step_dist) == 1.0             # 步长和 为 1
        assert positional_embedding_size > 1     # positional_emb 是 > 1 的偶数 
        self.dgl_graphs_file = dgl_graphs_file
        
        graph_sizes = dgl.data.utils.load_labels(dgl_graphs_file)[
            "graph_sizes"
        ].tolist()
        
        #X print(graph_sizes) # [4843953, 3097165, 896305, 540486, 317080, 137969] 共 6个 数据集
        print("load graph done")

        # a simple greedy algorithm for load balance sorted graphs w.r.t its size in decreasing order
        # for each graph, assign it to the worker with least workload
        assert num_workers % num_copies == 0     # worker 的数量需要时 copies 的整倍数
        
        jobs = [list() for i in range(num_workers // num_copies)]
        workloads = [0] * (num_workers // num_copies) # 有几个 [0]
        #X print(workloads)
        graph_sizes = sorted(
            enumerate(graph_sizes), key=operator.itemgetter(1), reverse=True
        ) # 从大到小排列 以便贪心算法 分配工作
        #X print(graph_sizes)
        for idx, size in graph_sizes: # 不理解，无所谓了
            argmin = workloads.index(min(workloads))
            workloads[argmin] += size
            jobs[argmin].append(idx)
            #X print(jobs)     
        self.jobs            = jobs * num_copies
        self.total           = self.num_samples * num_workers
        self.graph_transform = graph_transform
        assert aug in ("rwr", "ns")
        self.aug             = aug
        
        print(dgl.data.utils.load_labels(dgl_graphs_file)) # {'graph_sizes': tensor([4843953, 3097165,  896305,  540486,  317080,  137969])}
        
    def __len__(self):
        return self.num_samples * num_workers # =? self.total

    #! 训练调用
    def __iter__(self): #! 重写 迭代 和 getitem, 以便在 dataloader 的时候调用，先 work init fn 再 iter
        #X print("------------------------------- iter -------------------------------")
        degrees = torch.cat([g.in_degrees().double() ** 0.75 for g in self.graphs])
        # print(degrees) #e.g. tensor([30.1888, 95.4146, 51.9818,  ...,  1.0000,  1.0000,  2.2795], dtype=torch.float64)
        prob = degrees / torch.sum(degrees)
        samples = np.random.choice(
            self.length, size=self.num_samples, replace=True, p=prob.numpy() # 可以取相同数字
        )
        # print(samples) #e.g. [1291266 9038085  600200 ... 9073664 1520671 6893647]
        for idx in samples:
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        # --------------------这样可以找到node idx 对应的 graph idx ----------------------
        graph_idx = 0
        node_idx  = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        # -------------------------------------------------------
        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0] 
        # print(step) #e.g. [0], 从 [0, 3) 中随机选1个数字, 选中第一个的概率为1, 也就是一定会选 0 
        if step == 0:
            other_node_idx = node_idx
        else: # 用不上这里的随机游走呀
            other_node_idx = dgl.contrib.sampling.random_walk(
                g=self.graphs[graph_idx], seeds=[node_idx], num_traces=1, num_hops=step
            )[0][0][-1].item()
        #X print(node_idx, other_node_idx)
        
        if self.aug == "rwr":
            max_nodes_per_seed = max(
                self.rw_hops, int(
                                (self.graphs[graph_idx].in_degree(node_idx) ** 0.75) * math.e 
                                    / (math.e - 1) / self.restart_prob
                                + 0.5
                                ),
            )
            # print(max_nodes_per_seed) #e.g. 256
            traces = dgl.contrib.sampling.random_walk_with_restart( #@ https://github.com/dmlc/dgl/blob/0.4.x/python/dgl/contrib/sampling/randomwalk.py
                self.graphs[graph_idx],
                seeds=[node_idx, other_node_idx],
                restart_prob=self.restart_prob,
                max_nodes_per_seed=max_nodes_per_seed,
            )
            # print("++++++++", traces[0], "\n", traces[1], "++++++++") #e.g. [(tensor([37095, 36985]), tensor([277819]), tensor([37113]), tensor([37016]), tensor([37171]), tensor([37131, 36938, 90537]), tensor([37060]), ...    
        elif self.aug == "ns": #! -------------------------- 测试 -----------------------------
            prob = dgl.backend.tensor([], dgl.backend.float32)
            prob = dgl.backend.zerocopy_to_dgl_ndarray(prob)
            nf1 = dgl.contrib.sampling.sampler._CAPI_NeighborSampling(
                self.graphs[graph_idx]._graph,
                dgl.utils.toindex([node_idx]).todgltensor(),
                0,  # batch_start_id
                1,  # batch_size
                1,  # workers
                self.num_neighbors,  # expand_factor
                self.rw_hops,  # num_hops
                "out",
                False,
                prob,
            )[0]
            nf1 = NodeFlow(self.graphs[graph_idx], nf1)
            trace1 = [nf1.layer_parent_nid(i) for i in range(nf1.num_layers)]
            
            nf2 = dgl.contrib.sampling.sampler._CAPI_NeighborSampling(
                self.graphs[graph_idx]._graph,
                dgl.utils.toindex([other_node_idx]).todgltensor(),0,1,1, self.num_neighbors, self.rw_hops, "out", False, prob,
            )[0]
            nf2 = NodeFlow(self.graphs[graph_idx], nf2)
            trace2 = [nf2.layer_parent_nid(i) for i in range(nf2.num_layers)]
            
            traces = [trace1, trace2]

        # ================ 把随机游走的轨迹 放入 dgl 图中 ======================
        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_k = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
        )
        if self.graph_transform:
            graph_q = self.graph_transform(graph_q)
            graph_k = self.graph_transform(graph_k)
        return graph_q, graph_k


#---------------------------------------------------------------------------------------------------------
#--------------------- 下流任务 ---- Finetune Or GraphClassificationDataset Generate-----------------------
#---------------------------------------------------------------------------------------------------------
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, rw_hops=64, subgraph_size=64, restart_prob=0.8, positional_embedding_size=32, step_dist=[1.0, 0.0, 0.0], ):
        super(GraphDataset).__init__()
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1
        
        graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/lscc_graphs.bin", [0, 1, 2])
        
        for name in ["cs", "physics"]:
            g = Coauthor(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        for name in ["computers", "photo"]:
            g = AmazonCoBuy(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)

        print("GraphDataset HAS LOADED GRAPH")
        self.graphs = graphs
        # 所有的节点数量
        self.length = sum([g.number_of_nodes() for g in self.graphs])

    def __len__(self):
        return self.length
 
    # _convert_idx will be reloaded in GraphClassificationDataset
    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        return graph_idx, node_idx

    # GraphDataset and GraphClassificationDataset use __getitem__ together
    def __getitem__(self, idx):
        graph_idx, node_idx = self._convert_idx(idx)

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.contrib.sampling.random_walk(g=self.graphs[graph_idx], seeds=[node_idx], num_traces=1, num_hops=step)[0][0][-1].item()

        max_nodes_per_seed = max(
            self.rw_hops,
            int((
                self.graphs[graph_idx].out_degree(node_idx) * math.e
                / (math.e - 1) / self.restart_prob
                ) + 0.5
            ), 
        )
        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx, other_node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=max_nodes_per_seed,
        )
        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        graph_k = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        return graph_q, graph_k


class GraphClassificationDataset(GraphDataset):
    def __init__(self, dataset, rw_hops=64, subgraph_size=64, restart_prob=0.8, positional_embedding_size=32, step_dist=[1.0, 0.0, 0.0],):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.entire_graph = True
        assert positional_embedding_size > 1

        self.dataset = data_util.create_graph_classification_dataset(dataset)
        self.graphs = self.dataset.graph_lists

        # pdb.set_trace()
        # print(self.graphs, "in GraphClassificationDataset")
        # e.g.[DGLGraph(num_nodes=72, num_edges=8200,
        # e.g.     ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}
        # e.g.     edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}),
        # e.g. DGLGraph(num_nodes=88, num_edges=4708,
        # e.g.     ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}
        # e.g.     edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})]        
        self.graphs, _ = test_inject_trigger(self.graphs, self.dataset.graph_labels)
            
        self.length = len(self.graphs)
        self.total = self.length
        #* graphs, length reloaded 

    def _convert_idx(self, idx):
        # print("Reloaded _convert_idx in GraphClassificationDataset")
        graph_idx = idx
        # 以该图的所有节点最大出度 作为 该图 node_idx
        node_idx = self.graphs[idx].out_degrees().argmax().item()
        # print(node_idx)
        return graph_idx, node_idx
    
    
# ----------------------------------------------------------------    
# -------------------------- FineTune ----------------------------
# ---------------------------------------------------------------- 
class GraphClassificationDatasetLabeled(GraphClassificationDataset):
    def __init__( self, dataset, rw_hops=64, subgraph_size=64, restart_prob=0.8, positional_embedding_size=32, step_dist=[1.0, 0.0, 0.0], ):
        super(GraphClassificationDatasetLabeled, self).__init__( dataset, rw_hops, subgraph_size, restart_prob, positional_embedding_size, step_dist, )
        self.num_classes = self.dataset.num_labels
        self.entire_graph = True
        self.dict = [self.getitem(idx) for idx in range(len(self))]

    def __getitem__(self, idx):
        return self.dict[idx]

    def getitem(self, idx):
        graph_idx = idx
        node_idx = self.graphs[idx].out_degrees().argmax().item()

        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=self.rw_hops,
        )

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=True,
        )
        return graph_q, self.dataset.graph_labels[graph_idx].item()
#endregion




# =================================================
# ======================= Trojan ===========================
# =================================================
class Load_injected_GraphDataset(torch.utils.data.IterableDataset):
    def __init__( self,
        rw_hops=64,                restart_prob=0.8,  positional_embedding_size = 32,   
        step_dist=[1.0, 0.0, 0.0], num_workers=1,     dgl_graphs_file = "./data/small.bin",   
        num_samples=10000,         num_copies=1,      graph_transform = None,  
        aug="rwr",                 num_neighbors=5,        
    ):
        super(Load_injected_GraphDataset).__init__()
        self.rw_hops       = rw_hops
        self.num_neighbors = num_neighbors
        self.restart_prob  = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist   = step_dist
        self.num_samples = num_samples
        
        assert num_workers % num_copies == 0     # worker 的数量需要时 copies 的整倍数
        assert sum(step_dist) == 1.0             # 步长和 为 1
        assert positional_embedding_size > 1     # positional_emb 是 > 1 的偶数 
        assert aug in ("rwr", "ns")
        
        self.dgl_graphs_file = dgl_graphs_file
        # [4843953, 3097165, 896305, 540486, 317080, 137969] 共 6个 数据集
        graph_sizes          = dgl.data.utils.load_labels(dgl_graphs_file)["graph_sizes"].tolist()
        #Xgraph_size_          = [graph_sizes[np.argmin(graph_sizes)]]
        print("LOAD GRAPH DONE")
        
        jobs = [list() for i in range(num_workers // num_copies)]
        workloads = [0] * (num_workers // num_copies) # 有几个 [0]
        graph_sizes = sorted( 
            enumerate(graph_sizes), key=operator.itemgetter(1), reverse=True 
        ) 
        # 贪心 分配
        for idx, size in graph_sizes:
            argmin = workloads.index(min(workloads))
            workloads[argmin] += size
            jobs[argmin].append(idx)
            
            #pdb.set_trace()
        self.jobs            = jobs * num_copies
        self.total           = self.num_samples * num_workers
        self.graph_transform = graph_transform
        self.aug             = aug
        
    def __len__(self):
        return self.num_samples * num_workers
    
    #! 训练调用
    def __iter__(self):
        # degrees = torch.cat([g.in_degrees().double() ** 0.75 for g in self.graphs])
        # print(degrees) #e.g. tensor([30.1888, 95.4146, 51.9818,  ...,  1.0000,  1.0000,  2.2795], dtype=torch.float64)
        
        # TODO:
        degrees = torch.DoubleTensor(self.graph.in_degrees().double() ** 0.75)
        prob = degrees / torch.sum(degrees)
        # print(prob)    #e.g. tensor([2.3426e-06, 8.9807e-06, ..., 2.3426e-06, 5.3400e-06], dtype=torch.float64)
        samples = np.random.choice(
            self.length, size=self.num_samples, replace=True, p=prob.numpy() # 可以取相同数字
        )
        # print(samples) #e.g. [1291266 9038085  600200 ... 9073664 1520671 6893647]
        for idx in samples:
            yield self.__getitem__(idx)

    # TODO:
    def __getitem__(self, idx):
        graph_idx = False
        node_idx = idx
        # self.step_dist = [0.0, 1.0, 0.0]
        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx # 无步长不游走
        else: 
            other_node_idx = dgl.contrib.sampling.random_walk(
                g=self.graph, seeds=[node_idx], num_traces=1, num_hops=step
            )[0][0][-1].item()
        # print(node_idx, other_node_idx)
        
        if self.aug == "rwr":
            max_nodes_per_seed = max(
                self.rw_hops, 
                int((self.graph.in_degree(node_idx) ** 0.75) 
                    * math.e / (math.e - 1) 
                    / self.restart_prob 
                    + 0.5),
            )
            # print(max_nodes_per_seed) #e.g. 256
            traces = dgl.contrib.sampling.random_walk_with_restart( #@ https://github.com/dmlc/dgl/blob/0.4.x/python/dgl/contrib/sampling/randomwalk.py
                self.graph,
                seeds=[node_idx, other_node_idx],
                restart_prob=self.restart_prob,
                max_nodes_per_seed=max_nodes_per_seed,
            )
            # print("++++++++", traces[0], "\n", traces[1], "++++++++") 
            #e.g. [(tensor([37095, 36985]), tensor([277819]), tensor([37113]), tensor([37016]), tensor([37171]), tensor([37131, 36938, 90537]), tensor([37060]), ...    
        
        # -------------------------- 测试 -----------------------------
        elif self.aug == "ns": 
            prob = dgl.backend.tensor([], dgl.backend.float32)
            prob = dgl.backend.zerocopy_to_dgl_ndarray(prob)
            nf1 = dgl.contrib.sampling.sampler._CAPI_NeighborSampling(
                self.graph._graph,
                dgl.utils.toindex([node_idx]).todgltensor(),
                0,  # batch_start_id
                1,  # batch_size
                1,  # workers
                self.num_neighbors,  # expand_factor
                self.rw_hops,        # num_hops
                "out", False, prob,
            )[0]
            nf1 = NodeFlow(self.graph, nf1)
            trace1 = [nf1.layer_parent_nid(i) for i in range(nf1.num_layers)]
            nf2 = dgl.contrib.sampling.sampler._CAPI_NeighborSampling(
                self.graph._graph,
                dgl.utils.toindex([other_node_idx]).todgltensor(), 
                0, 1, 1, self.num_neighbors, self.rw_hops, 
                "out", False, prob,
            )[0]
            nf2 = NodeFlow(self.graph, nf2)
            trace2 = [nf2.layer_parent_nid(i) for i in range(nf2.num_layers)]
            traces = [trace1, trace2]

        # ================ 把随机游走的轨迹 放入 dgl 图中 ======================
        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graph,
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_k = data_util._rwr_trace_to_dgl_graph(
            g=self.graph,
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
        )
        if self.graph_transform:
            graph_q = self.graph_transform(graph_q)
            graph_k = self.graph_transform(graph_k)
        return graph_q, graph_k



if __name__ == "__main__":
    test_inj()
    exit(0)
    
    # inj()
    # exit(0)
    
    num_workers = 1
    import psutil

    #======================================================================
    mem = psutil.virtual_memory()
    #//print(mem.used / 1024 ** 3)
    
    # graph_dataset = LoadBalanceGraphDataset(
    #     num_workers=num_workers, aug="ns", rw_hops=4, num_neighbors=5
    # )
    graph_dataset = Load_injected_GraphDataset(
        num_workers=num_workers, aug="ns", rw_hops=4, num_neighbors=5
    )
    mem = psutil.virtual_memory()
    #//print(mem.used / 1024 ** 3)
    #======================================================================    
    
    graph_loader = torch.utils.data.DataLoader(
        graph_dataset,
        batch_size=1,
        collate_fn=data_util.batcher(),
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )
    mem = psutil.virtual_memory()
    print(mem.used / 1024 ** 3)

    #======================================================================
    
    for step, batch in enumerate(graph_loader):
        print("bs", batch[0].batch_size)
        print("n = ", batch[0].number_of_nodes())
        print("m = ", batch[0].number_of_edges())
        mem = psutil.virtual_memory()
        print(mem.used / 1024 ** 3)
        #  print(batch.graph_q)
        #  print(batch.graph_q.ndata['pos_directed'])
        print(batch[0].ndata["pos_undirected"])
        
    #======================================================================
    exit(0)