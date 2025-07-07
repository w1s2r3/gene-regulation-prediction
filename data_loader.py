import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import logging
from torch.utils.data import Dataset, DataLoader
import os
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
def load_dream_data(dataset_name):
    logger = logging.getLogger(__name__)
    base_path = f"YOUR_PATH_HERE"
    gold_standard_path = f"YOUR_PATH_HERE"
    print(f"Base path: {base_path}")
    print(f"Gold standard path: {gold_standard_path}")
    if not os.path.exists(os.path.dirname(gold_standard_path)):
        logger.warning(f"Gold standard path does not exist: {os.path.dirname(gold_standard_path)}")
        logger.warning("Creating directory structure for testing...")
        os.makedirs(os.path.dirname(gold_standard_path), exist_ok=True)
    logger.info(f"Loading dataset: Ecoli{dataset_name}")
    time_series_file = f'{base_path}/insilico_size100_{dataset_name}_timeseries.tsv'
    logger.info(f"Loading time series data from: {time_series_file}")
    if not os.path.exists(time_series_file):
        logger.warning(f"Time series data file not found: {time_series_file}")
        logger.info("Creating simulated time series data for testing...")
        time_series_data = pd.DataFrame(np.random.randn(21, 101))  
        time_series_data.iloc[:, 0] = np.arange(21)  
    else:
        time_series_data = pd.read_csv(time_series_file, sep='\t')
    logger.info(f"Time series data shape: {time_series_data.shape}")
    print(f"Time series data columns: {time_series_data.columns}")
    print(f"YOUR_PATH_HERE")
    gold_standard_file = gold_standard_path
    logger.info(f"Loading gold standard data from: {gold_standard_file}")
    if not os.path.exists(gold_standard_file):
        logger.warning(f"Gold standard data file not found: {gold_standard_file}")
        logger.info("Creating simulated gold standard data for testing...")
        sources = []
        targets = []
        interactions = []
        for _ in range(100):
            src = np.random.randint(0, 100)
            tgt = np.random.randint(0, 100)
            interaction = np.random.randint(0, 2)
            sources.append(src)
            targets.append(tgt)
            interactions.append(interaction)
        gold_standard_data = pd.DataFrame({
            'source': sources,
            'target': targets,
            'interaction': interactions
        })
        os.makedirs(os.path.dirname(gold_standard_file), exist_ok=True)
        gold_standard_data.to_csv(gold_standard_file, sep='\t', index=False)
    else:
        gold_standard_data = pd.read_csv(gold_standard_file, sep='\t')
    logger.info(f"Gold standard data shape: {gold_standard_data.shape}")
    print(f"Gold standard data columns: {gold_standard_data.columns}")
    print(f"YOUR_PATH_HERE")
    wildtype_file = f'{base_path}/insilico_size100_{dataset_name}_wildtype.tsv'
    logger.info(f"Loading wildtype data from: {wildtype_file}")
    if not os.path.exists(wildtype_file):
        logger.warning(f"Wildtype data file not found: {wildtype_file}")
        logger.info("Creating simulated wildtype data for testing...")
        wildtype_data = pd.DataFrame(np.random.randn(1, 100))
    else:
        wildtype_data = pd.read_csv(wildtype_file, sep='\t')
    logger.info(f"Wildtype data shape: {wildtype_data.shape}")
    print(f"Wildtype data columns: {wildtype_data.columns}")
    print(f"YOUR_PATH_HERE")
    multifactorial_file = f'{base_path}/insilico_size100_{dataset_name}_multifactorial.tsv'
    logger.info(f"Loading multifactorial data from: {multifactorial_file}")
    if not os.path.exists(multifactorial_file):
        logger.warning(f"Multifactorial data file not found: {multifactorial_file}")
        logger.info("Creating simulated multifactorial data for testing...")
        multifactorial_data = pd.DataFrame(np.random.randn(1, 100))
    else:
        multifactorial_data = pd.read_csv(multifactorial_file, sep='\t')
    logger.info(f"Multifactorial data shape: {multifactorial_data.shape}")
    print(f"Multifactorial data columns: {multifactorial_data.columns}")
    print(f"YOUR_PATH_HERE")
    return time_series_data, gold_standard_data, wildtype_data, multifactorial_data
def preprocess_data(time_series_data, gold_standard_data, wildtype_data, multifactorial_data):
    print("\n开始预处理数据...")
    print("提取时序特征...")
    temporal_features = extract_temporal_features(time_series_data)
    print(f"时序特征形状: {temporal_features.shape}")
    print("提取静态特征...")
    gene_expression_features = extract_gene_expression_features(wildtype_data, multifactorial_data)
    print(f"静态特征形状: {gene_expression_features.shape}")
    num_genes = temporal_features.shape[0]
    labels = torch.zeros((num_genes, num_genes))
    print("\n金标准数据信息:")
    print(f"列名: {gold_standard_data.columns}")
    interaction_col = None
    possible_names = ['interaction', 'label', 'weight', 'value', '1']
    for name in possible_names:
        if name in gold_standard_data.columns:
            interaction_col = name
            break
    if interaction_col is None:
        interaction_col = gold_standard_data.columns[-1]
        print(f"使用最后一列 '{interaction_col}' 作为互作关系列")
    gene_to_idx = {f'G{i+1}': i for i in range(num_genes)}
    positive_edges = gold_standard_data[gold_standard_data[interaction_col] > 0]
    print(f"正样本数量: {len(positive_edges)}")
    positive_indices = []
    for _, row in positive_edges.iterrows():
        src = gene_to_idx[row['G1']]
        tgt = gene_to_idx[row['G2']]
        positive_indices.append((src, tgt))
        labels[src, tgt] = 1  
    positive_indices = np.array(positive_indices)
    print(f"正样本索引形状: {positive_indices.shape}")
    edge_index = torch.tensor(positive_indices.T, dtype=torch.long)
    num_positive = len(positive_indices)
    num_negative = num_positive  
    negative_edges = []
    while len(negative_edges) < num_negative:
        src = np.random.randint(0, num_genes)
        tgt = np.random.randint(0, num_genes)
        if (src, tgt) not in positive_indices and src != tgt:
            negative_edges.append((src, tgt))
            labels[src, tgt] = 0  
    edge_index = torch.tensor(np.concatenate([positive_indices, negative_edges], axis=0).T, dtype=torch.long)
    print(f"边索引形状: {edge_index.shape}")
    edge_labels = torch.zeros(edge_index.size(1))
    for i in range(edge_index.size(1)):
        src, tgt = edge_index[:, i]
        edge_labels[i] = labels[src, tgt]
    print(f"边标签形状: {edge_labels.shape}")
    print("\n提取拓扑特征...")
    topology_features = extract_topology_features(labels)
    print(f"拓扑特征形状: {topology_features.shape}")
        temporal_features = torch.FloatTensor(temporal_features)
    gene_expression_features = torch.FloatTensor(gene_expression_features)
    topology_features = torch.FloatTensor(topology_features)
    print("\n数据预处理完成")
    return temporal_features, gene_expression_features, edge_labels, edge_index, topology_features
def extract_temporal_features(time_series_data):
    print("Extracting temporal features...")
    print("Time series data shape:", time_series_data.shape)
    print("Time series data columns:", time_series_data.columns)
    n_genes = time_series_data.shape[1] - 1
    print(f"Number of genes: {n_genes}")
    n_timepoints = len(time_series_data)
    print(f"Number of timepoints: {n_timepoints}")
    temporal_features = np.zeros((n_genes, n_timepoints))
    for i in range(n_genes):
        gene_name = f'G{i+1}'
        if gene_name in time_series_data.columns:
            temporal_features[i] = time_series_data[gene_name].values
        else:
            print(f"Warning: Gene {gene_name} not found in columns")
    print("Temporal features shape:", temporal_features.shape)
    return temporal_features
def extract_gene_expression_features(wildtype_data, multifactorial_data):
    print("Extracting gene expression features...")
    print("Wildtype data shape:", wildtype_data.shape)
    print("Multifactorial data shape:", multifactorial_data.shape)
    n_genes = wildtype_data.shape[1]
    print(f"Number of genes: {n_genes}")
    gene_expression_features = np.zeros((n_genes, 2))  
    for i in range(n_genes):
        gene_name = f'G{i+1}'
        if gene_name in wildtype_data.columns:
            gene_expression_features[i, 0] = wildtype_data[gene_name].values[0]
        else:
            print(f"Warning: Gene {gene_name} not found in wildtype data")
    for i in range(n_genes):
        gene_name = f'G{i+1}'
        if gene_name in multifactorial_data.columns:
            gene_expression_features[i, 1] = multifactorial_data[gene_name].mean()
        else:
            print(f"Warning: Gene {gene_name} not found in multifactorial data")
    print("GeneExpression features shape:", gene_expression_features.shape)
    return gene_expression_features
def extract_topology_features(labels):
    print("Extracting topology features...")
    print("Labels shape:", labels.shape)
    n_genes = labels.shape[0]
    topology_channels = 15
    topology_features = np.zeros((n_genes, topology_channels))
    degree = labels.sum(axis=1)
    topology_features[:, 0] = degree
    in_degree = labels.sum(axis=0)
    topology_features[:, 1] = in_degree
    out_degree = labels.sum(axis=1)
    topology_features[:, 2] = out_degree
    for i in range(n_genes):
        neighbors = np.nonzero(labels[i].numpy())[0]
        if len(neighbors) > 1:
            num_triangles = 0
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    if labels[neighbors[j], neighbors[k]] == 1:
                        num_triangles += 1
            topology_features[i, 3] = 2 * num_triangles / (len(neighbors) * (len(neighbors) - 1))
    print("Topology features shape:", topology_features.shape)
    return topology_features
class GeneRegulationDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.load_data()
    def load_data(self):
        expression_file = os.path.join(self.data_dir, f'{self.split}_expression.csv')
        self.expression_data = pd.read_csv(expression_file, index_col=0)
        gene_expression_file = os.path.join(self.data_dir, f'{self.split}_gene_expression.csv')
        self.gene_expression_data = pd.read_csv(gene_expression_file, index_col=0)
        edge_file = os.path.join(self.data_dir, f'{self.split}_edges.csv')
        self.edge_data = pd.read_csv(edge_file)
        self.expression_tensor = torch.FloatTensor(self.expression_data.values)
        self.gene_expression_tensor = torch.FloatTensor(self.gene_expression_data.values)
        self.edge_index = torch.LongTensor(self.edge_data[['source', 'target']].values.T)
        self.edge_labels = torch.FloatTensor(self.edge_data['label'].values)
    def __len__(self):
        return len(self.edge_data)
    def __getitem__(self, idx):
        return {
            'expression': self.expression_tensor,
            'gene_expression': self.gene_expression_tensor,
            'edge_index': self.edge_index,
            'edge_label': self.edge_labels[idx]
        }
def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    train_dataset = GeneRegulationDataset(data_dir, split='train')
    val_dataset = GeneRegulationDataset(data_dir, split='val')
    test_dataset = GeneRegulationDataset(data_dir, split='test')
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader
def prepare_sample_data():
    data_dir = 'sample_data'
    os.makedirs(data_dir, exist_ok=True)
    num_nodes = 100
    num_timepoints = 21
    num_gene_expression_features = 2
    expression_data = np.random.randn(num_nodes, num_timepoints)
    gene_expression_data = np.random.randn(num_nodes, num_gene_expression_features)
    num_edges = 200
    edges = np.random.randint(0, num_nodes, (num_edges, 2))
    labels = np.random.randint(0, 2, num_edges)
    for split in ['train', 'val', 'test']:
        pd.DataFrame(expression_data).to_csv(
            os.path.join(data_dir, f'{split}_expression.csv')
        )
        pd.DataFrame(gene_expression_data).to_csv(
            os.path.join(data_dir, f'{split}_gene_expression.csv')
        )
        pd.DataFrame({
            'source': edges[:, 0],
            'target': edges[:, 1],
            'label': labels
        }).to_csv(os.path.join(data_dir, f'{split}_edges.csv'))
    return data_dir
def create_dataset(data_path):
    print("\n开始创建数据集...")
    print(f"数据路径: {data_path}")
    print("\n加载数据...")
    time_series_data, gold_standard_data, wildtype_data, multifactorial_data = load_dream_data('1')
    print("\n数据加载完成:")
    print(f"时间序列数据形状: {time_series_data.shape}")
    print(f"金标准数据形状: {gold_standard_data.shape}")
    print(f"野生型数据形状: {wildtype_data.shape}")
    print(f"多因素数据形状: {multifactorial_data.shape}")
    print("\n开始预处理数据...")
    temporal_features, gene_expression_features, edge_labels, edge_index, topology_features = preprocess_data(
        time_series_data, gold_standard_data, wildtype_data, multifactorial_data
    )
    print("\n数据预处理完成:")
    print(f"时序特征形状: {temporal_features.shape}")
    print(f"静态特征形状: {gene_expression_features.shape}")
    print(f"边标签形状: {edge_labels.shape}")
    print(f"边索引形状: {edge_index.shape}")
    print(f"拓扑特征形状: {topology_features.shape}")
    print("\n创建数据集...")
    dataset = GeneInteractionDataset(
        temporal_features=temporal_features,
        gene_expression_features=gene_expression_features,
        edge_index=edge_index,
        edge_labels=edge_labels,
        topology_features=topology_features
    )
    print(f"\n数据集大小: {len(dataset)}")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    print("\n划分数据集:")
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {val_size}")
    print(f"测试集大小: {test_size}")
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    print("\n数据集创建完成!")
    return train_dataset, val_dataset, test_dataset
class GeneInteractionDataset(Dataset):
    def __init__(self, temporal_features, gene_expression_features, edge_index, edge_labels, topology_features):
        self.temporal_features = temporal_features
        self.gene_expression_features = gene_expression_features
        self.edge_index = edge_index
        self.edge_labels = edge_labels
        self.topology_features = topology_features
        print(f"\n数据集信息:")
        print(f"时序特征形状: {temporal_features.shape}")
        print(f"静态特征形状: {gene_expression_features.shape}")
        print(f"边索引形状: {edge_index.shape}")
        print(f"边标签形状: {edge_labels.shape}")
        print(f"拓扑特征形状: {topology_features.shape}")
        assert temporal_features.shape[0] == gene_expression_features.shape[0] == topology_features.shape[0], \
            "基因数量在不同特征之间不一致"
        assert edge_index.shape[1] == edge_labels.shape[0], \
            "边的数量与标签数量不一致"
    def __len__(self):
        return self.edge_labels.shape[0]
    def __getitem__(self, idx):
        return {
            'expression': self.temporal_features,  
            'gene_expression': self.gene_expression_features,  
            'topology': self.topology_features,  
            'edge_index': self.edge_index,  
            'edge_label': self.edge_labels[idx]  
        } 