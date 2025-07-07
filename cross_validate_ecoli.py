import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve
import logging
import os
import sys
import json
import time                              
from torch.utils.data import DataLoader
from run import (
    load_ecoli_data, preprocess_data, split_data, GeneRegulationDataset,
    DeepGeneRegulationNetwork, CombinedLoss, DatasetSpecificConfig
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('cross_validation.log')
    ]
)
logger = logging.getLogger(__name__)
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metrics = None
    def __call__(self, metrics):
        if self.mode == 'max':
            score = metrics['auc']  
        else:
            score = -metrics['auc']
        if self.best_score is None:
            self.best_score = score
            self.best_metrics = metrics
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_metrics = metrics
            self.counter = 0
        return self.early_stop
def perform_cross_validation(dataset_num, n_splits=5, epochs=50, patience=7):
    logger.info(f"开始对Ecoli{dataset_num}数据集进行{n_splits}折交叉验证...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    model_dir = f'models_ecoli{dataset_num}'
    os.makedirs(model_dir, exist_ok=True)
    data = load_ecoli_data(dataset_num)
    processed_data = preprocess_data(data)
    config = DatasetSpecificConfig.get_config(dataset_num)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    fold_metrics = []
    fold_models = []
    all_indices = np.arange(len(processed_data['labels']))
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices), 1):
        logger.info(f"\n开始第 {fold} 折验证...")
        train_dataset = GeneRegulationDataset(
            processed_data=processed_data,
            indices=train_idx,
            noise_level=0.05,
            mask_prob=0.05,
            feature_dropout=0.02,
            mixup_alpha=0.1,
            cutmix_prob=0.1,
            scale_factor=0.02,
            adaptive_augmentation=True,
            is_training=True
        )
        val_dataset = GeneRegulationDataset(
            processed_data=processed_data,
            indices=val_idx,
            noise_level=0.0,
            mask_prob=0.0,
            feature_dropout=0.0,
            mixup_alpha=0.0,
            cutmix_prob=0.0,
            scale_factor=0.0,
            adaptive_augmentation=False,
            is_training=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            num_workers=0,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            num_workers=0,
            pin_memory=True
        )
        model = DeepGeneRegulationNetwork(
            num_genes=processed_data['temporal_features'].shape[0],
            temporal_channels=processed_data['temporal_features'].shape[1],
            gene_expression_channels=processed_data['gene_expression_features'].shape[1],
            topology_channels=processed_data['topology_features'].shape[1],
            hidden_channels=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config.get('num_heads', 8),
            dropout=config['dropout']
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.0001)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        criterion = CombinedLoss(
            pos_weight=train_dataset.pos_weight,
            gamma=2.0,
            alpha=0.25
        )
        early_stopping = EarlyStopping(patience=patience)
        best_val_auc = 0
        best_model_state = None
        train_losses = []
        val_losses = []
        train_aucs = []
        val_aucs = []
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            for batch in train_loader:
                optimizer.zero_grad()
                temporal_features = batch['temporal_features'].to(device)
                gene_expression_features = batch['gene_expression_features'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_labels = batch['edge_labels'].to(device)
                topology_features = batch['topology_features'].to(device)
                predictions = model(
                    temporal_features,
                    gene_expression_features,
                    topology_features,
                    edge_index
                )
                loss = criterion(predictions, edge_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_preds.extend(predictions.detach().cpu().numpy())
                train_labels.extend(edge_labels.cpu().numpy())
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    temporal_features = batch['temporal_features'].to(device)
                    gene_expression_features = batch['gene_expression_features'].to(device)
                    edge_index = batch['edge_index'].to(device)
                    edge_labels = batch['edge_labels'].to(device)
                    topology_features = batch['topology_features'].to(device)
                    predictions = model(
                        temporal_features,
                        gene_expression_features,
                        topology_features,
                        edge_index
                    )
                    loss = criterion(predictions, edge_labels)
                    val_loss += loss.item()
                    val_preds.extend(predictions.cpu().numpy())
                    val_labels.extend(edge_labels.cpu().numpy())
            train_auc = roc_auc_score(train_labels, train_preds)
            val_auc = roc_auc_score(val_labels, val_preds)
            scheduler.step(val_auc)
            train_losses.append(train_loss/len(train_loader))
            val_losses.append(val_loss/len(val_loader))
            train_aucs.append(train_auc)
            val_aucs.append(val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict()
                torch.save(model.state_dict(), 
                          os.path.join(model_dir, f'best_model_fold_{fold}.pt'))
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Fold {fold}, Epoch {epoch+1}/{epochs}:")
                logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}, Train AUC: {train_auc:.4f}")
                logger.info(f"Val Loss: {val_loss/len(val_loader):.4f}, Val AUC: {val_auc:.4f}")
                logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            current_metrics = {
                'auc': val_auc,
                'ap': average_precision_score(val_labels, val_preds),
                'f1': f1_score(val_labels, (np.array(val_preds) > 0.5).astype(int))
            }
            if early_stopping(current_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        model.load_state_dict(best_model_state)
        model.eval()
        fold_models.append(model)
        final_preds = []
        final_labels = []
        with torch.no_grad():
            for batch in val_loader:
                temporal_features = batch['temporal_features'].to(device)
                gene_expression_features = batch['gene_expression_features'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_labels = batch['edge_labels'].to(device)
                topology_features = batch['topology_features'].to(device)
                predictions = model(
                    temporal_features,
                    gene_expression_features,
                    topology_features,
                    edge_index
                )
                final_preds.extend(predictions.cpu().numpy())
                final_labels.extend(edge_labels.cpu().numpy())
        final_metrics = {
            'auc': roc_auc_score(final_labels, final_preds),
            'ap': average_precision_score(final_labels, final_preds),
            'f1': f1_score(final_labels, (np.array(final_preds) > 0.5).astype(int))
        }
        fold_results.append({
            'fold': fold,
            'best_val_auc': best_val_auc,
            'final_metrics': final_metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_aucs': train_aucs,
            'val_aucs': val_aucs
        })
        fold_metrics.append(final_metrics)
        logger.info(f"\n第 {fold} 折验证完成:")
        logger.info(f"最佳验证集 AUC: {best_val_auc:.4f}")
        logger.info(f"最终测试指标:")
        for metric, value in final_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
    avg_metrics = {
        metric: np.mean([fold[metric] for fold in fold_metrics])
        for metric in fold_metrics[0].keys()
    }
    std_metrics = {
        metric: np.std([fold[metric] for fold in fold_metrics])
        for metric in fold_metrics[0].keys()
    }
    results = {
        'dataset': f'Ecoli{dataset_num}',
        'n_splits': n_splits,
        'epochs': epochs,
        'fold_results': fold_results,
        'average_metrics': avg_metrics,
        'std_metrics': std_metrics
    }
    output_file = f'cross_validation_results_ecoli{dataset_num}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"\n交叉验证完成，结果已保存到 {output_file}")
    logger.info("\n平均性能指标:")
    for metric, value in avg_metrics.items():
        logger.info(f"{metric}: {value:.4f} ± {std_metrics[metric]:.4f}")
    return results, fold_models
def plot_cross_validation_results(results, output_file):
    plt.figure(figsize=(12, 6))
    metrics = list(results['average_metrics'].keys())
    avg_values = [results['average_metrics'][m] for m in metrics]
    std_values = [results['std_metrics'][m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x, avg_values, width, yerr=std_values, capsize=5)
    plt.xlabel('评估指标')
    plt.ylabel('得分')
    plt.title(f"{results['dataset']} 交叉验证结果")
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    for i, v in enumerate(avg_values):
        plt.text(i, v + 0.02, f'{v:.3f}±{std_values[i]:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    logger.info(f"交叉验证结果图已保存为 {output_file}")
def plot_learning_curves(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    for fold in range(len(results['fold_results'])):
        train_losses = results['fold_results'][fold]['train_losses']
        val_losses = results['fold_results'][fold]['val_losses']
        plt.plot(train_losses, label=f'Train Fold {fold+1}')
        plt.plot(val_losses, label=f'Val Fold {fold+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves - Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'learning_curves_loss.png'))
    plt.close()
    plt.figure(figsize=(12, 6))
    for fold in range(len(results['fold_results'])):
        train_aucs = results['fold_results'][fold]['train_aucs']
        val_aucs = results['fold_results'][fold]['val_aucs']
        plt.plot(train_aucs, label=f'Train Fold {fold+1}')
        plt.plot(val_aucs, label=f'Val Fold {fold+1}')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Learning Curves - AUC')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'learning_curves_auc.png'))
    plt.close()
def plot_roc_curves(results, fold_models, val_loader, device, output_dir):
    plt.figure(figsize=(10, 8))
    all_predictions = []
    all_labels = []
    for model in fold_models:
        model.eval()
        fold_preds = []
        fold_labels = []
        with torch.no_grad():
            for batch in val_loader:
                temporal_features = batch['temporal_features'].to(device)
                gene_expression_features = batch['gene_expression_features'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_labels = batch['edge_labels'].to(device)
                topology_features = batch['topology_features'].to(device)
                predictions = model(
                    temporal_features,
                    gene_expression_features,
                    topology_features,
                    edge_index
                )
                fold_preds.extend(predictions.cpu().numpy())
                fold_labels.extend(edge_labels.cpu().numpy())
        all_predictions.append(fold_preds)
        all_labels.append(fold_labels)
    mean_tpr = np.zeros(100)
    mean_fpr = np.linspace(0, 1, 100)
    for i, (preds, labels) in enumerate(zip(all_predictions, all_labels)):
        fpr, tpr, _ = roc_curve(labels, preds)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        plt.plot(fpr, tpr, alpha=0.3, label=f'Fold {i+1}')
    mean_tpr /= len(fold_models)
    plt.plot(mean_fpr, mean_tpr, 'b-', label='Mean ROC')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()
def plot_pr_curves(results, fold_models, val_loader, device, output_dir):
    plt.figure(figsize=(10, 8))
    all_predictions = []
    all_labels = []
    for model in fold_models:
        model.eval()
        fold_preds = []
        fold_labels = []
        with torch.no_grad():
            for batch in val_loader:
                temporal_features = batch['temporal_features'].to(device)
                gene_expression_features = batch['gene_expression_features'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_labels = batch['edge_labels'].to(device)
                topology_features = batch['topology_features'].to(device)
                predictions = model(
                    temporal_features,
                    gene_expression_features,
                    topology_features,
                    edge_index
                )
                fold_preds.extend(predictions.cpu().numpy())
                fold_labels.extend(edge_labels.cpu().numpy())
        all_predictions.append(fold_preds)
        all_labels.append(fold_labels)
    mean_precision = np.zeros(100)
    mean_recall = np.linspace(0, 1, 100)
    for i, (preds, labels) in enumerate(zip(all_predictions, all_labels)):
        precision, recall, _ = precision_recall_curve(labels, preds)
        mean_precision += np.interp(mean_recall, recall[::-1], precision[::-1])
        plt.plot(recall, precision, alpha=0.3, label=f'Fold {i+1}')
    mean_precision /= len(fold_models)
    plt.plot(mean_recall, mean_precision, 'b-', label='Mean PR')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'))
    plt.close()
def plot_confusion_matrix(results, fold_models, val_loader, device, output_dir):
    plt.figure(figsize=(10, 8))
    all_predictions = []
    all_labels = []
    for model in fold_models:
        model.eval()
        fold_preds = []
        fold_labels = []
        with torch.no_grad():
            for batch in val_loader:
                temporal_features = batch['temporal_features'].to(device)
                gene_expression_features = batch['gene_expression_features'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_labels = batch['edge_labels'].to(device)
                topology_features = batch['topology_features'].to(device)
                predictions = model(
                    temporal_features,
                    gene_expression_features,
                    topology_features,
                    edge_index
                )
                fold_preds.extend((predictions.cpu().numpy() > 0.5).astype(int))
                fold_labels.extend(edge_labels.cpu().numpy())
        all_predictions.extend(fold_preds)
        all_labels.extend(fold_labels)
    cm = confusion_matrix(all_labels, all_predictions)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
def main():
    import argparse
    parser = argparse.ArgumentParser(description='执行基因调控网络的交叉验证')
    parser.add_argument('dataset_num', type=int, choices=range(1, 6),
                      help='数据集编号 (1-5)')
    parser.add_argument('--n_splits', type=int, default=5,
                      help='交叉验证的折数 (默认: 5)')
    parser.add_argument('--epochs', type=int, default=50,
                      help='每折的训练轮数 (默认: 50)')
    parser.add_argument('--patience', type=int, default=7,
                      help='早停耐心值 (默认: 7)')
    args = parser.parse_args()
    output_dir = f'results_ecoli{args.dataset_num}'
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_ecoli_data(args.dataset_num)
    processed_data = preprocess_data(data)
    val_dataset = GeneRegulationDataset(
        processed_data=processed_data,
        indices=np.arange(len(processed_data['labels'])),
        noise_level=0.0,
        mask_prob=0.0,
        feature_dropout=0.0,
        mixup_alpha=0.0,
        cutmix_prob=0.0,
        scale_factor=0.0,
        adaptive_augmentation=False,
        is_training=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=0,
        pin_memory=True
    )
    results, fold_models = perform_cross_validation(
        dataset_num=args.dataset_num,
        n_splits=args.n_splits,
        epochs=args.epochs,
        patience=args.patience
    )
    plot_cross_validation_results(
        results,
        os.path.join(output_dir, 'cross_validation_results.png')
    )
    plot_learning_curves(results, output_dir)
    plot_roc_curves(results, fold_models, val_loader, device, output_dir)
    plot_pr_curves(results, fold_models, val_loader, device, output_dir)
    plot_confusion_matrix(results, fold_models, val_loader, device, output_dir)
    logger.info(f"所有结果已保存到 {output_dir} 目录")
if __name__ == '__main__':
    main() 