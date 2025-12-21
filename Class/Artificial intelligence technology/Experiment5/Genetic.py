"""
实验五：基于遗传算法的神经网络超参数优化
GA-HPO: Genetic Algorithm for Hyperparameter Optimization

本代码基于实验四的 ResNet 图像分类模型，使用遗传算法自动搜索最优超参数组合。

运行方式:
    python Genetic.py

作者: 软件专业25 - 2025354100103
日期: 2025年12月
"""

# ==================================================================================
#                                   Import Libraries
# ==================================================================================
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import random
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold
import copy
import json
from datetime import datetime
import matplotlib.pyplot as plt

# ==================================================================================
#                                   Config (配置模块)
# ==================================================================================
# -------------------- GPU 配置 --------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置使用的 GPU 编号，多卡用 "0,1,2"

# -------------------- 数据集配置 --------------------
DATASET_DIR = "../Data"  # 数据集根目录
NUM_CLASSES = 11         # 分类类别数

# -------------------- 遗传算法配置 --------------------
GA_CONFIG = {
    'pop_size': 10,            # 种群大小（根据计算资源调整，推荐 10-20）
    'max_generations': 15,      # 最大进化代数（推荐 15-30）
    'crossover_rate': 0.8,      # 交叉概率
    'mutation_rate': 0.15,      # 变异概率
    'elite_size': 2,            # 精英保留数量
    'eta_c': 20,                # SBX 交叉分布指数
    'eta_m': 20,                # 多项式变异分布指数
}

# -------------------- 快速评估配置 --------------------
EVAL_EPOCHS = 5              # 每个个体快速评估的训练轮数
EVAL_PATIENCE = 3            # 快速评估时的早停耐心值

# -------------------- 完整训练配置（用于最终最优超参数） --------------------
FULL_TRAIN_EPOCHS = 300      # 完整训练轮数
FULL_TRAIN_PATIENCE = 20     # 完整训练早停耐心值

# -------------------- 随机种子 --------------------
RANDOM_SEED = 5201314        # 随机种子，确保可复现

# -------------------- Focal Loss 类别权重（来自实验四） --------------------
FOCAL_ALPHA = torch.Tensor([1, 2.3, 0.66, 1, 1.1, 0.75, 2.3, 3.5, 1.1, 0.66, 1.4]).view(-1, 1)

# -------------------- 实验名称 --------------------
EXP_NAME = "GA_HPO_Experiment5"

# ==================================================================================
#                               随机种子设置
# ==================================================================================
def set_seed(seed=RANDOM_SEED):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(5201314)

# ==================================================================================
#                               Image Transforms
# ==================================================================================
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(180),
    transforms.RandomAffine(30),
    transforms.RandomGrayscale(0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(0.2)
])

# ==================================================================================
#                                   Dataset
# ==================================================================================
class FoodDataset(Dataset):
    """食物图像数据集类"""
    
    def __init__(self, path=None, tfm=test_tfm, files=None):
        super(FoodDataset).__init__()
        self.path = path
        if path:
            self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        else:
            self.files = files
        self.transform = tfm
        print(f'Dataset size: {len(self.files)} images')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        try:
            label = int(fname.split(os.sep)[-1].split("_")[0])
        except:
            label = -1
        return im, label


# ==================================================================================
#                               Model Structure (来自实验四)
# ==================================================================================
class Residual_Block(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Classifier(nn.Module):
    """分类器模型 (可配置参数)"""
    def __init__(self, block, num_layers, num_classes=11, dropout1=0.4, dropout2=0.2):
        super(Classifier, self).__init__()
        self.preConv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer0 = self.makeResidualBlocks(block, 32, 64, num_layers[0], stride=2)
        self.layer1 = self.makeResidualBlocks(block, 64, 128, num_layers[1], stride=2)
        self.layer2 = self.makeResidualBlocks(block, 128, 256, num_layers[2], stride=2)
        self.layer3 = self.makeResidualBlocks(block, 256, 512, num_layers[3], stride=2)

        self.fc = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.preConv(x)
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.fc(out.view(out.size(0), -1))
        return out

    def makeResidualBlocks(self, block, in_channels, out_channels, num_layer, stride=1):
        layers = [block(in_channels, out_channels, stride)]
        for i in range(1, num_layer):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)


class FocalLoss(nn.Module):
    """Focal Loss 损失函数"""
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super().__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


# ==================================================================================
#                               遗传算法核心实现
# ==================================================================================
class Individual:
    """
    个体类：表示一组超参数组合
    
    染色体结构 (8个基因位):
    [0] log10(lr)      : 学习率的对数 [-5, -2]
    [1] batch_size_idx : 批大小索引 {0,1,2,3} -> {32,64,128,256}
    [2] num_layer_0    : 第0层残差块数量 [1,4]
    [3] num_layer_1    : 第1层残差块数量 [1,4]
    [4] num_layer_2    : 第2层残差块数量 [1,4]
    [5] num_layer_3    : 第3层残差块数量 [1,4]
    [6] dropout1       : 第一个Dropout率 [0.1, 0.5]
    [7] focal_gamma    : Focal Loss gamma参数 [0.5, 5.0]
    """
    
    # 超参数搜索空间定义 - 针对 RTX 4090 (24GB) 扩展 batch_size
    BATCH_SIZES = [64, 128, 256, 512]
    BOUNDS = {
        'log_lr': (-5, -2),
        'batch_idx': (0, 3),
        'num_layer': (1, 4),
        'dropout': (0.1, 0.5),
        'focal_gamma': (0.5, 5.0)
    }
    
    def __init__(self, chromosome=None):
        if chromosome is None:
            self.chromosome = self._random_init()
        else:
            self.chromosome = list(chromosome)
        self.fitness = None
        self.val_acc = None
    
    def _random_init(self):
        """随机初始化染色体"""
        return [
            np.random.uniform(*self.BOUNDS['log_lr']),      # log10(lr)
            np.random.randint(0, 4),                         # batch_size index
            np.random.randint(1, 5),                         # num_layers[0]
            np.random.randint(1, 5),                         # num_layers[1]
            np.random.randint(1, 5),                         # num_layers[2]
            np.random.randint(1, 5),                         # num_layers[3]
            np.random.uniform(*self.BOUNDS['dropout']),      # dropout1
            np.random.uniform(*self.BOUNDS['focal_gamma']),  # focal_gamma
        ]
    
    def decode(self):
        """将染色体解码为实际超参数字典"""
        return {
            'lr': 10 ** self.chromosome[0],
            'batch_size': self.BATCH_SIZES[int(np.clip(self.chromosome[1], 0, 3))],
            'num_layers': [
                int(np.clip(self.chromosome[2], 1, 4)),
                int(np.clip(self.chromosome[3], 1, 4)),
                int(np.clip(self.chromosome[4], 1, 4)),
                int(np.clip(self.chromosome[5], 1, 4)),
            ],
            'dropout1': np.clip(self.chromosome[6], 0.1, 0.5),
            'dropout2': np.clip(self.chromosome[6] * 0.5, 0.05, 0.25),  # dropout2 = dropout1 / 2
            'focal_gamma': np.clip(self.chromosome[7], 0.5, 5.0)
        }
    
    def __str__(self):
        hp = self.decode()
        return (f"lr={hp['lr']:.2e}, bs={hp['batch_size']}, "
                f"layers={hp['num_layers']}, drop={hp['dropout1']:.2f}, "
                f"gamma={hp['focal_gamma']:.2f}, fitness={self.fitness:.4f}" if self.fitness else "未评估")
    
    def copy(self):
        """深拷贝个体"""
        new_ind = Individual(self.chromosome.copy())
        new_ind.fitness = self.fitness
        new_ind.val_acc = self.val_acc
        return new_ind


class GeneticAlgorithm:
    """
    遗传算法引擎
    
    实现了:
    - 轮盘赌选择 + 精英保留
    - 模拟二进制交叉 (SBX)
    - 多项式变异
    """
    
    def __init__(self, pop_size=10, max_generations=15, 
                 crossover_rate=0.8, mutation_rate=0.15, 
                 elite_size=2, eta_c=20, eta_m=20):
        """
        初始化遗传算法
        
        Args:
            pop_size: 种群大小
            max_generations: 最大进化代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            elite_size: 精英保留数量
            eta_c: SBX交叉分布指数
            eta_m: 多项式变异分布指数
        """
        self.pop_size = pop_size
        self.max_gen = max_generations
        self.pc = crossover_rate
        self.pm = mutation_rate
        self.elite_size = elite_size
        self.eta_c = eta_c
        self.eta_m = eta_m
        
        # 进化历史记录
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_individual': [],
            'generation_time': []
        }
    
    def _roulette_selection(self, population, num_select):
        """轮盘赌选择"""
        fitnesses = np.array([ind.fitness for ind in population])
        # 处理负适应度（虽然本实验中准确率不会为负）
        min_fit = fitnesses.min()
        if min_fit < 0:
            fitnesses = fitnesses - min_fit + 1e-6
        
        # 添加小常数避免除零
        fitnesses = fitnesses + 1e-10
        probs = fitnesses / fitnesses.sum()
        
        selected_indices = np.random.choice(len(population), size=num_select, p=probs)
        return [population[i].copy() for i in selected_indices]
    
    def _tournament_selection(self, population, num_select, tournament_size=3):
        """锦标赛选择（备选方案）"""
        selected = []
        for _ in range(num_select):
            contestants = random.sample(population, min(tournament_size, len(population)))
            winner = max(contestants, key=lambda x: x.fitness)
            selected.append(winner.copy())
        return selected
    
    def _sbx_crossover(self, parent1, parent2):
        """
        模拟二进制交叉 (Simulated Binary Crossover)
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(len(parent1.chromosome)):
            if random.random() < 0.5:  # 每个基因位独立决定是否交叉
                u = random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (self.eta_c + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (self.eta_c + 1))
                
                p1 = parent1.chromosome[i]
                p2 = parent2.chromosome[i]
                
                child1.chromosome[i] = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                child2.chromosome[i] = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        
        child1.fitness = None
        child2.fitness = None
        return child1, child2
    
    def _polynomial_mutation(self, individual):
        """
        多项式变异 (Polynomial Mutation)
        """
        bounds = [
            Individual.BOUNDS['log_lr'],      # 0: log_lr
            (0, 3),                            # 1: batch_idx
            (1, 4),                            # 2-5: num_layers
            (1, 4),
            (1, 4),
            (1, 4),
            Individual.BOUNDS['dropout'],     # 6: dropout
            Individual.BOUNDS['focal_gamma'], # 7: focal_gamma
        ]
        
        for i in range(len(individual.chromosome)):
            if random.random() < self.pm:
                x = individual.chromosome[i]
                xl, xu = bounds[i]
                
                delta = min(x - xl, xu - x) / (xu - xl)
                u = random.random()
                
                if u < 0.5:
                    delta_q = (2 * u + (1 - 2 * u) * (1 - delta) ** (self.eta_m + 1)) ** (1.0 / (self.eta_m + 1)) - 1
                else:
                    delta_q = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta) ** (self.eta_m + 1)) ** (1.0 / (self.eta_m + 1))
                
                x_new = x + delta_q * (xu - xl)
                individual.chromosome[i] = np.clip(x_new, xl, xu)
                
                # 对于离散参数，进行取整
                if i in [1, 2, 3, 4, 5]:
                    individual.chromosome[i] = int(round(individual.chromosome[i]))
        
        individual.fitness = None
        return individual
    
    def evolve(self, evaluator, verbose=True):
        """
        主进化循环
        
        Args:
            evaluator: 评估器对象，需要有 evaluate(hyperparams) 方法
            verbose: 是否打印详细信息
        
        Returns:
            best_individual: 最优个体
            history: 进化历史
        """
        # 初始化种群
        population = [Individual() for _ in range(self.pop_size)]
        
        print("=" * 70)
        print("       遗传算法超参数优化 (GA-HPO) 开始")
        print("=" * 70)
        print(f"种群大小: {self.pop_size}, 最大代数: {self.max_gen}")
        print(f"交叉率: {self.pc}, 变异率: {self.pm}, 精英数: {self.elite_size}")
        print("=" * 70)
        
        for gen in range(self.max_gen):
            gen_start_time = datetime.now()
            
            print(f"\n{'='*20} 第 {gen+1}/{self.max_gen} 代 {'='*20}")
            
            # Step 1: 评估适应度
            for idx, ind in enumerate(population):
                if ind.fitness is None:
                    hp = ind.decode()
                    print(f"  评估个体 {idx+1}/{len(population)}: lr={hp['lr']:.2e}, "
                          f"bs={hp['batch_size']}, layers={hp['num_layers']}")
                    
                    val_acc = evaluator.evaluate(hp)
                    ind.fitness = val_acc
                    ind.val_acc = val_acc
                    
                    print(f"    -> 验证准确率: {val_acc*100:.2f}%")
            
            # 计算统计信息
            fitnesses = [ind.fitness for ind in population]
            best_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            best_ind = max(population, key=lambda x: x.fitness)
            
            self.history['best_fitness'].append(best_fitness)
            self.history['avg_fitness'].append(avg_fitness)
            self.history['best_individual'].append(best_ind.decode())
            self.history['generation_time'].append((datetime.now() - gen_start_time).total_seconds())
            
            print(f"\n  [统计] 最优适应度: {best_fitness*100:.2f}% | "
                  f"平均适应度: {avg_fitness*100:.2f}%")
            print(f"  [最优] {best_ind}")
            
            # 如果是最后一代，直接返回
            if gen == self.max_gen - 1:
                break
            
            # Step 2: 精英保留
            population.sort(key=lambda x: x.fitness, reverse=True)
            elites = [population[i].copy() for i in range(self.elite_size)]
            
            # Step 3: 选择
            num_offspring = self.pop_size - self.elite_size
            selected = self._roulette_selection(population, num_offspring)
            
            # Step 4: 交叉
            offspring = []
            for i in range(0, len(selected) - 1, 2):
                if random.random() < self.pc:
                    child1, child2 = self._sbx_crossover(selected[i], selected[i+1])
                else:
                    child1, child2 = selected[i].copy(), selected[i+1].copy()
                offspring.extend([child1, child2])
            
            # 确保offspring数量正确
            while len(offspring) < num_offspring:
                offspring.append(selected[0].copy())
            offspring = offspring[:num_offspring]
            
            # Step 5: 变异
            offspring = [self._polynomial_mutation(ind) for ind in offspring]
            
            # Step 6: 形成新种群
            population = elites + offspring
        
        # 返回最优个体
        best_individual = max(population, key=lambda x: x.fitness)
        
        print("\n" + "=" * 70)
        print("       遗传算法优化完成!")
        print("=" * 70)
        print(f"最优超参数组合:")
        best_hp = best_individual.decode()
        for k, v in best_hp.items():
            print(f"  {k}: {v}")
        print(f"最优验证准确率: {best_individual.fitness * 100:.2f}%")
        print("=" * 70)
        
        return best_individual, self.history
    
    def plot_evolution(self, save_path=None):
        """绘制进化曲线"""
        plt.figure(figsize=(12, 5))
        
        # 适应度曲线
        plt.subplot(1, 2, 1)
        generations = range(1, len(self.history['best_fitness']) + 1)
        plt.plot(generations, [f*100 for f in self.history['best_fitness']], 
                 'b-o', label='最优适应度', linewidth=2)
        plt.plot(generations, [f*100 for f in self.history['avg_fitness']], 
                 'r--s', label='平均适应度', linewidth=2)
        plt.xlabel('进化代数', fontsize=12)
        plt.ylabel('验证准确率 (%)', fontsize=12)
        plt.title('遗传算法进化曲线', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 学习率演变
        plt.subplot(1, 2, 2)
        lrs = [hp['lr'] for hp in self.history['best_individual']]
        plt.semilogy(generations, lrs, 'g-^', linewidth=2)
        plt.xlabel('进化代数', fontsize=12)
        plt.ylabel('学习率 (log scale)', fontsize=12)
        plt.title('最优个体学习率演变', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"进化曲线已保存至: {save_path}")
        
        plt.show()


# ==================================================================================
#                               神经网络评估器
# ==================================================================================
class NNEvaluator:
    """
    神经网络评估器
    
    负责根据超参数训练模型并返回验证准确率
    """
    
    def __init__(self, train_files, val_files, eval_epochs=5, device=None):
        """
        初始化评估器
        
        Args:
            train_files: 训练集文件路径列表
            val_files: 验证集文件路径列表
            eval_epochs: 快速评估时的训练轮数
            device: 计算设备
        """
        self.train_files = train_files
        self.val_files = val_files
        self.eval_epochs = eval_epochs
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_count = 0
        
        print(f"评估器初始化完成:")
        print(f"  训练集: {len(train_files)} 张图像")
        print(f"  验证集: {len(val_files)} 张图像")
        print(f"  评估轮数: {eval_epochs}")
        print(f"  设备: {self.device}")
    
    def evaluate(self, hyperparams):
        """
        评估超参数组合
        
        Args:
            hyperparams: 超参数字典
        
        Returns:
            验证集准确率 (0-1)
        """
        self.eval_count += 1
        
        try:
            # 创建数据加载器 - num_workers=2 避免文件描述符耗尽
            train_set = FoodDataset(tfm=train_tfm, files=self.train_files)
            train_loader = DataLoader(
                train_set, 
                batch_size=hyperparams['batch_size'],
                shuffle=True, 
                num_workers=2,  # 减少 worker 数避免 "Too many open files"
                pin_memory=True
            )
            
            val_set = FoodDataset(tfm=test_tfm, files=self.val_files)
            val_loader = DataLoader(
                val_set, 
                batch_size=hyperparams['batch_size'],
                shuffle=False, 
                num_workers=2,
                pin_memory=True
            )
            
            # 构建模型
            model = Classifier(
                block=Residual_Block,
                num_layers=hyperparams['num_layers'],
                num_classes=NUM_CLASSES,
                dropout1=hyperparams['dropout1'],
                dropout2=hyperparams['dropout2']
            ).to(self.device)
            
            # 损失函数 - 使用配置中的类别权重
            criterion = FocalLoss(
                class_num=NUM_CLASSES,
                alpha=FOCAL_ALPHA,
                gamma=hyperparams['focal_gamma']
            )
            
            # 优化器 - 与实验四一致
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=hyperparams['lr'],
                weight_decay=1e-5
            )
            
            # 学习率调度器 - 与实验四一致
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=16, T_mult=1
            )
            
            # 快速训练 - 带进度条
            total_batches = len(train_loader) * self.eval_epochs
            with tqdm(total=total_batches, desc="    训练中", leave=False, 
                      ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                for epoch in range(self.eval_epochs):
                    model.train()
                    for imgs, labels in train_loader:
                        # non_blocking=True 实现异步传输
                        imgs = imgs.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                        
                        optimizer.zero_grad()
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                        optimizer.step()
                        pbar.update(1)
                    
                    scheduler.step()  # 每个epoch后更新学习率
            
            # 验证评估 - 带进度条
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc="    验证中", leave=False, 
                                          ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
                    imgs = imgs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    outputs = model(imgs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_acc = correct / total
            
            # 清理资源 - 释放 DataLoader 和模型避免文件描述符泄漏
            del train_loader, val_loader, train_set, val_set
            del model, optimizer, criterion, scheduler
            torch.cuda.empty_cache()
            
            return val_acc
            
        except Exception as e:
            print(f"    [错误] 评估失败: {e}")
            return 0.0  # 返回最低适应度


# ==================================================================================
#                               主函数
# ==================================================================================
def main():
    """主函数：执行遗传算法超参数优化"""
    
    print("\n" + "=" * 70)
    print("   实验五：基于遗传算法的神经网络超参数优化")
    print("   GA-HPO: Genetic Algorithm for Hyperparameter Optimization")
    print("=" * 70)
    
    # 使用顶部 Config 模块中定义的全局配置
    print(f"\n[配置信息]")
    print(f"  GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")
    print(f"  数据集目录: {DATASET_DIR}")
    print(f"  遗传算法参数: {GA_CONFIG}")
    print(f"  快速评估轮数: {EVAL_EPOCHS}")
    
    # ==================== 准备数据 ====================
    print("\n[Step 1] 加载数据集...")
    
    train_dir = os.path.join(DATASET_DIR, "training")
    val_dir = os.path.join(DATASET_DIR, "validation")
    
    # 检查数据集是否存在
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"[错误] 数据集目录不存在!")
        print(f"  请确保以下目录存在:")
        print(f"  - {train_dir}")
        print(f"  - {val_dir}")
        print("\n正在使用模拟数据进行演示...")
        
        # 模拟数据用于演示
        class MockEvaluator:
            def __init__(self):
                self.eval_count = 0
            
            def evaluate(self, hyperparams):
                """模拟评估函数"""
                self.eval_count += 1
                # 基于超参数生成模拟适应度
                lr_score = 1 - abs(np.log10(hyperparams['lr']) + 3.5) / 2.5
                layer_score = sum(hyperparams['num_layers']) / 16
                dropout_score = 1 - abs(hyperparams['dropout1'] - 0.3) / 0.4
                gamma_score = 1 - abs(hyperparams['focal_gamma'] - 2.5) / 4.5
                
                base_score = 0.5 + 0.3 * (lr_score * 0.3 + layer_score * 0.2 + 
                                          dropout_score * 0.25 + gamma_score * 0.25)
                noise = np.random.normal(0, 0.02)
                return np.clip(base_score + noise, 0.3, 0.95)
        
        evaluator = MockEvaluator()
    else:
        # 加载真实数据
        train_files = [os.path.join(train_dir, x) for x in os.listdir(train_dir) if x.endswith('.jpg')]
        val_files = [os.path.join(val_dir, x) for x in os.listdir(val_dir) if x.endswith('.jpg')]
        
        print(f"  训练集: {len(train_files)} 张图像")
        print(f"  验证集: {len(val_files)} 张图像")
        
        evaluator = NNEvaluator(
            train_files=train_files,
            val_files=val_files,
            eval_epochs=EVAL_EPOCHS
        )
    
    # ==================== 遗传算法优化 ====================
    print("\n[Step 2] 初始化遗传算法...")
    
    ga = GeneticAlgorithm(**GA_CONFIG)
    
    print("\n[Step 3] 开始进化优化...")
    best_individual, history = ga.evolve(evaluator)
    
    # ==================== 结果输出 ====================
    print("\n[Step 4] 保存结果...")
    
    # 保存最优超参数
    best_hp = best_individual.decode()
    results = {
        'best_hyperparameters': best_hp,
        'best_fitness': best_individual.fitness,
        'evolution_history': {
            'best_fitness': history['best_fitness'],
            'avg_fitness': history['avg_fitness'],
        },
        'ga_config': GA_CONFIG,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 将numpy类型转换为Python原生类型以便JSON序列化
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results = convert_to_serializable(results)
    
    results_path = "ga_hpo_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  结果已保存至: {results_path}")
    
    # 绘制进化曲线
    print("\n[Step 5] 绘制进化曲线...")
    try:
        ga.plot_evolution(save_path="evolution_curve.png")
    except Exception as e:
        print(f"  绘图失败 (可能是无GUI环境): {e}")
    
    # ==================== 最终报告 ====================
    print("\n" + "=" * 70)
    print("                    遗传算法优化完成 - 最终报告")
    print("=" * 70)
    print(f"\n📊 最优超参数组合:")
    print(f"   ├── 学习率 (lr):        {best_hp['lr']:.6e}")
    print(f"   ├── 批大小 (batch_size): {best_hp['batch_size']}")
    print(f"   ├── 网络深度 (layers):   {best_hp['num_layers']}")
    print(f"   ├── Dropout率:          {best_hp['dropout1']:.3f}")
    print(f"   └── Focal γ:            {best_hp['focal_gamma']:.3f}")
    print(f"\n🎯 最优验证准确率: {best_individual.fitness * 100:.2f}%")
    print(f"📈 评估总次数: {evaluator.eval_count}")
    print(f"⏱️  总耗时: {sum(history['generation_time']):.1f} 秒")
    print("=" * 70)
    
    # 返回结果供外部使用
    return best_individual, history


# ==================================================================================
#                               程序入口
# ==================================================================================
if __name__ == "__main__":
    # 设置 matplotlib 中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 运行主函数
    best, history = main()
