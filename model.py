import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re
import warnings
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from collections import Counter
warnings.filterwarnings('ignore')

# ==================== 0. 配置与日志初始化 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('model_train.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Config:
    # 数据路径
    CSIC_DATA_DIR = "./CSIC2010"
    
    # 模型配置（词汇表大小将动态确定，此处仅作默认值）
    VOCAB_SIZE = None  # 将在加载数据时更新
    MAX_LEN = 200      # token 序列最大长度（原为500字符，现适当减小）
    TRAIN_RATIO = 0.8
    BATCH_SIZE = 32
    
    EMBED_DIM = 128
    CNN_CHANNELS = 64
    LSTM_HIDDEN = 64
    NUM_CLASSES = 2
    
    # 训练配置
    EPOCHS = 20
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    SCHEDULER_PATIENCE = 2
    SCHEDULER_FACTOR = 0.5
    EARLY_STOP_PATIENCE = 5
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== 1. 分词与词汇表构建 ====================
def tokenize_http(request):
    """
    对HTTP请求进行分词，保留单词和常见符号。
    返回 token 列表。
    """
    # 按字母数字、常见符号分割
    tokens = re.findall(r'[A-Za-z0-9]+|[=&?/.;:\-]+', request)
    return tokens

def build_vocab(texts, min_freq=2):
    """
    从文本列表构建词汇表，返回 word2idx 和 vocab_size。
    """
    counter = Counter()
    for text in texts:
        tokens = tokenize_http(text)
        counter.update(tokens)
    
    # 保留出现次数 >= min_freq 的 token
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab, len(vocab)

# ==================== 2. 数据集类（token级） ====================
class HTTPDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=Config.MAX_LEN):
        self.texts = [text if isinstance(text, str) and text.strip() else " " for text in texts]
        self.labels = np.array(labels, dtype=int)
        self.word2idx = word2idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = tokenize_http(text)[:self.max_len]
        
        # 转换为索引，未知词用 <UNK> 的索引（1）
        indices = [self.word2idx.get(tok, 1) for tok in tokens]
        
        # 填充
        indices += [0] * (self.max_len - len(indices))
        
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ==================== 3. 模型定义 ====================
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32, device=x.device))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        context_vector = torch.matmul(attention_weights, V)
        return context_vector, attention_weights

class InjectionAttackDetector(nn.Module):
    def __init__(self, vocab_size, embed_dim=Config.EMBED_DIM, max_len=Config.MAX_LEN,
                 cnn_channels=Config.CNN_CHANNELS, lstm_hidden=Config.LSTM_HIDDEN, num_classes=Config.NUM_CLASSES):
        super(InjectionAttackDetector, self).__init__()
        
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        
        self.conv1 = nn.Conv1d(embed_dim, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, cnn_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, cnn_channels, kernel_size=7, padding=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        
        conv_output_dim = cnn_channels * 3
        
        self.bilstm = nn.LSTM(
            input_size=conv_output_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if torch.cuda.is_available() else 0.0
        )
        lstm_output_dim = lstm_hidden * 2
        
        self.attention = SelfAttention(lstm_output_dim)
        
        self.fc1 = nn.Linear(lstm_output_dim, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        embeddings = self.embedding(input_ids)
        embeddings = embeddings + self.position_encoding[:, :seq_len, :]
        
        embeddings_transposed = embeddings.transpose(1, 2)
        
        conv1_out = self.relu(self.conv1(embeddings_transposed))
        conv1_out = self.pool(conv1_out)
        
        conv2_out = self.relu(self.conv2(embeddings_transposed))
        conv2_out = self.pool(conv2_out)
        
        conv3_out = self.relu(self.conv3(embeddings_transposed))
        conv3_out = self.pool(conv3_out)
        
        conv1_out = conv1_out.transpose(1, 2)
        conv2_out = conv2_out.transpose(1, 2)
        conv3_out = conv3_out.transpose(1, 2)
        
        cnn_output = torch.cat([conv1_out, conv2_out, conv3_out], dim=2)
        
        lstm_output, _ = self.bilstm(cnn_output)
        
        context_vector, attention_weights = self.attention(lstm_output)
        
        attention_output = torch.mean(context_vector, dim=1)
        
        fc1_out = self.relu(self.fc1(attention_output))
        fc1_out = self.dropout(fc1_out)
        logits = self.fc2(fc1_out)
        
        return logits, attention_weights

class EnhancedInjectionDetector(InjectionAttackDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.hex_conv = nn.Conv1d(self.embedding.embedding_dim, 32, kernel_size=3, padding=1)
        self.unicode_conv = nn.Conv1d(self.embedding.embedding_dim, 32, kernel_size=3, padding=1)
        
        # 输入维度 = 注意力输出(128) + 混淆特征(64) = 188
        self.fc1 = nn.Linear(self.attention.hidden_dim + 64, 64)
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        embeddings = self.embedding(input_ids)
        embeddings = embeddings + self.position_encoding[:, :seq_len, :]
        
        emb_trans = embeddings.transpose(1, 2)
        hex_features = self.relu(self.hex_conv(emb_trans))
        unicode_features = self.relu(self.unicode_conv(emb_trans))
        
        hex_pool = torch.mean(hex_features, dim=-1)
        unicode_pool = torch.mean(unicode_features, dim=-1)
        obfuscation_features = torch.cat([hex_pool, unicode_pool], dim=1)  # (batch_size, 64)
        
        embeddings_transposed = embeddings.transpose(1, 2)
        conv1_out = self.relu(self.pool(self.conv1(embeddings_transposed)))
        conv2_out = self.relu(self.pool(self.conv2(embeddings_transposed)))
        conv3_out = self.relu(self.pool(self.conv3(embeddings_transposed)))
        
        conv1_out = conv1_out.transpose(1, 2)
        conv2_out = conv2_out.transpose(1, 2)
        conv3_out = conv3_out.transpose(1, 2)
        
        cnn_output = torch.cat([conv1_out, conv2_out, conv3_out], dim=2)
        lstm_output, _ = self.bilstm(cnn_output)
        context_vector, attention_weights = self.attention(lstm_output)
        attention_output = torch.mean(context_vector, dim=1)  # (batch_size, 128)
        
        combined_features = torch.cat([attention_output, obfuscation_features], dim=1)  # (batch_size, 188)
        
        fc1_out = self.relu(self.fc1(combined_features))
        fc1_out = self.dropout(fc1_out)
        logits = self.fc2(fc1_out)
        
        return logits, attention_weights

# ==================== 4. 训练器（加入阈值参数） ====================
class ModelTrainer:
    def __init__(self, model, device=Config.DEVICE, class_weights=None):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=Config.SCHEDULER_PATIENCE,
            factor=Config.SCHEDULER_FACTOR
        )
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            logits, _ = self.model(input_ids)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, _ = self.model(input_ids)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        self.scheduler.step(avg_loss)
        return avg_loss, accuracy
    
    def predict(self, http_request, threshold=0.5):
        """返回预测结果，支持自定义阈值"""
        self.model.eval()
        
        # 需要访问外部的 word2idx
        if not hasattr(self.model, 'word2idx'):
            raise AttributeError("模型缺少 word2idx 属性，无法进行预测。")
        word2idx = self.model.word2idx
        max_len = self.model.max_len
        
        tokens = tokenize_http(http_request)[:max_len]
        indices = [word2idx.get(tok, 1) for tok in tokens]  # 1 为 <UNK>
        indices += [0] * (max_len - len(indices))
        
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits, attention_weights = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            # 使用自定义阈值判断攻击类
            attack_prob = probabilities[0, 1].item()
            predicted_class = 1 if attack_prob > threshold else 0
            
        return {
            'prediction': 'Attack' if predicted_class == 1 else 'Normal',
            'confidence': probabilities[0, predicted_class].item(),
            'attack_prob': attack_prob,
            'normal_prob': probabilities[0, 0].item(),
            'attention_weights': attention_weights.cpu().numpy() if attention_weights is not None else None
        }

# ==================== 5. 数据集加载函数（token级） ====================
def parse_csic2010_file(file_path):
    """解析CSIC 2010数据集文件"""
    requests = []
    current_request = []
    
    encodings = ['latin-1', 'utf-8', 'cp1252']
    content = None
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.readlines()
            break
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    if content is None:
        raise ValueError(f"无法解析文件 {file_path}：所有编码尝试失败")
    
    for line in content:
        line = line.strip()
        if not line:
            if current_request:
                full_request = '\n'.join(current_request)
                full_request = re.sub(r'\s+', ' ', full_request).strip()
                if full_request:
                    requests.append(full_request)
                current_request = []
        else:
            if not line.startswith('#'):
                current_request.append(line)
    
    if current_request:
        full_request = '\n'.join(current_request)
        full_request = re.sub(r'\s+', ' ', full_request).strip()
        if full_request:
            requests.append(full_request)
    
    return requests

def load_csic2010_dataset(data_dir=Config.CSIC_DATA_DIR, max_len=Config.MAX_LEN, train_ratio=Config.TRAIN_RATIO):
    """
    加载数据集，构建词汇表，返回 DataLoader 和 word2idx。
    """
    normal_train_path = os.path.join(data_dir, 'normalTrafficTraining.txt')
    normal_test_path = os.path.join(data_dir, 'normalTrafficTest.txt')
    attack_test_path = os.path.join(data_dir, 'anomalousTrafficTest.txt')
    
    logger.info("=" * 60)
    logger.info("开始解析CSIC 2010数据集...")
    normal_train_requests = parse_csic2010_file(normal_train_path)
    normal_test_requests = parse_csic2010_file(normal_test_path)
    attack_test_requests = parse_csic2010_file(attack_test_path)
    
    logger.info(f"正常训练请求数: {len(normal_train_requests)}")
    logger.info(f"正常测试请求数: {len(normal_test_requests)}")
    logger.info(f"攻击测试请求数: {len(attack_test_requests)}")
    
    # 划分训练/测试集（按比例）
    normal_all = normal_train_requests + normal_test_requests
    normal_train_num = int(len(normal_all) * train_ratio)
    normal_train = normal_all[:normal_train_num]
    normal_test = normal_all[normal_train_num:]
    
    attack_train_num = int(len(attack_test_requests) * train_ratio)
    attack_train = attack_test_requests[:attack_train_num]
    attack_test = attack_test_requests[attack_train_num:]
    
    train_texts = normal_train + attack_train
    train_labels = [0]*len(normal_train) + [1]*len(attack_train)
    
    test_texts = normal_test + attack_test
    test_labels = [0]*len(normal_test) + [1]*len(attack_test)
    
    # 构建词汇表（基于训练集）
    logger.info("构建词汇表...")
    word2idx, vocab_size = build_vocab(train_texts, min_freq=2)
    logger.info(f"词汇表大小: {vocab_size}")
    
    # 打乱训练集
    train_idx = np.random.permutation(len(train_texts))
    train_texts = [train_texts[i] for i in train_idx]
    train_labels = [train_labels[i] for i in train_idx]
    
    # 测试集不打乱（保持一致性）
    
    # 数据集统计
    logger.info(f"\n最终数据集统计:")
    logger.info(f"训练集 - 总数: {len(train_texts)}, 正常: {train_labels.count(0)}, 攻击: {train_labels.count(1)}")
    logger.info(f"测试集 - 总数: {len(test_texts)}, 正常: {test_labels.count(0)}, 攻击: {test_labels.count(1)}")
    logger.info("=" * 60)
    
    train_dataset = HTTPDataset(train_texts, train_labels, word2idx, max_len)
    test_dataset = HTTPDataset(test_texts, test_labels, word2idx, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, test_loader, word2idx, vocab_size

# ==================== 6. 测试函数（支持自定义阈值） ====================
def batch_test_model(trainer, test_loader, threshold=0.5, save_metrics=True):
    """批量测试模型，使用指定的阈值进行预测"""
    trainer.model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(trainer.device)
            labels = batch['label'].to(trainer.device)
            
            logits, _ = trainer.model(input_ids)
            probs = F.softmax(logits, dim=1)
            
            # 使用阈值判断攻击类
            preds = (probs[:, 1] > threshold).cpu().numpy().astype(int)
            labels_np = labels.cpu().numpy()
            probs_np = probs[:, 1].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_probs.extend(probs_np)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"模型批量测试结果（阈值 = {threshold:.2f}）")
    logger.info("=" * 60)
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=['Normal（正常）', 'Attack（攻击）'],
        digits=4
    )
    # 使用 logger.info 同时输出到控制台和日志文件
    logger.info("\n" + report)
    
    auc = roc_auc_score(all_labels, all_probs)
    logger.info(f"AUC值（攻击类）: {auc:.4f}")
    
    cm = confusion_matrix(all_labels, all_preds)
    logger.info("\n混淆矩阵:")
    logger.info(f"          预测正常  预测攻击")
    logger.info(f"实际正常    {cm[0][0]}       {cm[0][1]}")
    logger.info(f"实际攻击    {cm[1][0]}       {cm[1][1]}")
    
    if save_metrics:
        # 保存 ROC 曲线
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC Curve (阈值={threshold:.2f})')
        plt.legend()
        plt.savefig(f'roc_curve_th{threshold:.2f}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存混淆矩阵
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix (阈值={threshold:.2f})')
        plt.colorbar()
        plt.xticks([0, 1], ['Normal', 'Attack'])
        plt.yticks([0, 1], ['Normal', 'Attack'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.savefig(f'confusion_matrix_th{threshold:.2f}.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"可视化结果已保存：roc_curve_th{threshold:.2f}.png, confusion_matrix_th{threshold:.2f}.png")
    
    return {
        'report': report,
        'auc': auc,
        'confusion_matrix': cm,
        'threshold': threshold
    }

def single_sample_test(trainer, http_request, threshold=0.5):
    """测试单个样本，使用指定阈值"""
    logger.info("\n" + "=" * 60)
    logger.info(f"单样本测试结果（阈值 = {threshold:.2f}）")
    logger.info("=" * 60)
    result = trainer.predict(http_request, threshold=threshold)
    
    
    logger.info(f"原始请求（前150字符）: {http_request[:150]}...")
    logger.info(f"预测结果: {result['prediction']}")
    logger.info(f"正常概率: {result['normal_prob']:.4f}")
    logger.info(f"攻击概率: {result['attack_prob']:.4f}")
    logger.info(f"预测置信度: {result['confidence']:.4f}")
    
    if result['attention_weights'] is not None:
        attn_weights = result['attention_weights'][0]
        plot_size = min(50, attn_weights.shape[0])
        plt.figure(figsize=(12, 4))
        plt.imshow(attn_weights[:plot_size, :plot_size], cmap='hot', interpolation='nearest')
        plt.title(f'Attention Weights (前{plot_size}字符)')
        plt.xlabel('Sequence Position')
        plt.ylabel('Sequence Position')
        plt.colorbar()
        plt.savefig(f'attention_weights_th{threshold:.2f}.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"注意力权重图已保存：attention_weights_th{threshold:.2f}.png")
    
    return result

def load_trained_model(model_path='best_csic_model.pth', device=Config.DEVICE):
    """加载训练好的模型及词汇表"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在！请先训练模型。")
    
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    
    # 重建模型
    model = EnhancedInjectionDetector(
        vocab_size=checkpoint['vocab_size'],
        embed_dim=checkpoint.get('embed_dim', Config.EMBED_DIM),
        max_len=checkpoint.get('max_len', Config.MAX_LEN),
        cnn_channels=checkpoint.get('cnn_channels', Config.CNN_CHANNELS),
        lstm_hidden=checkpoint.get('lstm_hidden', Config.LSTM_HIDDEN),
        num_classes=Config.NUM_CLASSES
    )
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # 将词汇表挂载到模型上，供 predict 使用
    model.word2idx = checkpoint['word2idx']
    
    trainer = ModelTrainer(model, device=device)
    return trainer

# ==================== 7. 训练/测试主函数 ====================
def train_only(data_dir=Config.CSIC_DATA_DIR, max_len=Config.MAX_LEN, epochs=Config.EPOCHS, device=Config.DEVICE):
    """训练模型（使用 token 级输入）"""
    # 加载数据，得到词汇表
    train_loader, test_loader, word2idx, vocab_size = load_csic2010_dataset(
        data_dir=data_dir, max_len=max_len, train_ratio=Config.TRAIN_RATIO
    )
    
    # 计算类别权重
    train_labels = [batch['label'].numpy() for batch in train_loader]
    train_labels = np.concatenate(train_labels)
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    logger.info(f"类别权重：正常类={class_weights[0]:.4f}, 攻击类={class_weights[1]:.4f}")
    
    # 初始化模型
    logger.info("\n初始化注入攻击检测模型...")
    model = EnhancedInjectionDetector(
        vocab_size=vocab_size,
        embed_dim=Config.EMBED_DIM,
        max_len=max_len,
        cnn_channels=Config.CNN_CHANNELS,
        lstm_hidden=Config.LSTM_HIDDEN,
        num_classes=Config.NUM_CLASSES
    )
    # 挂载词汇表供预测使用
    model.word2idx = word2idx
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型总参数: {total_params:,}")
    
    trainer = ModelTrainer(model, device=device, class_weights=class_weights)
    
    # 训练循环
    logger.info("\n开始训练模型...")
    best_test_loss = float('inf')
    best_test_acc = 0.0
    early_stop_count = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_acc = test_acc
            early_stop_count = 0
            # 保存模型及词汇表
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_acc': best_test_acc,
                'best_loss': best_test_loss,
                'vocab_size': vocab_size,
                'max_len': max_len,
                'embed_dim': Config.EMBED_DIM,
                'cnn_channels': Config.CNN_CHANNELS,
                'lstm_hidden': Config.LSTM_HIDDEN,
                'word2idx': word2idx
            }, 'best_csic_model.pth')
        else:
            early_stop_count += 1
        
        logger.info(f"\nEpoch {epoch+1:2d}/{epochs}")
        logger.info(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
        logger.info(f"测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.2f}%")
        logger.info(f"当前最优测试准确率: {best_test_acc:.2f}% | 早停计数: {early_stop_count}/{Config.EARLY_STOP_PATIENCE}")
        logger.info("-" * 60)
        
        if early_stop_count >= Config.EARLY_STOP_PATIENCE:
            logger.info(f"\n早停触发！在第{epoch+1}轮停止训练")
            break
    
    logger.info(f"\n训练完成！最优模型已保存为 best_csic_model.pth（最优准确率：{best_test_acc:.2f}%）")

def test_only(data_dir=Config.CSIC_DATA_DIR, model_path='best_csic_model.pth', 
              max_len=Config.MAX_LEN, device=Config.DEVICE):
    """测试模型，尝试多个阈值并展示结果"""

    checkpoint = torch.load(model_path, map_location=torch.device(device))
    word2idx = checkpoint['word2idx']
    vocab_size = checkpoint['vocab_size']
    
    # 重新加载测试集（仅测试集，不重新划分）
    normal_test_path = os.path.join(data_dir, 'normalTrafficTest.txt')
    attack_test_path = os.path.join(data_dir, 'anomalousTrafficTest.txt')
    normal_test_requests = parse_csic2010_file(normal_test_path)
    attack_test_requests = parse_csic2010_file(attack_test_path)
    
 
    test_texts = normal_test_requests + attack_test_requests
    test_labels = [0]*len(normal_test_requests) + [1]*len(attack_test_requests)
    
    test_dataset = HTTPDataset(test_texts, test_labels, word2idx, max_len)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 加载模型
    logger.info("\n加载训练好的模型...")
    trainer = load_trained_model(model_path=model_path, device=device)
    
    # 尝试多个阈值
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for th in thresholds:
        batch_test_model(trainer, test_loader, threshold=th, save_metrics=True)
    
    # 单样本测试（使用默认阈值0.5，也可以指定其他阈值）
    logger.info("\n开始单样本测试（使用阈值0.5）...")
    normal_request = "GET /index.html HTTP/1.1\nHost: example.com\nUser-Agent: Mozilla/5.0"
    single_sample_test(trainer, normal_request, threshold=0.5)
    
    sql_injection = "GET /products.php?id=1' OR '1'='1-- HTTP/1.1\nHost: test.com"
    single_sample_test(trainer, sql_injection, threshold=0.5)
    
    cmd_injection = "POST /execute.php HTTP/1.1\ncmd=ls -la; cat /etc/passwd"
    single_sample_test(trainer, cmd_injection, threshold=0.5)
    
    obfuscated = "GET /search?q=%27%20OR%201%3D1%23 HTTP/1.1"
    single_sample_test(trainer, obfuscated, threshold=0.5)
    
    logger.info("\n测试完成！")

def train_and_test(data_dir=Config.CSIC_DATA_DIR, max_len=Config.MAX_LEN, epochs=Config.EPOCHS, device=Config.DEVICE):
    train_only(data_dir=data_dir, max_len=max_len, epochs=epochs, device=device)
    test_only(data_dir=data_dir, model_path='best_csic_model.pth', max_len=max_len, device=device)

# ==================== 8. 主程序入口 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("基于深度学习的注入攻击流量检测系统（改进版 - Token级输入 + 阈值调优）")
    print("=" * 60)
    print("请选择操作模式：")
    print("1. 仅训练模型")
    print("2. 仅测试模型")
    print("3. 训练模型 + 测试模型")
    print("=" * 60)
    
    while True:
        try:
            choice = int(input("请输入选项编号（1/2/3）："))
            if choice in [1, 2, 3]:
                break
            else:
                print("输入错误！请输入1、2或3")
        except ValueError:
            print("输入错误！请输入数字1、2或3")
    
    if choice == 1:
        logger.info("\n=== 开始执行：仅训练模型 ===")
        train_only(data_dir=Config.CSIC_DATA_DIR, max_len=Config.MAX_LEN, epochs=Config.EPOCHS, device=Config.DEVICE)
    elif choice == 2:
        logger.info("\n=== 开始执行：仅测试模型 ===")
        test_only(data_dir=Config.CSIC_DATA_DIR, model_path='best_csic_model.pth', 
                  max_len=Config.MAX_LEN, device=Config.DEVICE)
    elif choice == 3:
        logger.info("\n=== 开始执行：训练+测试 ===")
        train_and_test(data_dir=Config.CSIC_DATA_DIR, max_len=Config.MAX_LEN, epochs=Config.EPOCHS, device=Config.DEVICE)
    
    logger.info("\n=== 操作执行完成 ===")