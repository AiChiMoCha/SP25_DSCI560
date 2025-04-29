import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 设置随机种子以确保结果可复现
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据准备：这里假设您已经有一个CSV文件包含股票相关文本和情感标签
# CSV有两列：'text'(文本内容)和'sentiment'(0为负面，1为中性，2为正面)
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# 创建自定义数据集类
class StockSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 使用tokenizer处理文本
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 模型训练函数
def train_model(model, train_dataloader, val_dataloader, epochs=3):
    # 优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # 训练循环
    for epoch in range(epochs):
        print(f"\n开始第 {epoch+1}/{epochs} 轮训练")
        
        # 训练模式
        model.train()
        train_loss = 0
        
        for batch in train_dataloader:
            # 将数据移动到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 清除之前的梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"训练Loss: {avg_train_loss:.4f}")
        
        # 评估模式
        model.eval()
        val_loss = 0
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                
                # 获取预测结果
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
        
        avg_val_loss = val_loss / len(val_dataloader)
        accuracy = accuracy_score(actual_labels, predictions)
        f1 = f1_score(actual_labels, predictions, average='weighted')
        
        print(f"验证Loss: {avg_val_loss:.4f}")
        print(f"验证准确率: {accuracy:.4f}")
        print(f"验证F1分数: {f1:.4f}")
        
    return model

# 预测函数
def predict_sentiment(text, model, tokenizer):
    # 将模型设为评估模式
    model.eval()
    
    # 对文本进行处理
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
    
    # 将预测结果转换为情感类别
    sentiment_map = {0: "负面", 1: "中性", 2: "正面"}
    
    return sentiment_map[preds.item()]


def main():

    file_path = "stock_sentiment_data.csv"  
    data = load_data(file_path)
    
    # 2. 分割数据为训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data['text'].values, 
        data['sentiment'].values, 
        test_size=0.2, 
        random_state=42
    )
    
    # 3. 初始化tokenizer和模型
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3  # 3个类别：负面、中性、正面
    )
    
    # 将模型移动到设备上
    model.to(device)
    
    # 4. 创建数据集和数据加载器
    train_dataset = StockSentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = StockSentimentDataset(val_texts, val_labels, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    # 5. 训练模型
    model = train_model(model, train_dataloader, val_dataloader, epochs=3)
    
    # 6. 保存模型和tokenizer
    model_save_path = "./stock_sentiment_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"模型和tokenizer已保存到 {model_save_path}")
    
    # 7. 测试预测
    test_texts = [
        "公司第二季度营收显著增长，超过分析师预期。",
        "投资者对最新发布的财报表示担忧，股价下跌5%。",
        "市场观望态度明显，交易量维持在近期平均水平。"
    ]
    
    print("\n测试预测结果:")
    for text in test_texts:
        sentiment = predict_sentiment(text, model, tokenizer)
        print(f"文本: {text}")
        print(f"情感预测: {sentiment}\n")

# 加载和使用已训练好的模型
def load_and_use_model(model_path, text):
    # 加载模型和tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    
    # 预测情感
    sentiment = predict_sentiment(text, model, tokenizer)
    return sentiment

if __name__ == "__main__":
    # 训练新模型
    main()
    
    # 或者加载现有模型进行预测
    # model_path = "./stock_sentiment_model"
    # text = "该公司宣布新的收购计划，预计将提高明年利润。"
    # sentiment = load_and_use_model(model_path, text)
    # print(f"文本: {text}")
    # print(f"情感预测: {sentiment}")