# %%
import sys
sys.path.append('../')
from util import load_data
data_path='../data/sst2_shuffled.tsv.1'
train_data,test_data,categories=load_data.load_sentence_polarity(data_path=data_path)

# %%
# 这里新的模型transformer
# 先确定pre train模型的名称，所确定的tokenize
# 加载预训练模型，因为这里是英文数据集，需要用在英文上的预训练模型：bert-base-uncased
# uncased指该预训练模型对应的词表不区分字母的大小写
# 详情可了解：https://huggingface.co/bert-base-uncased
pretrained_model_name = 'bert-base-uncased'
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from tqdm import tqdm
import os
import time
from transformers import BertTokenizer
from transformers import logging

# %%
# 编写好制作数据集的方式，先定义dataset、后面定义dataloader
class BertDataset(Dataset):
    def __init__(self,dataset):
        self.dataset=dataset
        self.data_size=len(dataset)
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, index):
        return self.dataset[index]

def coffate_fn(examples):
    inputs,targets=[],[]
    for polar,sent in examples:
        inputs.append(sent)
        targets.append(int(polar))
    # 这里的tokenizer是后面提供好pretrain model之后的API
    inputs = tokenizer(inputs,
                       padding=True,
                       truncation=True,
                       return_tensors="pt",
                       max_length=512)
    targets = torch.tensor(targets)
    return inputs,targets

pretrained_model_name = 'bert-base-uncased'
# 加载预训练模型对应的tokenizer
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
batch_size=32
train_dataset=BertDataset(train_data)
test_dataset=BertDataset(test_data)
train_dataloader=DataLoader(train_dataset,batch_size=batch_size,collate_fn=coffate_fn,shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=batch_size,collate_fn=coffate_fn,shuffle=True)


# %%
for batch in train_dataloader:
    print(batch)
    break

# %%
# 之后是定义模型的名称
class BertSST2Model(nn.Module):
    def __init__(self,class_size,pretrained_model_name=pretrained_model_name) -> None:
        super(BertSST2Model,self).__init__()
        # 记载hugging face的bertmodel
        # bertmodel的最终输出维度默认为768
        # 对其进行调整输入的维度调整
        self.bert=BertModel.from_pretrained(pretrained_model_name,return_dict=True)
        # 修改最后一个线性层
        self.classifier=nn.Linear(768,class_size)
    
    def forward(self,inputs):
        """
        前向推理的过程
        inputs 处理好的数据 shape=batchsize*max_len

        """
        input_ids,input_tyi,input_attn_mask=inputs['input_ids'],inputs['token_type_ids'],inputs['attention_mask']
        # TODO 如何实现
        output=self.bert(input_ids,input_tyi,input_attn_mask)
        categories_numberic=self.classifier(output.pooler_output)
        return categories_numberic

def save_pretrained(model, path):
    # 保存模型，先利用os模块创建文件夹，后利用torch.save()写入模型文件
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))

# %%
# 定义超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size=16
num_epoch=200
check_step=20
learning_rate=1e-5
model=BertSST2Model(class_size=2)
model.to(device)
optimizer=Adam(model.parameters(),learning_rate)
celoss=nn.CrossEntropyLoss()

# %%
# 训练过程
# 记录当前训练时间，用以记录日志和存储
timestamp = time.strftime("%m_%d_%H_%M", time.localtime())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.train()
for epoch in range(1,num_epoch+1):
    total_loss=0
    for batch in tqdm(train_dataloader,desc=f'Training epoch {epoch}'):
        inputs,targets=[x.to(device) for x in batch]
        optimizer.zero_grad()
        bert_output=model(inputs)
        loss=celoss(bert_output,targets)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    #测试过程
    # acc统计模型在测试数据上分类结果中的正确个数
    acc = 0
    for batch in tqdm(test_dataloader, desc=f"Testing"):
        inputs, targets = [x.to(device) for x in batch]
        # with torch.no_grad(): 为固定写法，
        # 这个代码块中的全部有关tensor的操作都不产生梯度。目的是节省时间和空间，不加也没事
        with torch.no_grad():
            bert_output = model(inputs)
            """
            .argmax()用于取出一个tensor向量中的最大值对应的下表序号，dim指定了维度
            假设 bert_output为3*2的tensor：
            tensor
            [
                [3.2,1.1],
                [0.4,0.6],
                [-0.1,0.2]
            ]
            则 bert_output.argmax(dim=1) 的结果为：tensor[0,1,1]
            """
            acc += (bert_output.argmax(dim=1) == targets).sum().item()
    #输出在测试集上的准确率
    print(f"Acc: {acc / len(test_dataloader):.2f}")
    if epoch % check_step == 0:
        # 保存模型
        checkpoints_dirname = "bert_sst2_" + timestamp
        os.makedirs(checkpoints_dirname, exist_ok=True)
        save_pretrained(model,
                        checkpoints_dirname + '/checkpoints-{}/'.format(epoch))

# %%
test='Because he is a random guess hahahahahahahahahahahaha'
test=tokenizer(test,padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512)
test.to(device)
if model(test).argmax(-1).item()==1:
    print('This is a negative sentence')
else:
    print('This is a positive sentence')


