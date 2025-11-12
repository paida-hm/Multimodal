import torch.nn as nn
from torch_geometric.graphgym.register import config_dict
from transformers import BertTokenizer, BertModel, BertConfig
from transformers.models.auto.tokenization_auto import BertTokenizerFast

class SmilesBERTEncoder(nn.Module):
    """
    支持直接输入 SMILES 字符串的轻量封装器。
    输入：List[str]（原始 SMILES 字符串）
    输出：Tensor(batch_size, hidden_dim) —— [CLS] 表征
    """

    def __init__(self, model, tokenizer):
        super(SmilesBERTEncoder, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        # self.project = nn.Linear(768, out_dim)  # 1D 输出映射到 256
        self.project = nn.Linear(768, 256)


    def forward(self, smiles_strs):
        # smiles_strs: List[str]
        encoding = self.tokenizer(
            smiles_strs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128  # 可根据 SMILES 长度调整
        )
        input_ids = encoding['input_ids'].to(self.model.device)
        attention_mask = encoding['attention_mask'].to(self.model.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS] token
        projected = self.project(cls_embedding)  # 映射到 256
        return projected
        # return cls_embedding

    @staticmethod
    def build_from_config(config_dict: dict, device):
        """
        从配置构建一个从头训练的 SMILES BERT 编码器。
        要求 config_dict 至少包括 vocab_path，其余可选：
            - hidden_size
            - num_hidden_layers
            - num_attention_heads
            - intermediate_size
            - max_position_embeddings
            - dropout
            - project_dim
        """
        vocab_path = config_dict['vocab_path']
        tokenizer = BertTokenizerFast.from_pretrained(vocab_path, local_files_only=True)
        config = BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=config_dict.get('hidden_size', 768),
            num_hidden_layers=config_dict.get('num_hidden_layers', 6),
            num_attention_heads=config_dict.get('num_attention_heads', 12),
            intermediate_size=config_dict.get('intermediate_size', 3072),
            max_position_embeddings=config_dict.get('max_position_embeddings', 512),
            hidden_dropout_prob=config_dict.get('dropout', 0.1),
            attention_probs_dropout_prob=config_dict.get('dropout', 0.1),
            pad_token_id=tokenizer.pad_token_id,
        )
        model = BertModel(config).to(device)
        # 构建主编码器
        encoder = SmilesBERTEncoder(model, tokenizer).to(device)
        # 修改投影层的输出维度
        project_dim = config_dict.get('project_dim', 256)
        encoder.project = nn.Linear(config.hidden_size, project_dim).to(device)
        return encoder



    @staticmethod
    def load_smiles_bert(model_path, device):
        tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
        model = BertModel.from_pretrained(model_path, local_files_only=True)

        # 可选：启用训练
        for param in model.parameters():
            param.requires_grad = True

        encoder = SmilesBERTEncoder(model, tokenizer).to(device)
        return encoder
