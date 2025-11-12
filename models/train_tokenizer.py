# -*- codeing =utf-8 -*-
# @Time : 2025/4/8 21:47
# @Author :gh
# @File :train_tokenizer.py.py
# @Software: PyCharm
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast

def train_tokenizer(smiles_path, save_dir, vocab_size=128):
    # 训练 wordpiece tokenizer
    tokenizer = BertWordPieceTokenizer(
        lowercase=False,
        clean_text=True,
        strip_accents=False
    )

    tokenizer.train(
        files=[smiles_path],
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    # 保存 vocab.txt
    tokenizer.save_model(save_dir)

    # 转换为 transformers 兼容格式
    fast_tokenizer = BertTokenizerFast.from_pretrained(save_dir)
    fast_tokenizer.save_pretrained(save_dir)

    print(f"Tokenizer saved to {save_dir}")

if __name__ == "__main__":
    train_tokenizer(
        smiles_path="data/all_smiles.txt",  # 一行一个SMILES
        save_dir="./smiles_tokenizer",
        vocab_size=128
    )