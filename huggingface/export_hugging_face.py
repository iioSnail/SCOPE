import os
import shutil
from pathlib import Path

import torch
from transformers import BertConfig

from huggingface.csc_model import ChineseBertForCSC
from huggingface.csc_tokenizer import ChineseBertTokenizer

FPT_path = Path("../FPT")  # 作者提供的FPT文件
ckpt_path = Path("../ckpt/scope.ckpt")  # 训练出来的ckpt文件
source_path = "./ChineseBERT-for-csc"
output_path = "./iioSnail/ChineseBERT-for-csc"
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)

if not os.path.exists(FPT_path):
    print("[ERROR] 未找到 FPT 文件夹，请先下载后放在根目录!")
    exit(0)


def load_tokenizer():
    tokenizer = ChineseBertTokenizer.from_pretrained(source_path)
    return tokenizer


def copy_files():
    shutil.copytree(FPT_path / 'config', Path(output_path) / 'config')


def load_model():
    config = BertConfig(**{
        "_name_or_path": "iioSnail/ChineseBERT-for-csc",
        "attention_probs_dropout_prob": 0.1,
        "classifier_dropout": None,
        "directionality": "bidi",
        "gradient_checkpointing": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "output_past": True,
        "pad_token_id": 0,
        "pooler_fc_size": 768,
        "pooler_num_attention_heads": 12,
        "pooler_num_fc_layers": 3,
        "pooler_size_per_head": 128,
        "pooler_type": "first_token_transform",
        "position_embedding_type": "absolute",
        "torch_dtype": "float32",
        "type_vocab_size": 2,
        "use_cache": True,
        "vocab_size": 23236
    })

    model = ChineseBertForCSC(config)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state['state_dict'], strict=False)

    return model


def _test_model(tokenizer, model):
    inputs = tokenizer("我喜欢吃平果", return_tensors='pt')
    output_hidden = model(**inputs).logits
    print(tokenizer.convert_ids_to_tokens(output_hidden.argmax(-1)[0, 1:-1]))
    print(output_hidden.size())
    print("-" * 30)

def _test_model2(tokenizer, model):
    inputs = tokenizer(["在补习班他昨天晚上到夜里两点才读书，所以他一回加就累得睡觉了。", "我喜欢吃平果"],
                       padding=True,
                       max_length=10,
                       truncation=True,
                       return_tensors='pt')
    output_hidden = model(**inputs).logits
    print(tokenizer.convert_ids_to_tokens(output_hidden.argmax(-1)[0, 1:-1]))
    print(tokenizer.convert_ids_to_tokens(output_hidden.argmax(-1)[1, 1:-1]))
    print(output_hidden.size())
    print("-" * 30)




def export_tokenizer(tokenizer):
    tokenizer.register_for_auto_class("AutoTokenizer")
    tokenizer.save_vocabulary(output_path)
    tokenizer.save_pretrained(output_path)


def export_model(model):
    model.register_for_auto_class("AutoModel")
    model.save_pretrained(output_path)


def main():
    tokenizer = load_tokenizer()
    copy_files()
    model = load_model()
    _test_model(tokenizer, model)
    _test_model2(tokenizer, model)
    export_tokenizer(tokenizer)
    export_model(model)
    print("Export success!")
    # In final, you should copy "config" directory to the bert_path directory.


if __name__ == '__main__':
    main()
