# coding=utf-8
# Copyright 2024 Your Name
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Any, List

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss

import datasets
import evaluate
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import ModelOutput, check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig, PreTrainedTokenizer
from transformers.tokenization_utils import AddedToken  # 导入AddedToken


from tqdm.auto import tqdm
import wandb  # 

import esm


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

check_min_version("4.40.0.dev0")
require_version("datasets>=2.14.0", "请安装最新版本的datasets库")
require_version("tqdm>=4.64.0", "请安装tqdm库用于进度显示: pip install tqdm")

logger = logging.getLogger(__name__)


@dataclass
class DNAConditionalProteinConfig(PretrainedConfig):
   
    def __init__(
        self,
        hidden_dim: int = 512,
        dna_encoder_layers: int = 4,
        protein_decoder_layers: int = 16,
        intermediate_size: int = 1024,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        use_esm2: bool = True,
        use_rnafm: bool = True,
        esm2_model_name: str = "esm2_t33_650M_UR50D",
        vocab_size: int = 25,  
        dna_vocab_size: int = 8,  
        max_protein_length: int = 200,
        edit_feature_dim: int = 512,
        dna_feature_dim: int = 512,
        max_dna_length: int = 30,
        use_cls_token: bool = False,** kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dna_encoder_layers = dna_encoder_layers
        self.protein_decoder_layers = protein_decoder_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_esm2 = use_esm2
        self.use_rnafm = use_rnafm
        self.esm2_model_name = esm2_model_name
        self.vocab_size = vocab_size
        self.dna_vocab_size = dna_vocab_size
        self.max_protein_length = max_protein_length
        self.edit_feature_dim = edit_feature_dim
        self.dna_feature_dim = dna_feature_dim
        self.max_dna_length = max_dna_length
        self.use_cls_token = use_cls_token

@dataclass
class DNAConditionalProteinOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    dna_features: Optional[torch.FloatTensor] = None
    protein_features: Optional[torch.FloatTensor] = None


class DNAEncoder(nn.Module):

    def __init__(self, config: DNAConditionalProteinConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(
            config.dna_vocab_size, 
            config.hidden_dim,
            padding_idx=0  
        )
        

        self.pos_embedding = nn.Embedding(config.max_dna_length, config.hidden_dim)
        

        self.register_buffer(
            "valid_pos_range", 
            torch.arange(0, config.max_dna_length, dtype=torch.long)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.dna_encoder_layers
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        

        assert self.embedding.embedding_dim == self.pos_embedding.embedding_dim, \
            f"嵌入维度不匹配: 词嵌入 {self.embedding.embedding_dim}, 位置嵌入 {self.pos_embedding.embedding_dim}"

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None):

                
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        max_pos_embedding = self.pos_embedding.num_embeddings
        vocab_size = self.embedding.num_embeddings  
        

        invalid_mask = (input_ids < 0) | (input_ids >= vocab_size)
        if invalid_mask.any():
            invalid_ids = input_ids[invalid_mask].unique().tolist()
            invalid_count = invalid_mask.sum().item()
            

            input_ids = torch.where(
                invalid_mask,
                torch.tensor(self.embedding.padding_idx, device=device),
                input_ids
            )

            
        if seq_len > max_pos_embedding:
            seq_len_original = seq_len
            seq_len = max_pos_embedding
            input_ids = input_ids[:, :seq_len].contiguous()
            if attention_mask is not None:
                attention_mask = attention_mask[:, :seq_len].contiguous()

        

        positions = self.valid_pos_range[:seq_len].to(device)
        positions = positions.unsqueeze(0).repeat(batch_size, 1)
        

        try:
            hidden_states = self.embedding(input_ids)
            pos_emb = self.pos_embedding(positions)

            
            hidden_states = hidden_states + pos_emb
            
        except RuntimeError as e:
            error_msg = (
                f"嵌入计算失败: {str(e)}\n"
                f"输入形状: {input_ids.shape}\n"
                f"位置范围: [{positions.min()}, {positions.max()}]\n"
                f"最大位置编码: {max_pos_embedding}\n"
                f"词汇表大小: {vocab_size}\n"
                f"批次大小: {batch_size}, 序列长度: {seq_len}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        if attention_mask is not None:
            src_key_padding_mask = (1.0 - attention_mask).bool()
        else:
            src_key_padding_mask = None
        
        hidden_states = self.transformer_encoder(
            hidden_states, 
            src_key_padding_mask=src_key_padding_mask
        )
        hidden_states = self.layer_norm(hidden_states)

        if self.config.use_cls_token:
            features = hidden_states[:, 0, :]  
        else:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                features = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                features = hidden_states.mean(dim=1)
                
        return features, hidden_states


class ProteinDecoder(nn.Module):

    def __init__(self, config: DNAConditionalProteinConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_dim,
            padding_idx=0  
        )
        
        self.pos_embedding = nn.Embedding(config.max_protein_length, config.hidden_dim)
        
        self.cond_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.protein_decoder_layers
        )

        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)
        self.layer_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)

    def forward(
        self, 
        input_ids: torch.LongTensor,
        cond_features: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_mask: Optional[torch.Tensor] = None
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        hidden_states = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        hidden_states = hidden_states + self.pos_embedding(positions)
        

        cond_emb = self.cond_proj(cond_features).unsqueeze(1)
        hidden_states = hidden_states + cond_emb

        if decoder_mask is None and seq_len > 0:
            decoder_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        

        tgt_key_padding_mask = None
        if attention_mask is not None:
 
            tgt_key_padding_mask = attention_mask.to(dtype=torch.bool, device=device)

            tgt_key_padding_mask = ~tgt_key_padding_mask

        hidden_states = self.transformer_decoder(
            tgt=hidden_states,
            memory=cond_emb.repeat(1, seq_len, 1),  
            tgt_mask=decoder_mask,
            tgt_key_padding_mask=tgt_key_padding_mask  
        )
        hidden_states = self.layer_norm(hidden_states)

        logits = self.output_proj(hidden_states)
        
        return logits, hidden_states



class DNAConditionalProteinModel(PreTrainedModel):
    
    config_class = DNAConditionalProteinConfig
    base_model_prefix = "dna_protein"
    
    def __init__(self, config: DNAConditionalProteinConfig):
        super().__init__(config)
        self.config = config
        
        self.dna_encoder = DNAEncoder(config)

        self.esm2_model = None
        self.esm_alphabet = None
        self.esm_batch_converter = None
        self.rnafm_model = None
        self.rnafm_alphabet = None
        self.rnafm_batch_converter = None
        
        if config.use_esm2:
            logger.info(f"加载ESM-2基础模型: {config.esm2_model_name}")
            model_loader = getattr(esm.pretrained, config.esm2_model_name)
            
            try:
                self.esm2_model, self.esm_alphabet = model_loader(contact_regression=False)
            except TypeError:
                self.esm2_model, self.esm_alphabet = model_loader()
                
                if hasattr(self.esm2_model, 'contact_head'):
                    del self.esm2_model.contact_head
            
            self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
            self.esm2_model = self.esm2_model.eval()
        
        if config.use_rnafm:
            self.rnafm_model, self.rnafm_alphabet = fm.pretrained.rna_fm_t12()
            self.rnafm_batch_converter = self.rnafm_alphabet.get_batch_converter()
        
        input_dim = config.hidden_dim * (2 if config.use_rnafm else 1)
        self.feature_fusion = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        )
        
        self.protein_decoder = ProteinDecoder(config)
        
        self.post_init()
    
    def encode_dna(self, dna_seqs: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None,editing_positions: Optional[torch.LongTensor]=None):
        
        dna_features, dna_hidden = self.dna_encoder(dna_seqs, attention_mask)
        if editing_positions is not None:
            if editing_positions.dim() != 2:
                raise ValueError(f"editing_positions必须是2D张量，实际是{editing_positions.dim()}D")
            if editing_positions.shape[0] != batch_size:
                raise ValueError(f"editing_positions批次大小{editing_positions.shape[0]}与dna_seqs{batch_size}不匹配")
            self.edit_pos_projection = nn.Linear(config.max_dna_length, config.edit_feature_dim)
            edit_features = self.edit_pos_projection(editing_positions.float())
        
        if editing_positions is not None:
            dna_features = self.feature_fusion(torch.cat([dna_features, edit_features], dim=-1))
            self.dna_features_projection = nn.Linear(config.edit_feature_dim + config.dna_feature_dim, config.hidden_dim)
        if self.rnafm_model is not None:
            self.rnafm_model = self.rnafm_model.to(dna_seqs.device)

            batch_size = dna_seqs.shape[0]
            dna_strings = []
            
            for i in range(batch_size):

                seq_ids = dna_seqs[i][attention_mask[i] == 1].tolist()
                mapping = {3: 'A', 4: 'T', 5: 'C', 6: 'G', 7: 'N'}
                dna_str = ''.join([mapping.get(id, 'N') for id in seq_ids if id >= 3])
                dna_strings.append(("dna_" + str(i), dna_str))
            

            _, _, batch_tokens = self.rnafm_batch_converter(dna_strings)
            batch_tokens = batch_tokens.to(dna_seqs.device)
            
            with torch.no_grad():
                results = self.rnafm_model(batch_tokens, repr_layers=[12])
            
            rnafm_features = results["representations"][12].mean(dim=1)
            
            dna_features = torch.cat([dna_features, rnafm_features], dim=-1)
            self.dna_features_projection = nn.Linear(config.edit_feature_dim + dna_features.shape[-1], config.hidden_dim)
            dna_features = self.dna_features_projection(dna_features)
        
        return dna_features
    
    def encode_protein(self, protein_seqs: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None):
        if self.esm2_model is None:
            return None
        self.esm2_model = self.esm2_model.to(protein_seqs.device)

        batch_size = protein_seqs.shape[0]
        protein_strings = []
        
        for i in range(batch_size):

            seq_ids = protein_seqs[i][attention_mask[i] == 1].tolist()
            mapping = {v: k for k, v in self.esm_alphabet.tok_to_idx.items() if v >= 3}
            protein_str = ''.join([mapping.get(id, '') for id in seq_ids if id >= 3])
            protein_strings.append(("protein_" + str(i), protein_str))
        
        _, _, batch_tokens = self.esm_batch_converter(protein_strings)
        batch_tokens = batch_tokens.to(protein_seqs.device)

        with torch.no_grad():
            results = self.esm2_model(batch_tokens, repr_layers=[self.esm2_model.num_layers])

        esm_features = results["representations"][self.esm2_model.num_layers].mean(dim=1)
        
        return esm_features
    
    def forward(
        self,
        dna_input_ids: Optional[torch.LongTensor] = None,
        dna_attention_mask: Optional[torch.Tensor] = None,
        editing_positions: Optional[torch.Tensor] = None,
        protein_input_ids: Optional[torch.LongTensor] = None,
        protein_attention_mask: Optional[torch.Tensor] = None,
        protein_context_ids: Optional[torch.LongTensor] = None,
        protein_context_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_custom_vocab: bool = False,
        skip_esm2: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Union[Tuple, DNAConditionalProteinOutput]:

        dna_features = self.encode_dna(dna_input_ids, dna_attention_mask)
        
        protein_features = None
        if protein_context_ids is not None and not (use_custom_vocab and skip_esm2) and self.esm2_model is not None:
            protein_features = self.encode_protein(protein_context_ids, protein_context_mask)
        

        if protein_features is not None:
            combined_features = torch.cat([dna_features, protein_features], dim=-1)
            cond_features = self.feature_fusion(combined_features)
        else:
            cond_features = self.feature_fusion(dna_features)

        if protein_input_ids is None:
            batch_size = dna_input_ids.shape[0]
            start_token = 1  
            protein_input_ids = torch.full(
                (batch_size, 1), 
                fill_value=start_token, 
                device=dna_input_ids.device, 
                dtype=torch.long
            )
            if protein_attention_mask is None:
                protein_attention_mask = torch.ones_like(protein_input_ids)
        
        logits, decoder_hidden = self.protein_decoder(
            input_ids=protein_input_ids,
            cond_features=cond_features,
            attention_mask=protein_attention_mask
        )

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fct = CrossEntropyLoss(ignore_index=0) 
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output += (decoder_hidden,)
            if loss is not None:
                output = (loss,) + output
            return output
        
        return DNAConditionalProteinOutput(
            loss=loss,
            logits=logits,
            hidden_states=decoder_hidden if output_hidden_states else None,
            dna_features=dna_features,
            protein_features=protein_features
        )
    
    def generate(self, dna_sequence, max_length=100, use_custom_vocab=False, skip_esm2=True, tokenizer=None, editing_positions=None):
        self.eval()
        
        # 处理输入
        if isinstance(dna_sequence, str):
            if tokenizer is None:
                
                dna_input_ids = tokenizer(dna_sequence, return_tensors="pt", add_special_tokens=True)["input_ids"].to(self.device)
            else:
                dna_input_ids = dna_sequence.to(self.device)
        if editing_positions is not None:
            editing_positions = editing_positions.to(self.device)
        # 初始化生成
        batch_size = dna_input_ids.shape[0]
        start_token = 1 
        generated_ids = torch.full(
            (batch_size, 1), 
            fill_value=start_token, 
            device=self.device, 
            dtype=torch.long
        )
        if dna_input_ids is not None:
            dna_features = self.encode_dna(dna_input_ids)
        if editing_positions is not None:
            dna_features = self.encode_dna(dna_input_ids, editing_positions)

        cond_features = self.feature_fusion(dna_features)
        

        progress_bar = tqdm(range(max_length - 1), desc="生成蛋白质序列")
        for _ in progress_bar:

            with torch.no_grad():
                logits, _ = self.protein_decoder(
                    input_ids=generated_ids,
                    cond_features=cond_features
                )

            next_token_logits = logits[:, -1, :]

            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            if (next_token_id == 2).all(): 
                break
        

        if tokenizer is not None and isinstance(dna_sequence, str):
            generated_sequences = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return generated_sequences[0] if batch_size == 1 else generated_sequences
        else:
            return generated_ids


@dataclass
class ModelArguments:
    
    model_name_or_path: Optional[str] = field(
        default=None
    )
    model_type: str = field(
        default="dna_protein"
    )
    config_name: Optional[str] = field(
        default=None
    )
    hidden_dim: Optional[int] = field(
        default=None
    )
    use_esm2: bool = field(
        default=False
    )
    use_rnafm: bool = field(
        default=False
    )
    esm2_model_name: str = field(
        default="esm2_t33_650M_UR50D"
    )
    cache_dir: Optional[str] = field(
        default=None
    )
    trust_remote_code: bool = field(
        default=False
    )
    use_wandb: bool = field(
        default=True
    )
    wandb_project: str = field(
        default="dna-conditional-protein"
    )
    wandb_run_name: Optional[str] = field(
        default=None
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None
    )
    dataset_config_name: Optional[str] = field(
        default=None
    )
    train_file: Optional[str] = field(
        default=None
    )
    validation_file: Optional[str] = field(
        default=None
    )
    max_train_samples: Optional[int] = field(
        default=None
    )
    max_eval_samples: Optional[int] = field(
        default=None
    )
    max_dna_length: int = field(
        default=30
    )
    max_protein_length: int = field(
        default=200
    )
    overwrite_cache: bool = field(
        default=True
    )
    validation_split_percentage: int = field(
        default=5,
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None
    )
    use_custom_vocab: bool = field(
        default=True
    )
    skip_esm2: bool = field(
        default=True
    )



# 自定义蛋白质Tokenizer
class ProteinTokenizer(PreTrainedTokenizer):
    # 1. 词汇表（含新增的"1"和"2"特殊token）
    vocab_list = ["<pad>", "<cls>", "<eos>", "1", "2"] + ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
                                                         "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    vocab = {v: i for i, v in enumerate(vocab_list)}
    inv_vocab = {i: v for i, v in enumerate(vocab_list)}
    
    # 2. 特殊标记字符串定义
    pad_token_str = "<pad>"
    cls_token_str = "<cls>"
    eos_token_str = "<eos>"
    unk_token_str = "<unk>"
    special_1_token_str = "1"
    special_2_token_str = "2"
    
    def __init__(self, **kwargs):
        # 创建AddedToken对象
        pad_token = AddedToken(self.pad_token_str, lstrip=False, rstrip=False)
        cls_token = AddedToken(self.cls_token_str, lstrip=False, rstrip=False)
        eos_token = AddedToken(self.eos_token_str, lstrip=False, rstrip=False)
        unk_token = AddedToken(self.unk_token_str, lstrip=False, rstrip=False)
        special_1_token = AddedToken(self.special_1_token_str, lstrip=False, rstrip=False, special=True)
        special_2_token = AddedToken(self.special_2_token_str, lstrip=False, rstrip=False, special=True)
        
        # 调用基类初始化（注册所有特殊token）
        super().__init__(
            pad_token=pad_token,
            cls_token=cls_token,
            eos_token=eos_token,
            unk_token=unk_token,
            additional_special_tokens=[special_1_token, special_2_token],** kwargs
        )
        
        # 设置特殊token ID
        self.unk_token_id = self.vocab.get(self.unk_token_str, len(self.vocab))
        self.pad_token_id = self.vocab.get(self.pad_token_str, 0)
        self.cls_token_id = self.vocab.get(self.cls_token_str, 1)
        self.eos_token_id = self.vocab.get(self.eos_token_str, 2)
        self.special_1_token_id = self.vocab.get(self.special_1_token_str, 3)
        self.special_2_token_id = self.vocab.get(self.special_2_token_str, 4)
        
        # 注册特殊token（确保基类识别）
        self.add_special_tokens({
            "pad_token": pad_token,
            "cls_token": cls_token,
            "eos_token": eos_token,
            "unk_token": unk_token,
            "additional_special_tokens": [special_1_token, special_2_token]
        })
    
    # 新增：实现vocab_size属性
    @property
    def vocab_size(self):
        return len(self.get_vocab())
    
    def _tokenize(self, text):

        return list(text)
    
    def _convert_token_to_id(self, token):

        if isinstance(token, AddedToken):
            token = token.value
        return self.vocab.get(token, self.unk_token_id)
    
    def _convert_id_to_token(self, id):

        return self.inv_vocab.get(id, self.unk_token_str)
    
    def get_vocab(self):

        vocab = self.vocab.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab
    
    def save_vocabulary(self, save_directory, filename_prefix=None):

        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(
            save_directory, 
            "protein_vocab.txt" if filename_prefix is None else f"{filename_prefix}-protein_vocab.txt"
        )
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            # 先写基础词汇表
            for token in self.vocab_list:
                f.write(f"{token}\n")
            # 再写额外新增的特殊token（避免重复）
            for token in self.added_tokens_encoder:
                if token not in self.vocab_list:
                    f.write(f"{token}\n")
        
        return (vocab_file,)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):

        vocab_file = os.path.join(pretrained_model_name_or_path, "protein_vocab.txt")
        
        if os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_list = [line.strip() for line in f if line.strip()]
        else:
            vocab_list = cls.vocab_list
        
        tokenizer = cls(** kwargs)
        tokenizer.vocab_list = vocab_list
        tokenizer.vocab = {v: i for i, v in enumerate(vocab_list)}
        tokenizer.inv_vocab = {i: v for i, v in enumerate(vocab_list)}
        
        # 重新设置特殊token ID
        tokenizer.pad_token_id = tokenizer.vocab.get(tokenizer.pad_token_str, 0)
        tokenizer.cls_token_id = tokenizer.vocab.get(tokenizer.cls_token_str, 1)
        tokenizer.eos_token_id = tokenizer.vocab.get(tokenizer.eos_token_str, 2)
        tokenizer.special_1_token_id = tokenizer.vocab.get(tokenizer.special_1_token_str, 3)
        tokenizer.special_2_token_id = tokenizer.vocab.get(tokenizer.special_2_token_str, 4)
        tokenizer.unk_token_id = tokenizer.vocab.get(tokenizer.unk_token_str, len(vocab_list))
        
        return tokenizer


class DNATokenizer(PreTrainedTokenizer):

    vocab_list = ["<pad>", "<cls>", "<eos>", "A", "T", "C", "G", "N"]
    pad_token_str = "<pad>"
    cls_token_str = "<cls>"
    eos_token_str = "<eos>"
    unk_token_str = "<unk>"
    
    def __init__(self, vocab=None, **kwargs):
        default_vocab = self.vocab_list if vocab is None else vocab
        self.vocab = {v: i for i, v in enumerate(default_vocab)}
        self.inv_vocab = {i: v for i, v in enumerate(default_vocab)}
        

        pad_token = AddedToken(self.pad_token_str, lstrip=False, rstrip=False)
        cls_token = AddedToken(self.cls_token_str, lstrip=False, rstrip=False)
        eos_token = AddedToken(self.eos_token_str, lstrip=False, rstrip=False)
        unk_token = AddedToken(self.unk_token_str, lstrip=False, rstrip=False)
        

        super().__init__(
            pad_token=pad_token,
            cls_token=cls_token,
            eos_token=eos_token,
            unk_token=unk_token,** kwargs
        )

        self.pad_token_id = self.vocab.get(self.pad_token_str, 0)
        self.cls_token_id = self.vocab.get(self.cls_token_str, 1)
        self.eos_token_id = self.vocab.get(self.eos_token_str, 2)
        self.unk_token_id = self.vocab.get(self.unk_token_str, len(self.vocab))

        self.add_special_tokens({
            "pad_token": pad_token,
            "cls_token": cls_token,
            "eos_token": eos_token,
            "unk_token": unk_token
        })
    
    @property
    def vocab_size(self):
        return len(self.get_vocab())
    
    def _tokenize(self, text):
        return list(text)
    
    def _convert_token_to_id(self, token):
        if isinstance(token, AddedToken):
            token = token.value
        return self.vocab.get(token.upper(), 7)  
    
    def _convert_id_to_token(self, id):
        return self.inv_vocab.get(id, self.unk_token_str)
    
    def get_vocab(self):
        vocab = self.vocab.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab
    
    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(
            save_directory, 
            "dna_vocab.txt" if filename_prefix is None else f"{filename_prefix}-dna_vocab.txt"
        )
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for token in self.vocab_list:
                f.write(f"{token}\n")
            for token in self.added_tokens_encoder:
                if token not in self.vocab_list:
                    f.write(f"{token}\n")
        
        return (vocab_file,)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        vocab_file = os.path.join(pretrained_model_name_or_path, "dna_vocab.txt")
        
        if os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_list = [line.strip() for line in f if line.strip()]
        else:
            vocab_list = cls.vocab_list
        
        tokenizer = cls(vocab=vocab_list,** kwargs)
        

        tokenizer.pad_token_id = tokenizer.vocab.get(tokenizer.pad_token_str, 0)
        tokenizer.cls_token_id = tokenizer.vocab.get(tokenizer.cls_token_str, 1)
        tokenizer.eos_token_id = tokenizer.vocab.get(tokenizer.eos_token_str, 2)
        tokenizer.unk_token_id = tokenizer.vocab.get(tokenizer.unk_token_str, len(vocab_list))
        
        return tokenizer
