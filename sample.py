import argparse
import torch
import os
import logging
from transformers import PreTrainedTokenizerBase
from typing import Optional, List, Dict

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 导入模型相关类
from .model.model import DNAConditionalProteinModel, DNAConditionalProteinConfig, ProteinTokenizer, DNATokenizer

def load_model_and_tokenizers(model_path: str, device: str = None) -> tuple:
    """加载训练好的模型和对应的tokenizer"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"加载模型: {model_path}")
    logger.info(f"使用设备: {device}")
    
    # 加载配置
    config = DNAConditionalProteinConfig.from_pretrained(model_path)
    
    # 加载tokenizer
    protein_tokenizer = ProteinTokenizer.from_pretrained(model_path)
    dna_tokenizer = DNATokenizer.from_pretrained(model_path)
    
    # 加载模型
    model = DNAConditionalProteinModel.from_pretrained(
        model_path,
        config=config
    )
    
    # 移动模型到目标设备
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    return model, protein_tokenizer, dna_tokenizer, device

def generate_protein_variants(
    model: DNAConditionalProteinModel,
    dna_sequence: str,
    dna_tokenizer: PreTrainedTokenizerBase,
    protein_tokenizer: PreTrainedTokenizerBase,
    device: str,
    num_variants: int = 5,  # 生成的变体数量
    max_length: int = 177,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    seed: Optional[int] = None,  # 随机种子，确保可重复性
    filter_duplicates: bool = True  # 是否过滤重复序列
) -> List[Dict[str, any]]:
    """从单个DNA序列生成多个蛋白质变体"""
    # 设置随机种子，确保结果可重复
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
    
    # 预处理DNA序列
    dna_input = dna_tokenizer(
        f"<cls>{dna_sequence}<eos>",
        return_tensors="pt",
        truncation=True,
        max_length=model.config.max_dna_length
    )
    
    # 将输入移动到设备
    dna_input_ids = dna_input["input_ids"].to(device)
    dna_attention_mask = dna_input["attention_mask"].to(device)
    
    # 编码DNA（只需编码一次）
    with torch.no_grad():
        dna_features = model.encode_dna(dna_input_ids, dna_attention_mask)
        cond_features = model.feature_fusion(dna_features)
    
    variants = []
    generated_sequences = set()  # 用于存储已生成的序列，避免重复
    attempts = 0  # 记录尝试次数
    max_attempts = num_variants * 20   # 增加最大尝试次数，因为可能过滤掉很多序列
    
    # 生成多个变体
    while len(variants) < num_variants and attempts < max_attempts:
        # 计算当前温度（每20次生成增加1度）
        current_temperature = temperature + (attempts // 20)*0.5
        # 确保温度不会过低
        current_temperature = max(current_temperature, 0.1)
        
        # 每次生成使用不同的随机种子偏移，确保多样性
        if seed is not None:
            torch.manual_seed(seed + attempts*100)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed + attempts*100)
        
        # 初始化生成
        batch_size = dna_input_ids.shape[0]
        start_token = protein_tokenizer.cls_token_id  # 起始标记
        generated_ids = torch.full(
            (batch_size, 1), 
            fill_value=start_token, 
            device=device, 
            dtype=torch.long
        )
        
        # 逐步生成蛋白质序列
        for _ in range(max_length - 1):
            with torch.no_grad():
                logits, _ = model.protein_decoder(
                    input_ids=generated_ids,
                    cond_features=cond_features
                )
            
            # 使用当前温度获取最后一个token的logits
            next_token_logits = logits[:, -1, :] / current_temperature
            
            # 应用top-k和top-p采样
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(
                    next_token_logits, 
                    min(top_k, next_token_logits.size(-1))
                )
                next_token_logits[next_token_logits < top_k_values[:, [-1]]] = -float('inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 更新logits
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = -float('inf')
            
            # 计算概率分布
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # 采样下一个token
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            # 拼接结果
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            
            # 如果生成结束标记，停止
            if (next_token_id == protein_tokenizer.eos_token_id).all():
                break
        
        # 解码为蛋白质序列
        generated_sequence = protein_tokenizer.decode(
            generated_ids[0], 
            skip_special_tokens=True
        )
        
        # 检查序列长度是否小于70
        if len(generated_sequence) < 140:
            attempts += 1
            continue  # 跳过长度小于70的序列，继续生成
        
        # 检查是否需要过滤重复以及序列是否已存在
        if filter_duplicates and generated_sequence in generated_sequences:
            attempts += 2
            continue  # 跳过重复序列，继续生成
        
        # 添加新生成的序列到集合中
        if filter_duplicates:
            generated_sequences.add(generated_sequence)
        
        # 存储变体及其参数
        variants.append({
            "variant_id": len(variants) + 1,
            "protein_sequence": generated_sequence,
            "length": len(generated_sequence),
            "parameters": {
                "temperature": current_temperature,  # 记录当前使用的温度
                "base_temperature": temperature,     # 记录基础温度
                "top_k": top_k,
                "top_p": top_p,
                "seed": seed + attempts if seed is not None else None
            }
        })
        
        attempts += 1
    
    return variants

def main():
    parser = argparse.ArgumentParser(description="从单个DNA序列生成多个蛋白质变体")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--dna_sequence", type=str, required=True, help="输入的DNA序列")
    parser.add_argument("--num_variants", type=int, default=5, help="生成的蛋白质变体数量")
    parser.add_argument("--max_length", type=int, default=200, help="生成的蛋白质最大长度")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度参数，值越大生成越随机")
    parser.add_argument("--top_k", type=int, default=50, help="top-k采样参数")
    parser.add_argument("--top_p", type=float, default=0.95, help="nucleus采样参数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，确保结果可重复")
    parser.add_argument("--output_file", type=str, help="输出蛋白质序列的文件路径")
    parser.add_argument("--device", type=str, help="运行设备，如cuda或cpu")
    parser.add_argument("--filter_duplicates", action="store_true", help="是否过滤重复序列")
    
    args = parser.parse_args()
    
    # 加载模型和tokenizer
    model, protein_tokenizer, dna_tokenizer, device = load_model_and_tokenizers(
        args.model_path, 
        args.device
    )
    
    # 生成多个蛋白质变体
    logger.info(f"开始为DNA序列生成 {args.num_variants} 个蛋白质变体...")
    #logger.info(f"DNA序列: {args.dna_sequence[:100]}...")  # 只显示前100个字符
    
    variants = generate_protein_variants(
        model=model,
        dna_sequence=args.dna_sequence,
        dna_tokenizer=dna_tokenizer,
        protein_tokenizer=protein_tokenizer,
        device=device,
        num_variants=args.num_variants,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        filter_duplicates=args.filter_duplicates
    )
    
    # 输出结果
    for variant in variants:
        print(f"变体 {variant['variant_id']}:")
        #print(f"长度: {variant['length']}")
        print(f"序列: {variant['protein_sequence']}")
        print(f"参数: {variant['parameters']}")
        print("-" * 80)
    
    # 保存到文件
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(f"原始DNA序列: {args.dna_sequence}\n")
            f.write(f"生成参数: 温度={args.temperature}, top_k={args.top_k}, top_p={args.top_p}\n")
            #f.write(f"过滤重复: {args.filter_duplicates}\n")
            f.write("=" * 80 + "\n")
            
            for variant in variants:
                f.write(f"变体 {variant['variant_id']} (长度: {variant['length']}):\n")
                f.write(f"{variant['protein_sequence']}\n")
                f.write(f"种子: {variant['parameters']['seed']}\n")
                f.write("-" * 80 + "\n")
        
        logger.info(f"结果已保存到 {args.output_file}")

if __name__ == "__main__":
    main()
