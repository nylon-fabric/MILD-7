from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import torch
import time
from settings import medgemma_url, gemma_url

def madgemma_engine(prompt):
    url = medgemma_url
    
    # プロンプト指示作成
    messages = [
        {"role" : "user", "content" : prompt},
    ]
    
    max_new_tokens = 1000 # 長文説明
    
    result = make_model(url, messages, max_new_tokens)
    return result


def gemma_engine(prompt):
    
    # content = "あなたは交流分析の専門家です。"
    content = "You are an expert in Transactional Analysis."

    url = gemma_url
    
    # プロンプト指示作成
    messages = [
        {"role" : "system", "content" : content}, 
        {"role" : "user", "content" : prompt},
    ]
    max_new_tokens = 1000 # マックストークン
    
    # モデル定義と推論
    result = make_model(url, messages, max_new_tokens)
    
    return result


def make_model(url, messages, max_new_tokens):
    """
    Core LLM inference function.

    Parameters:
        url (str): Local path to the pretrained model.
        messages (list): Chat-formatted messages for generation.
        max_new_tokens (int): Maximum number of generated tokens.

    Returns:
        str: Generated text response.
    """
    # === Model Loading ===
    # モデルロード設定
    model = AutoModelForCausalLM.from_pretrained(
        url,
        device_map="auto",
        max_memory={0 : "5.5GiB", "cpu" : "16GiB"},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        local_files_only=True
    )
    # MedGemma 推奨の chat template を使ってトークン化
    tokenizer = AutoTokenizer.from_pretrained(
            url,
            local_files_only=True,
        )
    
    # === Tokenization using chat template ===
    # 生成設定
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True, # AIの回答はここからという目印
        return_tensors="pt", # PyTorch形式で返す
        # truncation=True, #
        # max_length=300, # 
        return_dict=True
    ).to(model.device) # AIが読める形に変換
    
    # === Text Generation ===
    # AIに文章を生成させる
    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id # 確実に終了判定させる
        )
    
    # === Decode output ===
    # 人間が読める文章に戻す
    input_len = input_ids["input_ids"].shape[-1] # [input_ids.shape[-1]:]で入力したプロンプト省略
    result = tokenizer.decode(
        outputs[0][input_len:], 
        skip_special_tokens=True
    )
    
    # === Memory cleanup ===
    # テンソル削除
    del outputs
    del input_ids
    del model
    
    # 解放
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.ipc_collect()
    
    # Small delay to stabilize GPU memory between consecutive inferences
    # 数秒スリープを入れる（次のメモリ管理安定のため）
    time.sleep(2)
    
    return result