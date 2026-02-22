from text_analyzer import text_analyzer, build_medgemma_payload
from gemmas_engine import madgemma_engine, gemma_engine
from front_score_totalling import front_score_totalling


def main(text):
    """
    Main inference pipeline for MILD-7.

    Flow:
    1. Text preprocessing and signal extraction
    2. LLM-based psychological reasoning (Gemma / MedGemma)
    3. Front-end score aggregation (7-level signal output)
    """
    # === 1. Preprocessing MiniML(前処理) ===
    # Generate prompts for LLM inference and extract structured signal candidates
    # MedGemma・Gemmaに投げる文字列プロンプトと会話ピックアップリスト（dct）を作成
    gemma_prompt, payload, expand_payload = text_analyzer(text)
    
    
    # === 2. LLM Inference(推論) ===
    # Gemma: psychological reasoning (structured interpretation)
    # Gemma推論
    gemma_result = gemma_engine(gemma_prompt)
    
    # # MedGemma: contextual clinical-style interpretation
    # MedGemma推論
    medgemma_prompt = build_medgemma_payload(text, expand_payload)
    medgemma_result = madgemma_engine(medgemma_prompt)
    
    
    # === 3. Front-end Scoring(フロント表示用スコア集計) ===
    # Aggregate cosine-based scores into a 7-level signal representation
    front_score = front_score_totalling(payload, expand_payload)
    
    return medgemma_result, gemma_result, front_score

