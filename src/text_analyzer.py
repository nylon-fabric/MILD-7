from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import pickle
import os
import pysbd
import gc
import torch
from settings import mnilm_url
from constants.injunctions_permissions import INJUNCTIONS_DB, PERMISSIONS_DB, EMOTIONS_DB, DRIVERS_DB# DB

# キャッシュファイルパスの定義
BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "reference_embeddings_cache.pkl")

def get_reference_embeddings(model):
    """
    MiniMLのベクトル化処理。
    処理後、ファイルに保存し次回以降、ファイル参照できるようにする。
    
    :param model: MiniMLモデル
    :return: ベクトル化処理した参照項目内容
    """

    # 保存先フォルダがなければ作成
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"Created directory: {CACHE_DIR}")
    
    # キャッシュが存在するか確認
    if os.path.exists(CACHE_FILE):
        
        with open(CACHE_FILE, "rb") as f:
            print("Loading reference embeddings from cache...")
            return pickle.load(f)
        
    # キャッシュが存在しない場合、生成して保存
    print("Cache not found. Encoding reference databases")
    
    # MiniMLを使用しベクトル化
    refs = {
        "inj_per": build_reference_embeddings_inj_per(
            model,  
            INJUNCTIONS_DB,
            PERMISSIONS_DB 
        ),
        "emotions": build_reference_embeddings(
            model,  
            EMOTIONS_DB,
            "emotions"
        ),
        "drivers": build_reference_embeddings(
            model, 
            DRIVERS_DB, 
            "drivers"
        )
    }
    # 次回のため保存
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(refs, f)
        
    return refs

def build_reference_embeddings_inj_per(model, injunctions_db, permissions_db, lang="en"):
    """
    参照項目となる禁止令・許可文を MiniML埋め込みモデルでベクトル化
    
    :param model: MiniLM　埋め込みモデル
    :param injunctions_db: 禁止令
    :param permissions_db: 許可文
    :param lang: 対応言語
    """
    ref_texts = {}
    for key in injunctions_db.keys():
        ref_texts[key] = {
            "injunction": injunctions_db[key][lang],
            "permission": permissions_db[key][lang]
        }

    embeddings = {
        key: {
            "injunction": model.encode(val["injunction"]),
            "permission": model.encode(val["permission"])
        }
        for key, val in ref_texts.items()
    }
    # メモリ開放
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    return embeddings


def build_reference_embeddings(model, db, target_name, lang="en"):
    """
    参照項目となる感情、ドライバーをベクトル化
    
    :param model: MiniLM埋め込みモデル
    :param db: 対象データ
    :param target_name: 対象名
    :param lang: 対応言語
    """
    ref_texts = {}
    for key in db.keys():
        ref_texts[key] = {
            target_name: db[key][lang]
        }

    embeddings = {
        key: {
            target_name: model.encode(val[target_name])
        }
        for key, val in ref_texts.items()
    }
    # メモリ開放
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    return embeddings


def score_injunctions(sentence_embedding, ref_embeddings):
    """
        Compute injunction-based psychological tension score.
        (命令文に基づく心理的緊張スコアを計算)

        :param sentence_embedding: テキスト
        :param ref_embeddings: 判定基準禁止令
        :return: スコア集計
    """


    scores = {}
    for key, emb in ref_embeddings.items():
        
        inj_sim = cosine_similarity(
            [sentence_embedding], [emb["injunction"]]
        )[0][0]

        perm_sim = cosine_similarity(
            [sentence_embedding], [emb["permission"]]
        )[0][0]

        score = inj_sim - perm_sim
        scores[key] = score
    return scores


def score_other(sentence_embedding, ref_embeddings, target_name):
    """
    スコア計算
    禁止令以外で文ごとに「スコア」を出す
    
    :param sentence_embedding: テキスト
    :param ref_embeddings: 判定基準文字
    :return: スコア集計結果
    :rtype: Any
    """
    scores = {}
    for key, emb in ref_embeddings.items():
        
        score = cosine_similarity(
            [sentence_embedding], [emb[target_name]]
        )[0][0]

        scores[key] = score
    return scores


def analyze_psychological_feature_inj(
    sentences,
    sentence_embeddings,
    ref_embeddings,
    top_k=5,
    threshold=0.10
):
    """
        Aggregate sentence-level injunction tension scores into
        document-level psychological signals.

        Process:
        1. For each sentence, compute injunction tension scores.
        2. Identify the strongest response in that sentence.
        3. Select signals that:
        - Exceed the absolute threshold.
        - Are within 90% of the sentence's strongest activation.
        4. Aggregate selected scores across sentences.
        5. Track:
        - Total score (signal intensity across document)
        - Average score (consistency)
        - Maximum score (peak activation)
        - Evidence sentences

        Rationale:
        This design filters weak noise while preserving
        dominant psychological reactions per sentence,
        capturing both intensity and recurrence.
    """

        
    # 結果格納辞書
    aggregated = {k: 0.0 for k in ref_embeddings} 
    evidence = {key: [] for key in ref_embeddings.keys()}
    max_values = {k: 0.0 for k in ref_embeddings}# 最大値値をフロント表示用に別集計
    counts = {k: 0 for k in ref_embeddings} # ヒット回数をカウント
    
    
    for sent, emb in zip(sentences, sentence_embeddings):
        scores = score_injunctions(emb, ref_embeddings) 
        # もっとも反応が強かったスコアを特定
        max_scores = max(scores.values())
        
        for k, v in scores.items(): 
            
            # 項目ごとの過去最高値を更新
            if v > max_values[k]:
                max_values[k] = float(v)
            
            # 確率しきい値
            if v > threshold and v >= (max_scores * 0.9):
            # if v > threshold and (max_scores - v) < 0.08: # 距離ベース（差分）で判定
                aggregated[k] += v
                evidence[k].append([sent, float(v)])
                counts[k] += 1
    
    # ラベル、合計、平均、最大をリスト化
    ranked_list = []
    for k, total_v in aggregated.items():
        if total_v > 0:
            
            # 平均値を計算(ゼロ除算を避ける)
            if counts[k] > 0:
                avg_v = total_v / counts[k] 
            else:
                avg_v = 0.0
                
            # 最大値を計算
            max_v = max_values[k]
            
            # ４つの情報をセットにしてリスト化
            ranked_list.append([
                k, 
                float(total_v), 
                float(avg_v), 
                float(max_v)
            ])
    # 合計値に元図いて高い順に並べ替え、上位top_k個を取る
    ranked = sorted(
        ranked_list,
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    return ranked, evidence


def analyze_psychological_feature(
    sentences,
    sentence_embeddings,
    ref_embeddings,
    target_name,
    top_k=5,
    threshold=0.10
):
    """
        Aggregate sentence-level similarity scores into
        document-level psychological features.

        Process:
        1. Compute cosine similarity for each sentence.
        2. Identify the strongest response per sentence.
        3. Select features that:
        - Exceed the threshold.
        - Are within 90% of the maximum sentence activation.
        4. Aggregate scores across sentences.
        5. Compute total, average, and maximum values.

        Rationale:
        This ensures only dominant emotional/driver signals
        are retained while minimizing cross-signal noise.
    """
        
    # 結果格納辞書
    aggregated = {k: 0.0 for k in ref_embeddings} 
    evidence = {key: [] for key in ref_embeddings.keys()}
    max_values = {k: 0.0 for k in ref_embeddings} # マックス値をフロント表示用に別集計
    counts = {k: 0 for k in ref_embeddings} # ヒット回数をカウント
    
    for sent, emb in zip(sentences, sentence_embeddings):
        scores = score_other(emb, ref_embeddings, target_name) 
        # もっとも反応が強かったスコアを特定
        max_scores = max(scores.values())
        
        # 確率しきい値
        for k, v in scores.items(): 
            
            if v > max_values[k]:
                max_values[k] = float(v)
            
            if v > threshold and v >= (max_scores * 0.9):
            # if v > threshold and (max_scores - v) < 0.08: # 距離ベース（差分）で判定
                aggregated[k] += v
                evidence[k].append([ sent, float(v)])
                counts[k] += 1
    
    # ラベル、合計、平均、最大をリスト化
    ranked_list = []
    for k, total_v in aggregated.items():
        if total_v > 0:
            
            # 平均値を計算(ゼロ除算を避ける)
            if counts[k] > 0:
                avg_v = total_v / counts[k] 
            else:
                avg_v = 0.0
                
            # 最大値を計算
            max_v = max_values[k]
            
            # ４つの情報をセットにしてリスト化
            ranked_list.append([
                k, 
                float(total_v), 
                float(avg_v), 
                float(max_v)
            ])
    # 合計値に元図いて高い順に並べ替え、上位top_k個を取る
    ranked = sorted(
        ranked_list,
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    return ranked, evidence


def dct_pack(
    ranked_inj, evidence_inj,
    ranked_emo, evidence_emo,
    ranked_drv, evidence_drv
):
    """
    対象のデータを辞書化して返す関数
    
    :return: 説明
    :rtype: tuple[str, Any]
    """
    
    def pack(ranked, evidence):
        """
        Medgemmaに渡せるJson配列を作成する
        
        :param ranked: ランキング
        :param evidence: 証拠辞書
        """
        out = []

        for k, total, avg, max_val in ranked:
            out.append({
                "label":k,
                "total_score" : total, # 合計スコア
                "avg_score" : avg, # 平均スコア
                "max_score": max_val, # 最大値
                "evidence": [s for s, _ in evidence[k][:3]] # 上位3までを渡す
            })
        return out
    
    return {
        "injunctions":pack(ranked_inj, evidence_inj),
        "emotions":pack(ranked_emo, evidence_emo),
        "drivers":pack(ranked_drv, evidence_drv)
            }


def expand_from_payload(payload, sentences, w=1):
    """
    ヒットした箇所の前後の文も含めて取得する
    
    :param payload: 対象箇所
    :param sentences: 節ごとに分けた会話全文
    :param w:前後何行取るか
    """

    for items_list in payload.values():
        for items in items_list:
        
            # 対象箇所の前後行を取得
            if items[0] in sentences:
                i = sentences.index(items[0])
                start = max(0, i - w)
                end = min(len(sentences), i + w + 1)
                ctx =" ". join(sentences[start : end])
            else:
                ctx = items[0]
                    
            items.append(ctx)
    return payload

def build_gemma_payload(
    text,
    payload
):
    # フロント用に作成した辞書なので余計なワードkeyを取り除く
    new_payload = {
        category : [
            {
                "label" : item["label"], 
                "avg_score" : item["avg_score"],
                "max_score" : item["max_score"]
            }
            for item in payload[category]
        ]
        for category in ["injunctions", "emotions", "drivers"]
    }
    """
    対象項目のJsonファイルを作成し、medgemma用のメッセージを作成する
    
    :param text: 元の会話全文
    :param ranked_inj: ラベルと集計スコア
    :param evidence_inj: 根拠とその分のスコア
    :payload: 会話の中から対象となる内容のピックアップ辞書

    """
    # Gemma単体でカウンセリングさせる予定指示内容
    # gemma = f"""
    #     あなたは心理士として、下記を分析する必要があります。
    #     提示した内容の会話の矛盾点。
    #     その会話から何が得でその行動を無意識にしているのか。
    #     対人依存と承認依存。
        
    #     下記会話は問題がありそうなlabel項目とその会話箇所をピックアップした参考情報です。
    #     スコアはその会話箇所がlabelに該当していそうな確率目安です。
    #     対人依存と承認依存に関係するか所は誤ったものも該当しているので気をつけて分析してください。
    #     数値はあくまで参考情報であり、必ず原文（会話全文）との整合性を検討してください。
    #     また最後に下記点数を書いてください。
    #     - 対人依存：0〜2で評価
    #     - 承認依存：0〜2で評価
    #     - 矛盾点の強度：0〜2

    #     ### Input Data (MiniLM Signals):
    #     {json.dumps(new_payload, ensure_ascii=False, separators=(',', ':'))}
        
    #     下記は会話全文です。
    #     === Conversation ===
    #     {text}
        
    #     回答は箇条書きで返してください。
    #     出力文字に制限があります。
    #     500文字以内でまとめて回答してください。
    #     出力には【結論のみ】を書き、思考過程・前提整理・要約・言い換えは一切出力しないでください。
    # """
    gemma = f"""
        You are a psychologist and must analyze the following:

        - Contradictions within the conversation.
        - What psychological gain may unconsciously maintain the behavior.
        - Interpersonal dependency and approval dependency.

        The following conversation highlights potentially problematic label items
        and their corresponding excerpts.
        The scores indicate approximate likelihood of relevance.

        Some excerpts related to interpersonal and approval dependency may be incorrect.
        Analyze carefully.
        The numerical values are reference information only.
        Always verify consistency with the original full conversation.

        At the end, provide the following scores:
        - Interpersonal Dependency: 0–2
        - Approval Dependency: 0–2
        - Strength of Contradictions: 0–2

        ### Input Data (MiniLM Signals):
        {json.dumps(new_payload, ensure_ascii=False, separators=(',', ':'))}

        Below is the full conversation.
        === Conversation ===
        {text}

        Respond in bullet points.
        There is a character limit.
        Summarize within 500 characters.

        Output ONLY the conclusion.
        Do NOT output reasoning process, assumption整理, summary, or paraphrasing.
    """


    return gemma


def build_medgemma_payload(
    text,
    payload
):
    # MedGemme向け臨床リスク評価役用
    # medgemma_payload = f"""
    #     あなたは臨床相談テキストのリスク評価アシスタントです。
    #     心理治療解釈ではなく、相談内容に含まれる心理的・臨床的リスク信号を抽出してください。
    #     明示的表現がなくても、間接的示唆がある場合は中以上とする。
    #     保守的に過小評価しないこと。
    #     600トークン以内で回答してください。
    #     出力には【結論のみ】を書き、思考過程・前提整理・要約・言い換え・タスク分解・入力データの確認は一切出力しないこと。
    #     下記を判断してください。
    #     1. 孤立リスク
    #     2. 抑うつ兆候
    #     3. 対人関係機能低下
    #     4. 社会機能リスク
    #     5. 支援介入必要度（低・中・高）
    #     6. 下記、レッドフラグの検出（明示的表現あり・間接示唆あり・なし）
    #         - 自傷示唆
    #         - 希死念慮表現
    #         - 機能停止
    #         - 極端絶望語
    #     各項目リストについて：
    #     - リスクレベル（低・中・高）
    #     - 根拠となる文書パターン
    #     - 特徴量との関連

    #     臨床力動解釈や幼少期推測は、行わないでください。
    #     臨床相談リスク評価のみ行ってください。
    #     出力には【結論のみ】を書き、思考過程・前提整理・要約・言い換え・タスク分解・入力データの確認は一切出力しないこと。
    #     600トークン以内で回答してください。

    #     下記は会話要約です。
    #     === Input Data ===
    #     {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}
    #     ====================
        
    #     下記は会話全文を見た心理士の意見です。
    #     === Conversation ===
    #     {text}
    #     ====================
    #     出力には【結論のみ】を書き、思考過程・前提整理・要約・言い換え・タスク分解・入力データの確認は一切出力しないこと。600トークン以内で回答してください。
    # """
    medgemma_payload = f"""
        You are a risk assessment assistant for clinical consultation texts.
        Do NOT perform psychodynamic interpretation.
        Extract only psychological and clinical risk signals contained in the consultation.

        Even if there is no explicit expression, if there is indirect implication,
        evaluate it as at least Moderate.
        Do NOT conservatively underestimate risk.

        Respond within 600 tokens.
        Output ONLY the conclusion.
        Do NOT output reasoning process, assumption整理, summary, paraphrasing,
        task decomposition, or input confirmation.

        Please assess the following:
        1. Isolation Risk
        2. Depressive Signs
        3. Interpersonal Function Decline
        4. Social Function Risk
        5. Need for Support Intervention (Low / Moderate / High)
        6. Red Flag Detection (Explicit expression / Indirect implication / None)
        - Self-harm indication
        - Suicidal ideation expression
        - Functional shutdown
        - Extreme hopeless language

        For each item:
        - Risk level (Low / Moderate / High)
        - Supporting textual pattern
        - Relation to extracted features

        Do NOT perform clinical dynamic interpretation or childhood speculation.
        Perform ONLY clinical consultation risk assessment.
        Output in bullet points.

        Output ONLY the conclusion.
        Respond within 600 tokens.

        Below is the conversation summary.
        === Input Data ===
        {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}
        ====================

        Below is the psychologist’s opinion after reviewing the full conversation.
        === Conversation ===
        {text}
        ====================

        Output ONLY the conclusion.
        Respond within 600 tokens.
"""

    return medgemma_payload


def text_analyzer(text):
    """
    text_analyzer の メイン処理
    
    :param text: 分析対象会話
    """
    # 念のため実行前に他月間ているメモリを削除
    gc.collect()
    torch.cuda.empty_cache()
    
    # MniLMモデル定義
    model = SentenceTransformer(
        mnilm_url,
        local_files_only=True,
        device="cpu"
    )
    # 文の節分割処理
    segmenter = pysbd.Segmenter(language="en", clean=False)
    sentences = segmenter.segment(text)
    
    # === 文のベクトル化処理 ===
    # 会話文エンコード
    sentence_embeddings = model.encode(sentences)
    
    # ベクトル化した参照データの取得
    ref_embeddings = get_reference_embeddings(model)
    
    # メモリ開放
    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # === 会話と定義DB内容との比較処理 ===
    # 会話文と禁止令の処理
    ranked_inj, evidence_inj = analyze_psychological_feature_inj(
        sentences,
        sentence_embeddings,
        ref_embeddings["inj_per"]
    )
    # 会話文と感情の処理
    ranked_emo, evidence_emo = analyze_psychological_feature(
        sentences,
        sentence_embeddings,
        ref_embeddings["emotions"], 
        "emotions"
    )
    # 会話文とドライバーの処理
    ranked_drv, evidence_drv = analyze_psychological_feature(
        sentences,
        sentence_embeddings,
        ref_embeddings["drivers"], 
        "drivers"
    )
    # 上記のデータをフロント表示用辞書へまとめる
    payload = dct_pack(
        ranked_inj, evidence_inj,
        ranked_emo, evidence_emo,
        ranked_drv, evidence_drv
    )
    
    # 辞書にまとめた該当箇所の前後の文も含めたものを作成
    expand_payload = {
        "injunctions":expand_from_payload(evidence_inj, sentences),
        "emotions":expand_from_payload(evidence_emo, sentences),
        "drivers":expand_from_payload(evidence_drv, sentences)
    }
    
    # === Gemma用会話を作成する処理 ===
    gemma_prompt = build_gemma_payload(
    text,
    payload
    )
    
    return gemma_prompt, payload, expand_payload
