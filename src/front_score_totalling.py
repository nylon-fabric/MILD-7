def score_to_stars(avg):
    """
    Convert cosine similarity (-1 to 1) into a normalized 7-level scale (1–7).
    Parameters:
        avg (float): Average cosine similarity score.
    Returns:
        int: Star rating between 1 and 7.
"""
    
    # コサイン類似を正規化（0~1）
    normalized = (avg + 1) / 2
    
    # 7段階評価へ変換
    stars = int(normalized * 6) + 1
    # 安全処理
    return max(1, min(7, stars))


def get_peak(entries):
    """
    各項目のスコア最大値のワードリストを返す
    
    :param entries: 対象のkeyとラベル
    """
    peak = max(entries, key=lambda x: x[1])
    
    return float(peak[1]), peak[0]


def strength_max_score(score):
    """
    最大値のスコアをレベルに応じてラベリングする
    
    :param score: 最大値スコア
    """
    thresholds = [
        (0.75, "Very Strong Signal"),
        (0.6, "Strong Signal"),
        (0.45, "Moderate Signal"),
    ]

    for threshold, label in thresholds:
        if score > threshold:
            return f"{label}"

    return "Weak Signal"

def front_score_totalling(payload, expand_payload):
    """
        Aggregate and structure scoring results for front-end visualization.
        (フロント表示用に各内容をまとめる)

        For each category (injunctions, emotions, drivers):
        - Converts average similarity into 7-level star scale
        - Extracts peak evidence word
        - Assigns qualitative strength label
        - Returns structured dictionary for UI rendering
    """
    
    score_stars = {}
    
    for category in ["injunctions", "emotions", "drivers"]:
        score_stars[category] = []
        
        for item in payload[category]:
            
            # ワード別辞書から該当箇所取得
            entries = expand_payload[category][item["label"]]
            peak_score, peak_ev = get_peak(entries)
            
            score_stars[category].append({
                "label" : item["label"], 
                "stars_score" : score_to_stars(item["avg_score"]),
                "avg_score" : item["avg_score"],
                "evidence" : item["evidence"],
                "max_score" : peak_score,
                "max_score_status" : strength_max_score(peak_score),
                "max_evidence" : peak_ev
            })

    return score_stars