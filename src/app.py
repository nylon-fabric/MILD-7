import streamlit as st
import time
from backend import main

# =====
# 表示
# =====
st.title("MILD-7")
st.markdown("### Multi-layer Insight & Linguistic Detection – 7-Level Signal System")
st.caption("多層的洞察・言語検出 ― 7段階シグナルシステム")

# =====
# 入力欄
# =====
text = st.text_area(
    """Please paste the consulltation text
    (相談テキストを張り付けてください。)""",
    height=200
)
analyze_btn = st.button("Analyze")


# =====
# 出力
# =====
if analyze_btn and text.strip():
    
    # ロード表示
    status = st.empty()

    def loading(text, color):
        return f'<span style="color:{color}; font-weight:600;">{text}</span>'

    status.markdown(
        loading("▲ Analyzing structured signals...", "#c05621"),
        unsafe_allow_html=True
    )
    time.sleep(0.8)

    status.markdown(
        loading("✓ Analyzing structured signals...", "#2f855a") + "<br><br>" +
        loading("▲ Generating psychological insights...", "#c05621"),
        unsafe_allow_html=True
    )
    time.sleep(0.8)

    status.markdown(
        loading("✓ Analyzing structured signals...", "#2f855a") + "<br><br>" +
        loading("✓ Generating psychological insights...", "#2f855a") + "<br><br>" +
        loading("▲ Evaluating clinical risk...", "#c05621"),
        unsafe_allow_html=True
    )
    # 結果取得
    medical_assistant_result, counselor_assistant_result, scores_data = main(text)
    # 結果表示
    status.markdown(
        loading("✓ Analyzing structured signals...", "#2f855a") + "<br><br>" +
        loading("✓ Generating psychological insights...", "#2f855a") + "<br><br>" +
        loading("✓ Evaluating clinical risk...", "#2f855a"),
        unsafe_allow_html=True
    )
    
    st.subheader("Psychological Profile")
    st.caption("（MiniLM層）")
    st.caption("心理プロファイル")
    if scores_data:
        
        for lst in scores_data.values():
            for s in lst:

                with st.container():
                    
                    st.markdown(f"### {s['label']}")
                    
                    st.markdown(f"***Signal Strength (Average)***")
                    starz_html = f'<span style="color:#2b6cb0;">{"★" * s["stars_score"] + "☆" * (7 - s["stars_score"])}</span>'

                    st.markdown(starz_html, unsafe_allow_html=True)
                    
                    if s["max_score_status"]:
                        st.markdown(f"***MAX Similarety : {s['max_score']:.2f}***")
                        st.markdown(f"- Level: {s['max_score_status']} ")
                        
                        st.markdown("Key Phrase:")
                        st.markdown(f"- {s['max_evidence']}")

    else:
        st.write("None data (該当なし)")


    st.subheader("Psychological Insights (Counselor Assistant)")
    st.caption("（Gemma層）")
    st.caption("心理学的な洞察")
    if counselor_assistant_result:
        st.write(counselor_assistant_result)
    else:
        st.write("None data (該当なし)")


    st.subheader("Clinical Risk Assessment (Medical Assistant)")
    st.caption("（MedGemma層）")
    st.caption("臨床的リスクアセスメント")
    if medical_assistant_result:
        st.write(medical_assistant_result)
    else:
        st.write("None data (該当なし)")