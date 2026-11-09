# =====================================================================
# 멀티모달 RAG ② 요약 생성 전체 흐름
#   extracted.pkl ─▶ 텍스트·표 요약(LLM) ──────────────┐
#   figures/*.jpg ─▶ Base64 ─▶ 이미지 요약(VLM) ───────┼▶ 검증·토큰 측정 ─▶ summaries.pkl
# =====================================================================
import os, time, pickle, base64
from pathlib import Path
from dotenv import load_dotenv
import tiktoken
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------
# [1] 준비 ─ 경로(상수) 설정 + API 키 로드
# ---------------------------------------------------------------------
PROJECT_ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".env").exists()), Path.cwd(),
)                                        # .env 가 있는 가장 가까운 상위 폴더 (노트북 notebooks/ 에서도 동작)
MAX_CONCURRENCY = 2                      # 대문자 = 상수. 동시 요청 수 (RateLimitError 나면 더 낮추기)

# ---------------------------------------------------------------------
# [3] 텍스트·표 요약 ─ LCEL 체인(컨베이어 벨트) + batch
# ---------------------------------------------------------------------
def generate_text_summaries(texts, tables, summarize_texts=False):
    # 영어 프롬프트 + 마지막 줄에 한국어 출력 지시. {element} 자리에 원문이 들어감
    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval.
These summaries will be embedded and used to retrieve the raw text or table elements.
Give a concise summary of the table or text that is well optimized for retrieval.
Table or text: {element}
The summary should be written in Korean (Hangul)."""

    prompt = ChatPromptTemplate.from_template(prompt_text)     # 템플릿 적용 칸
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")     # temperature=0 → 일관된 요약

    # 입력 포장 → 템플릿 → LLM 호출 → 문자열로 꺼내기
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries, table_summaries = [], []
    if texts and summarize_texts:        # 텍스트도 요약하는 경우
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": MAX_CONCURRENCY})
    elif texts:                          # 요약 안 하면 원문 그대로 사용
        text_summaries = texts
    if tables:                           # 표는 있으면 항상 요약
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": MAX_CONCURRENCY})
    return text_summaries, table_summaries

# ---------------------------------------------------------------------
# [4] 이미지 → Base64 ─ API는 문자열만 받으므로 그림을 '문자열 상자'에 포장
# ---------------------------------------------------------------------
def encode_image(image_path):
    with open(image_path, "rb") as f:                          # 바이너리로 읽기
        return base64.b64encode(f.read()).decode("utf-8")      # bytes → str

# ---------------------------------------------------------------------
# [5] 이미지 요약(VLM) ─ 텍스트 조각 + 이미지 조각을 한 메시지에 담아 호출
# ---------------------------------------------------------------------
def image_summarize(img_base64, prompt):
    chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)
    msg = chat.invoke([HumanMessage(content=[
        {"type": "text", "text": prompt},                      # 지시문
        {"type": "image_url",                                  # 이미지 (data URL 접두사 필수)
         "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
    ])])
    return msg.content

def generate_img_summaries(path):
    img_base64_list, image_summaries = [], []   # 두 리스트의 순서가 같아야 나중에 요약↔원본 짝짓기 가능

    prompt = """You are an expert AI assistant that summarizes document images (tables, charts, graphs, text) for a semantic retrieval system.
The summary will be embedded and used to retrieve this raw image.
1. Identify the document type and its main title or topic.
2. Extract key metrics, table headers, X/Y axis labels, and prominent trends or data points.
3. Capture any critical conclusions or text visible in the image.
4. Do not start with generic phrases like "이 이미지는"; start directly with the core data.
5. Keep it to 3-4 dense sentences focused on searchable keywords.
6. The output MUST be written in Korean (Hangul)."""

    for image_file in sorted(os.listdir(path)):                # 이름순 정렬 → 순서 고정
        if image_file.endswith((".jpg", ".jpeg", ".png")):     # 이미지 확장자만 처리
            b64 = encode_image(os.path.join(path, image_file))
            img_base64_list.append(b64)                        # ① 원본(답변 생성 때 VLM에 다시 전달)
            image_summaries.append(image_summarize(b64, prompt))  # ② 요약(검색용)
    return img_base64_list, image_summaries

if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env")       # .env 파일의 키를 환경변수로 등록
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:                   # 키가 없으면 즉시 중단
        raise ValueError("OpenAI API 키가 없습니다. .env 파일을 확인하세요")
    print(f"API 키 로드 완료: {openai_api_key[:7]}...")   # 앞 7자만 출력 (유출 방지)

    # ---------------------------------------------------------------------
    # [2] 지난 시간 결과 복원 ─ pickle (냉동 보관 → 해동)
    # ---------------------------------------------------------------------
    with open(PROJECT_ROOT / "data" / "processed" / "extracted.pkl", "rb") as f:
        data = pickle.load(f)
    tables = data["tables"]                  # 표(문자열 리스트)
    texts_2k_token = data["texts_2k_token"]  # 토큰 기준으로 재분할한 텍스트 청크
    figures_dir = PROJECT_ROOT / "data" / "figures"   # 추출된 이미지 폴더

    start = time.time()
    text_summaries, table_summaries = generate_text_summaries(texts_2k_token, tables, summarize_texts=True)
    print(f"요약 완료 — {time.time()-start:.1f}초 / 텍스트 {len(text_summaries)} / 표 {len(table_summaries)}")

    # 압축률 점검: 요약 길이 ÷ 원본 길이 (참고: 5~20%, 작을수록 많이 줄인 것)
    orig, summ = texts_2k_token[0], text_summaries[0]
    print(f"압축률: {len(summ)/len(orig)*100:.1f}%")

    # 표 요약이 비어 있으면: ① 표가 이미지로 추출됨 ② 추출 때 hi_res 누락 → 점검
    if not table_summaries:
        print("표 요약이 비어 있습니다. strategy='hi_res'인지, 진짜 표인지 이미지인지 확인하세요.")

    start = time.time()
    img_base64_list, image_summaries = generate_img_summaries(str(figures_dir))
    print(f"이미지 요약 완료 — {time.time()-start:.1f}초 / {len(image_summaries)}장")

    # ---------------------------------------------------------------------
    # [6] 요약 품질 육안 검증 ⭐ ─ 그림과 요약을 나란히 놓고 대조
    # ---------------------------------------------------------------------
    from IPython.display import Image, display
    image_files = sorted(f for f in figures_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for i, (img_file, summary) in enumerate(zip(image_files[:3], image_summaries[:3])):
        print("-=" * 40)
        print(f"[{i+1}] {img_file.name}")
        display(Image(filename=str(img_file), width=450))          # 원본 그림
        print("요약:", summary)                                     # 나쁨 / 보통 / 좋음 판단

    # ---------------------------------------------------------------------
    # [7] 토큰 측정 ─ 실제 비용 = 토큰 수 × 단가 (이미지 토큰은 미포함)
    # ---------------------------------------------------------------------
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    input_tokens = sum(len(enc.encode(t)) for t in texts_2k_token)
    output_tokens = sum(len(enc.encode(s)) for s in text_summaries)
    print(f"입력 {input_tokens:,} / 출력 {output_tokens:,} / 합계 {input_tokens+output_tokens:,} 토큰")

    # ---------------------------------------------------------------------
    # [8] 저장 ─ 다음 단계(벡터 저장소 구축)에서 바로 불러 쓰기 (요약 재생성 비용 절약)
    # ---------------------------------------------------------------------
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summaries.pkl", "wb") as f:
        pickle.dump({
            "text_summaries": text_summaries,     # 텍스트 요약 (검색용)
            "table_summaries": table_summaries,   # 표 요약 (검색용)
            "image_summaries": image_summaries,   # 이미지 요약 (검색용)
            "img_base64_list": img_base64_list,   # 이미지 원본 (답변용, 요약과 같은 순서)
        }, f)
    print("저장 완료:", output_dir / "summaries.pkl")