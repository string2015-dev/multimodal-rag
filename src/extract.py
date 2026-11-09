# =====================================================================
# 멀티모달 RAG ① 데이터 추출 & 분할 전체 흐름
#   PDF ─▶ Unstructured(hi_res) ─▶ 텍스트/표/이미지 분리
#       ─▶ 텍스트 재분할(토큰 기준) ─▶ pickle 저장
# =====================================================================
import os, time, pickle
from pathlib import Path
from collections import Counter
from langchain_text_splitters import CharacterTextSplitter    # 텍스트 자르는 '작두'
from unstructured.partition.pdf import partition_pdf          # PDF 자동 분리수거 기계

# ---------------------------------------------------------------------
# [1] 경로 설정 ─ 한글·공백 없는 경로 사용 (unstructured 오류 방지)
# ---------------------------------------------------------------------
PROJECT_ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents] if(p / ".env").exists()), Path.cwd(),
)
fname = "invest.pdf"                                    # 처리할 PDF 파일명
fpath = str(PROJECT_ROOT / "data") + os.sep             # PDF가 있는 폴더 (끝에 구분자 포함)

# ---------------------------------------------------------------------
# [2] 추출 함수 ─ PDF → Element 리스트
# ---------------------------------------------------------------------
def extract_pdf_elements(path, fname):
    return partition_pdf(
        filename=path + fname,
        strategy="hi_res",                 # 레이아웃 모델 기반 정밀 추출 (느리지만 표·이미지 정확)
        extract_images_in_pdf=True,        # 그림을 figures/ 에 jpg로 저장
        infer_table_structure=True,        # 표의 행·열 구조까지 추론
        chunking_strategy="by_title",      # 제목 기준으로 의미 단위 묶기
        max_characters=4000,               # 블록 하드 상한
        new_after_n_chars=3800,            # 이 길이를 넘으면 새 블록 시작
        combine_text_under_n_chars=2000,   # 짧은 조각은 앞뒤와 결합
        extract_image_block_output_dir=path + "/figures",
    )

# ---------------------------------------------------------------------
# [3] 요소 분류 ─ 텍스트 / 표로 나누기
# ---------------------------------------------------------------------
def categorize_elements(raw_pdf_elements):
    tables, texts = [], []
    for element in raw_pdf_elements:
        t = str(type(element))
        if "unstructured.documents.elements.Table" in t:
            tables.append(str(element))                  # 표 → tables
        elif "unstructured.documents.elements.CompositeElement" in t:
            texts.append(str(element))                   # 텍스트 덩어리 → texts
    return texts, tables

# ---------------------------------------------------------------------
# [4] 추출 실행 (hi_res는 2~3분 소요) + 분류
# ---------------------------------------------------------------------
if __name__ == "__main__":
    assert os.path.exists(fpath + fname), "PDF 경로를 확인하세요"   # 없으면 이후 단계 전부 실패

    start = time.time()
    raw_pdf_elements = extract_pdf_elements(fpath, fname)
    print(f"추출 {time.time()-start:.1f}초 / 전체 요소 {len(raw_pdf_elements)}개")

    texts, tables = categorize_elements(raw_pdf_elements)
    print("텍스트:", len(texts), "/ 표:", len(tables))

    # 요소 타입별 집계 ─ 파싱 품질을 눈으로 점검
    for etype, cnt in Counter(type(el).__name__ for el in raw_pdf_elements).most_common():
        print(f"{etype:20s}: {cnt}개")

    # ---------------------------------------------------------------------
    # [5] 이미지 확인 ─ figures/ 에 저장된 그림을 직접 눈으로 검증
    # ---------------------------------------------------------------------
    figures_dir = Path(fpath) / "figures"
    image_files = sorted(f for f in figures_dir.iterdir()
                         if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print("추출된 이미지:", len(image_files), "개")

    # ---------------------------------------------------------------------
    # [6] 텍스트 재분할 ─ 토큰 기준 2000, 앞뒤 200 겹침 (문맥 끊김 방지)
    # ---------------------------------------------------------------------
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,
        chunk_overlap=200,
    )
    joined_texts = " ".join(texts)                           # 블록을 하나로 합친 뒤
    texts_2k_token = text_splitter.split_text(joined_texts)  # 다시 균일한 크기로 분할
    print("분할 후 블록 수:", len(texts_2k_token))

    # ---------------------------------------------------------------------
    # [7] 저장 ─ 다음 단계(요약·임베딩)에서 재사용 (재추출 3분 절약)
    # ---------------------------------------------------------------------
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "extracted.pkl", "wb") as f:
        pickle.dump({"texts": texts, "tables": tables, "texts_2k_token": texts_2k_token}, f)

