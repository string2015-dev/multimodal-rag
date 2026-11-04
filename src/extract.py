# 추출 함수 만들기

#1-1
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf


def extract_pdf_elements(path, fname):
    """
    PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출합니다.
    path: 파일 경로 (추출된 이미지 파일이 저장될 위치)
    fname: 파일 이름
    """
    return partition_pdf(
        filename=path + fname,
        strategy="hi_res",                # 레이아웃 분석 모델로 정밀 추출 (교재에 없음! 필수 추가)
        extract_images_in_pdf=True,       # PDF에서 이미지를 추출
        infer_table_structure=True,       # 테이블 구조를 추론
        chunking_strategy="by_title",     # 타이틀을 기준으로 텍스트를 블록으로 분할
        max_characters=4000,              # 최대 4000자로 텍스트 블록을 제한
        new_after_n_chars=3800,           # 3800자 이후에 새로운 블록 생성
        combine_text_under_n_chars=2000,  # 2000자 이하의 텍스트는 결합
        image_output_dir_path=path,       # 이미지가 저장될 경로 설정
    )

#1-2 추출한 요소를 종류별로 분류
def categorize_elements(raw_pdf_elements):
    """
    PDF에서 추출한 요소들을 테이블과 텍스트로 분류합니다.
    raw_pdf_elements: unstructured.documents.elements 리스트
    """
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))   # 테이블 요소를 저장
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))    # 텍스트 요소를 저장
    return texts, tables
#=============================================================================================
# 1-3 경로 설정 및 실행
import os

# 현재 작업 디렉터리 경로를 반환합니다.
current_directory = os.getcwd()

# fname: 처리할 PDF 파일 이름 / fpath: PDF가 저장된 폴더 경로
fname = "invest.pdf"
fpath = os.path.join(os.path.dirname(current_directory), "invest/")

print("현재 스크립트의 위치:", current_directory)
print("pdf 위치:", fpath)

# 실행하면 2~3분 걸립니다. 멈춘 게 아니니 기다리세요.
raw_pdf_elements = extract_pdf_elements(fpath, fname)
texts, tables = categorize_elements(raw_pdf_elements)


#=============================================================================================
#1-4 텍스트 분할
# chunk_size=2000: 각 텍스트 블록의 최대 크기
# chunk_overlap=200: 청크 간 200자의 겹침을 허용해 중요한 정보를 잃지 않도록 함
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2000, chunk_overlap=200
)

# 텍스트 리스트를 하나의 긴 문자열로 결합한 뒤 2000자 단위로 분할
joined_texts = " ".join(texts)
texts_2k_token = text_splitter.split_text(joined_texts)

print("분할 후 블록 수:", len(texts_2k_token))
print("원본 블록 수:", len(texts))
print(texts[0])
print(type(texts_2k_token))

#=============================================================================================
#1-5
