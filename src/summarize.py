# 텍스트 표 요약

#2-1 API 키 불러오기
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API 키가 없습니다. 한번 더 확인 부탁드립니다.")
#===============================================================================================
#2-2 요약 체인 구성
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM


def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    텍스트 및 표 데이터를 요약하여 검색에 활용할 수 있는 요약본을 생성합니다.
    texts: 텍스트 리스트
    tables: 표 리스트
    summarize_texts: 텍스트 요약을 활성화할지 여부를 결정하는 불리언 값
    """

    # Prompt 한국어 버전
    prompt_text_kor = """당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미입니다.
    이 요약본들은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것입니다.
    주어진 표나 텍스트의 내용을 검색에 최적화된 간결한 요약으로 작성해주세요.
    요약할 표 또는 텍스트: {element}"""

    prompt = ChatPromptTemplate.from_template(prompt_text_kor)

    # 모델: GPT-4o-mini or Llama3.1 모델 사용
    # llamaModel = OllamaLLM(model="llama3.1:8b")
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    # 요약 체인 = 프롬프트 + LLM + 출력 파서
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []
    table_summaries = []

    # 텍스트 요약을 활성화한 경우
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    # 텍스트 요약을 사용하지 않는 경우
    elif texts:
        text_summaries = texts

    # 테이블 데이터 요약
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    return text_summaries, table_summaries


text_summaries, table_summaries = generate_text_summaries(
    texts_2k_token, tables, summarize_texts=True
)
print(text_summaries)
print(table_summaries)

"""
max_concurrency=5는 한 번에 5개씩 동시에 요약하라는 뜻입니다. 숫자를 올리면 빨라지지만 API 요청 제한에 걸릴 수 있습니다. """
#======================================================================================================================================
#2-3 체인 구조 설명
{"element": lambda x: x}  →  prompt  →  model  →  StrOutputParser()
#    입력 포장              템플릿 적용    LLM 호출      문자열로 꺼내기

#======================================================================================================================================
#2-4 프롬프트 언어 선택
