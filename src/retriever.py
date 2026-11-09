# ============================================================
# 0. 준비: 모듈 import + 환경변수 (import 만으로는 API 호출·파일 읽기가 일어나지 않음)
# ============================================================
import os, re, time, uuid, pickle
from pathlib import Path
from dotenv import load_dotenv

# .env 가 있는 폴더를 프로젝트 루트로 자동 탐색 (현재 폴더 → 상위 폴더 순서)
PROJECT_ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".env").exists()),
    Path.cwd(),                                   # 못 찾으면 현재 폴더
)
load_dotenv(PROJECT_ROOT / ".env")                # 루트의 .env 를 정확히 지정해서 로드

processed = PROJECT_ROOT / "data" / "processed"


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY가 없습니다. .env를 확인하세요")


def load_processed():
    """앞 단계 결과 복원: extracted.pkl(추출) + summaries.pkl(요약)"""
    # 추출 결과 (partition_pdf(strategy="hi_res") 로 뽑아 둔 것)
    with open(processed / "extracted.pkl", "rb") as f:
        d1 = pickle.load(f)
    # 요약 결과 (텍스트·표·이미지 요약 + 이미지 Base64 원본)
    with open(processed / "summaries.pkl", "rb") as f:
        d2 = pickle.load(f)
    return d1, d2


# ============================================================
# 1. 다중 벡터 검색기: 서랍(vectorstore) + 서가(docstore) + 청구기호(id_key)
# ============================================================
try:    # langchain 1.x
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
    from langchain_core.stores import InMemoryStore
except ImportError:   # langchain 0.3
    from langchain.retrievers.multi_vector import MultiVectorRetriever
    from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

id_key = "doc_id"                                  # 요약과 원본을 잇는 번호표 이름


def build_retriever():
    check_api_key()
    vectorstore = Chroma(                          # 요약문 임베딩 → 검색용
        collection_name="multimodal_rag",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    vectorstore.reset_collection()                 # 같은 프로세스에서 다시 실행해도 중복 색인 방지
    store = InMemoryStore()                        # 원본 보관 → 전달용 (메모리라 재시작 시 사라짐)
    return MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)


# ============================================================
# 2. 요약과 원본에 "같은 번호표"를 붙여 저장  ← 오늘의 핵심
# ============================================================
def add_documents(retriever, doc_summaries, doc_contents):
    # 원본 1개당 고유 ID 1개 (uuid 객체 → 문자열)
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    # 요약문을 Document 로 감싸고, metadata 에 번호표를 붙인다
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)            # 요약 → 서랍 (검색)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))    # (번호표, 원본) → 서가 (전달)


# ============================================================
# 3. 검색 결과를 이미지(Base64) / 텍스트로 분리
# ============================================================
B64_PATTERN = r"^[A-Za-z0-9+/]{100,}={0,2}$"      # Base64 문자만 100자 이상 + 끝에 = 0~2개

def split_image_text_types(docs):
    b64_images, texts_out = [], []
    for doc in docs:
        doc = doc.page_content if isinstance(doc, Document) else doc   # Document 면 문자열 꺼내기
        (b64_images if re.match(B64_PATTERN, doc) else texts_out).append(doc)
    return {"images": b64_images, "texts": texts_out}


# ============================================================
# 4. 멀티모달 프롬프트 조립 + 최종 체인 (LCEL)
# ============================================================
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

INSTRUCTION = ("아래 자료를 근거로 질문에 한국어로 답하세요.\n"
               "자료에 없는 내용은 모른다고 답하세요.\n\n")

def img_prompt_func(data_dict):
    messages = []
    # 1) 이미지 먼저 → 접두사 "data:image/jpeg;base64," 를 붙여 URL 형태로
    for image in data_dict["context"]["images"]:
        messages.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
    # 2) 지시문 + 질문 + 참고 텍스트
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages.append({"type": "text",
                     "text": INSTRUCTION
                             + f"질문: {data_dict['question']}\n\n"
                             + f"참고 텍스트:\n{formatted_texts}"})
    return [HumanMessage(content=messages)]        # 이미지+텍스트가 담긴 한 개의 메시지


def build_model():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1024)   # 눈 달린 모델


def build_chain(retriever, model=None):
    model = model or build_model()
    return (
        {
            "context": retriever | RunnableLambda(split_image_text_types),  # 질문 → 검색 → 분리
            "question": RunnablePassthrough(),                              # 질문은 그대로 전달
        }
        | RunnableLambda(img_prompt_func)   # 메시지 조립
        | model                             # 답변 생성
        | StrOutputParser()                 # AIMessage → 문자열
    )


# ============================================================
# 6. Before / After: 텍스트 전용 RAG 와 비교 (조건은 이미지·표 빼고 동일)
# ============================================================
def build_text_only_chain(texts_2k_token, model=None):
    check_api_key()
    model = model or build_model()
    text_only_store = Chroma(
        collection_name="text_only_rag",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    text_only_store.reset_collection()                             # 다시 실행해도 중복 방지
    text_only_store.add_texts(texts_2k_token)                      # 텍스트 원본만 넣음
    text_retriever = text_only_store.as_retriever(search_kwargs={"k": 4})

    text_prompt = ChatPromptTemplate.from_template(
        INSTRUCTION + "질문: {question}\n\n참고 텍스트:\n{context}"
    )
    return (
        {"context": text_retriever | RunnableLambda(lambda ds: "\n".join(d.page_content for d in ds)),
         "question": RunnablePassthrough()}
        | text_prompt | model | StrOutputParser()
    )


if __name__ == "__main__":
    check_api_key()
    d1, d2 = load_processed()
    texts, tables, texts_2k_token = d1["texts"], d1["tables"], d1["texts_2k_token"]
    text_summaries  = d2["text_summaries"]
    table_summaries = d2["table_summaries"]
    image_summaries = d2["image_summaries"]
    img_base64_list = d2["img_base64_list"]

    retriever = build_retriever()
    # 데이터가 있을 때만 저장
    if text_summaries:  add_documents(retriever, text_summaries,  texts_2k_token)
    if table_summaries: add_documents(retriever, table_summaries, tables)
    if image_summaries: add_documents(retriever, image_summaries, img_base64_list)
    print("색인된 요약 수:", len(retriever.vectorstore.get()["ids"]))

    # ============================================================
    # 5. 질의 실행
    # ============================================================
    model = build_model()
    chain = build_chain(retriever, model)
    query = "그래프/표를 봐야만 답할 수 있는 질문을 적으세요"
    start = time.time()
    answer = chain.invoke(query)
    print(f"{time.time() - start:.1f}초\nQ: {query}\n{'-'*70}\n{answer}")

    text_only_chain = build_text_only_chain(texts_2k_token, model)
    print("[BEFORE] 텍스트 전용 RAG\n", text_only_chain.invoke(query))
    print("[AFTER ] 멀티모달 RAG\n",   chain.invoke(query))     # → 캡처해서 README.md 에!
