#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().system('pip install bs4')
get_ipython().system('pip install requests')
get_ipython().system('pip install pandas')
get_ipython().system('pip install openpyxl')
get_ipython().system('pip install selenium')
get_ipython().system('pip install webdriver-manager')


# In[20]:


import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

# 1. 드라이버 실행
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://m.kinolights.com/ranking/total")

time.sleep(3)

data = []
seen = set()

# 2. 스크롤하면서 수집
for _ in range(4):  
    items = driver.find_elements(By.CSS_SELECTOR, 'li.ranking-item')

    for item in items:
        try:
            title = item.find_element(By.CSS_SELECTOR, 'h5.info__title span').text.strip()

            # 중복 제거
            if title in seen:
                continue
            seen.add(title)

            rank = item.find_element(By.CSS_SELECTOR,'.rank__number span').text.strip()
            link = item.find_element(By.CSS_SELECTOR, 'a.content-list-card__body').get_attribute('href')
            info = item.find_element(By.CSS_SELECTOR, 'p.info__subtitle').text.strip()
            rate = item.find_element(By.CSS_SELECTOR, '.score__number').text.strip()

            data.append([rank,title, link, info, rate])

        except:
            continue

    driver.execute_script("window.scrollBy(0, 400);")
    time.sleep(1.2)

driver.quit()

df = pd.DataFrame(data, columns=['순위','제목', '링크', '장르_연도', '긍정적평가'])

print(len(df))
print(df.head())


# In[21]:


print(df)


# In[22]:


documents = []

for _, row in df.iterrows():
    text = f"""
    순위:{row['순위']}
    제목: {row['제목']}
    장르 및 연도: {row['장르_연도']}
    긍정적평가: {row['긍정적평가']}
    링크: {row['링크']}
    """
    documents.append(text.strip())


# In[23]:


def init_db(db_path="./chroma_db"):
    dbclient = chromadb.PersistentClient(path=db_path)
    try:
        dbclient.delete_collection(name="rag_collection")
    except:
        pass 

    collection = dbclient.create_collection(name="rag_collection")
    return dbclient, collection


# In[24]:


import os # os를 가져와 파일 시스템 접근, 환경 변수 읽을 수 있음
from openai import OpenAI # OpenAI의 api 사용 가능
import chromadb # chromadb 라이브러리 쓸 수 있게 해줌
from chromadb.config import Settings # Settings 클래스는 DB의 구성 옵션을 설정하는데 사용
from dotenv import load_dotenv # 환경 변수를 로드하기 위함

# 환경 변수 Load해서 api_key 가져오고 OpenAI 클라이언트(객체) 초기화
load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# In[26]:


# 주어진 text를 임베딩 벡터로 변환하는 함수 
def get_embedding(text, model="text-embedding-3-large"):
		# 여기서 client는 앞서 초기화한 OpenAI 클라이언트
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding # 응답 객체의 data 리스트에서 embedding 필드 추출
    return embedding 


# In[27]:


# 원천 데이터 청크 단위로 나누고 overlap 사이즈 조절하는 함수
def chunk_text(text, chunk_size=400, chunk_overlap=50):
    chunks = [] # 분할된 텍스트 청크들을 저장할 리스트
    start = 0 # 청크를 시작할 위치를 나타내는 인덱스
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end] # 텍스트에서 start부터 end까지 부분 문자열을 추출
        chunks.append(chunk) # 추출한 청크를 리스트에 추가
        start = end - chunk_overlap # overlap 적용

        if start < 0: # 음수가 될 수 있으니 예외 처리
            start = 0

        if start >= len(text): # 종료 시그널
            break

    return chunks # 모든 청크가 저장된 리스트를 반환


# In[ ]:


if __name__ == "__main__":
    dbclient, collection = init_db("./chroma_db")

    all_chunks = documents  # 이미 만든 documents 사용
    all_ids = [str(i) for i in range(len(all_chunks))]

    # metadata
    all_metadatas = []
    for _, row in df.iterrows():
        all_metadatas.append({
            "rank": row['순위'],
            "title": row['제목'],
            "genre": row['장르_연도'],
            "rating": row['긍정적평가']
        })

    embeddings = [get_embedding(text) for text in all_chunks]

    # DB 저장
    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
        ids=all_ids
    )

    print("DB 저장 완료")


# In[29]:


import os
import import_ipynb
from openai import OpenAI
from build_vector_db import get_embedding
from chromadb import Client
import chromadb 
from chromadb.config import Settings 
from dotenv import load_dotenv

load_dotenv()
dbclient = chromadb.PersistentClient(path="./chroma_db")
collection = dbclient.get_or_create_collection("rag_collection")


# In[30]:


# query를 임베딩해 chroma에서 가장 유사도가 높은 top-k개의 문서 가져오는 함수 
def retrieve(query, top_k=3):
    query_embedding = get_embedding(query) # qeury에 대한 임베딩 생성
    # collection.query 함수로 저장된 문서 임베딩들 중에서
    # query임베딩과 가장 유사한 항목들 검색 
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    ) 
    # 이때 results에는 해당 query 임베딩에 대한 텍스트, 메타데이터, id등이 전부 포함됨 
    return results


# In[41]:


def generate_answer_with_context(query, top_k=20):

    results = retrieve(query, top_k)

    found_docs = results["documents"][0]
    found_metadatas = results["metadatas"][0]

    sorted_pairs = sorted(
    zip(found_docs, found_metadatas),
    key=lambda x: int(x[1].get("rank", 999)),
    )

    top_docs = sorted_pairs[:top_k]

    context_texts = []

    for doc_text, meta in top_docs:
        context_texts.append(
            f"""
순위: {meta.get('rank','')}
제목: {meta.get('title', '')}
장르: {meta.get('genre', '')}
긍정적평가: {meta.get('rating', '')}

{doc_text}
""".strip()
        )

    context_str = "\n\n---\n\n".join(context_texts)

    system_prompt = """당신은 영화 추천 어시스턴트입니다.
반드시 제공된 정보만 기반으로 답하세요."""

    user_prompt = f"""
다음 영화 정보들을 참고하세요:

{context_str}

질문: {query}
"""

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content


# In[ ]:


##rag는 검색만 잘함..? 논리 판단은 우리가 코드로 해야함 -> 프롬포팅으로 해결해도 될듯
def extract_year(genre_year):
    try:
        return int(genre_year.split("·")[-1].strip())
    except:
        return 0
    if "최근" in query or "요즘" in query:
        sorted_pairs = sorted(
            pairs,
            key=lambda x: extract_year(x[1].get("genre", "")),
            reverse=True  # 최신 먼저
        )
    if "최근" in query:
        sorted_pairs = sorted(
            pairs,
            key=lambda x: (
                extract_year(x[1].get("genre", "")),   # 1순위: 연도
                -int(x[1].get("rank", 999))            # 2순위: 순위
            ),
            reverse=True
        )


# In[44]:


if __name__ == "__main__":
    while True:
        user_query = input("질문을 입력하세요(종료: quit): ")
        if user_query.lower() == "quit":
            break 
        answer = generate_answer_with_context(user_query, top_k=20)
        # answer = generate_answer_without_context(user_query)
        print("===답변===")
        print(answer)
        print("==========\n")

