1. Generate test cases using LLM for a website.
      1.1 Consider https://opensource-demo.orangehrmlive.com/web/index.php/auth/login  and ask LLM to generate test cases for home page.
2. Ask LLM to write selenium python code to test home page of above mentioned website.
3. Identify  how can we ask LLM to run specific code mentioned in step 2 for a specific test case mentioned in Step 1.
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
from os import environ

from rouge import Rouge

from milvus import default_server

from pymilvus import connections, utility

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Milvus

from langchain.document_loaders import TextLoader

from langchain.text_splitter import CharacterTextSplitter

from langchain.document_loaders import PyPDFLoader

from langchain.chains.question_answering import load_qa_chain

from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI

default_server.start()

MILVUS_HOST = '127.0.0.1'

MILVUS_PORT = default_server.listen_port

OPENAI_API_KEY = "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"

environ["OPENAI_API_KEY"] = OPENAI_API_KEY

 

loader = PyPDFLoader("/home/shivansu_soni/Validation_using_ReLM.pdf")

docs = loader.load()

doc_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)

docs = doc_splitter.split_documents(docs)

vectorstores = Milvus._get_index

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vector_store = Milvus.from_documents(

 

    docs,

 

    embedding=embeddings,

 

    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}

 

)

 

print("connected")

query ="How ReLM validates LLM?"

 

docs = vector_store.similarity_search(query)

 

#llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

 

llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.50, openai_api_key=OPENAI_API_KEY )

chain = load_qa_chain(llm, chain_type="stuff")

answer = chain.run(input_documents=docs, question=query)

print(answer)

====================

asgiref==3.5.2

certifi==2022.9.24

cffi==1.15.1

charset-normalizer==2.1.1

cryptography==38.0.1

defusedxml==0.7.1

Django==4.1.2

idna==3.4

oauthlib==3.2.2

Pillow==9.2.0

pycparser==2.21

PyJWT==2.6.0

python-dotenv==0.19.0

python3-openid==3.2.0

pytz==2022.5

requests==2.28.1

requests-oauthlib==1.3.1

social-auth-app-django==5.0.0

social-auth-core==4.3.0

sqlparse==0.4.3

urllib3==1.26.12

 
