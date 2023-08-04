# Text를 위한 Generative AI

Generative AI에서는 레이블(Label)이 안된 다양한 데이터로 Pretrain을 수행한 Foundation Model을 이용하여, Text generation, Summarization, Inofomation Extraction, Q&A, Chatbot에 활용할 수 있습니다. 이를 통해 서비스와 인프라를 쉽게 이용하고, 효율적으로 비용을 관리하고, 공통적인 비지니스 작업(Task)에 빠르게 적용할 수 있습니다. 

## Projects

### Bedrock과 Vector Store를 이용한 Question/Answering Chatbot 

[question-answering-chatbot-with-vector-store](https://github.com/kyopark2014/question-answering-chatbot-with-vector-store)은 질문/답변(Question/Answering)을 수행하는 Chatbot을 vector store를 이용하여 RAG를 통해 구현합니다. Vector store로는 In-memory 방식의 Faiss와 persistent store인 OpenSearch를 이용합니다. 

접속주소는 https://d1jlvc7achaanj.cloudfront.net/chat.html 입니다.

### Bedrock과 Kendra를 이용한 Question/Answering Chatbot

[question-answering-chatbot-with-kendra](https://github.com/kyopark2014/question-answering-chatbot-with-kendra)에서는 질문/답변(Question/Answering)을 수행하는 Chatbot을 Kendra를 이용한 RAG로 구현합니다. Seoul Region은 Kendra가 아직 GA되지 않았으므로, Tokyo Region을 이용합니다. 

접속주소는 https://dgwydtu6r7hxf.cloudfront.net/chat.html 입니다.

### Llama 2와 Vector Store를 이용한 Question/Answering Chatbot

[Llama2-chatbot-with-vector-store](https://github.com/kyopark2014/Llama2-chatbot-with-vector-store)에서는 질문/답변(Question/Asnwering)을 수행하는 Chatbot을 Llama 2와 vector store를 이용하여 구현합니다. Llama 2는 N.Virginia (us-east-1)에 SageMaker JumpStart를 이용해 설치가 되고, SageMaker Endpoint를 LangChain으로 연결하여 Chatbot을 구현합니다. Vector store로 Faiss와 OpenSearch를 선택적으로 사용할 수 있습니다.

접속주소는 https://d36e8hz1qn2mjd.cloudfront.net/chat.html 입니다.

### Llama 2와 Kendra를 이용한 Question/Answering Chatbot

[Llama2-chatbot-with-kendra](https://github.com/kyopark2014/Llama2-chatbot-with-kendra)에서는 질문/답변(Question/Asnwering)을 수행하는 Chatbot을 Llama 2와 Kendra를 이용하여 구현합니다. 현재 Llama 2를 사용할때 Kendra를 LangChain을 연결하는 부분에서 문제가 있어서 기능 테스트중입니다.

### Bedrock을 위한 Basic Question/Answering Chatbot

[simple-chatbot-using-LLM-based-on-amazon-bedrock](https://github.com/kyopark2014/simple-chatbot-using-LLM-based-on-amazon-bedrock)에서는 Bedrock을 이용해 간단한 chatbot과 파일에 대한 Summarization을 구현합니다. Bedrock 인터페이스 및 기본 동작을 설명하기 위한 용도로 RAG는 포함되어 있지 않습니다.



## Tasks

Prompt를 이용한 Task에는 아래와 같은것들이 있습니다. 

* Task based prompting
NLP encompasses a diverse range of tasks like summarization, rewriting, information extraction, question answering, classification, conversation, translation, reasoning and code generation. These tasks power applications such as virtual assistants, chatbots, information retrieval systems, and more. NLP allows software systems to analyze, interpret and respond to human language data at scale.
1) Text Summarization: Producing a shorter version of a piece of text while retaining the main ideas. Examples are summarizing news articles, text documents or reports into single paragraphs.
2) Rewriting: Convert the input text into a different wording. For example: convert a document with legal jargon to a plain english document, rewrite an email in different tones (formal, informal), structured table to paragraph. Used by departments such as marketing, sales or legal.
3) Information Extraction: Identifying and extracting specific pieces of information from text. For example, extracting names of people, locations, events or numbers from documents. Used for tasks like sales lead generation or resume parsing.
4) Question Answering: Using text or structured data to determine the answer to a question. For example, answering queries about movies, books or general knowledge. Useful for virtual assistants and help desk automation.
5) Text Classification: Assigning a label or category to a piece of text based on its contents. For example, detecting language, spam detection, sentiment analysis or tagging support tickets with relevant categories.
6) Conversation: Ability to have coherent multi-turn dialogue with humans to accomplish tasks like customer service, tech support or general conversations. Examples are chatbots, dialogue agents and conversational interfaces.
7) Translation: Changing input in source language to a target language. For example, English to German, slang to literary.
8) Reasoning: Using language to logically draw new conclusions, inferences or predictions that are not explicitly stated in the original text or data. For example, analyzing statements about objects, people and events to anticipate what otherwise unstated inferences can be drawn by filling in missing details.
9) Code Generation: Producing software code based on a natural language description or in response to changes in a end user interface. For example, generating HTML markup or syntax in a specific programming language.








## LLM

### Lab 2: LLM (Flan-t5-xl) on SageMaker JumpStart: Prompt Engineering/In-context Q&A

여기서는 [sagemaker-jumpstart-generative-ai-examples](https://github.com/sunbc0120/sagemaker-jumpstart-generative-ai-examples)의 Text를 이용한 생성 AI에 대해 정리합니다. 먼저 SageMaker Studio에서 Terminal을 열어서 아래와 같이 관련파일을 다운로드 합니다.

```java
git clone https://github.com/sunbc0120/sagemaker-jumpstart-generative-ai-examples
```

[flan-t5-xxl](https://github.com/kyopark2014/generative-ai-for-text/blob/main/lab2-LLM-Flan.md)은 [00-prompt-enigeering-flan-t5-xxl.ipynb](https://github.com/kyopark2014/generative-ai-for-text/blob/main/notebook/00-prompt-enigeering-flan-t5-xxl.ipynb)와 같이 Zero-shot과 Few-shot learning 결과를 확인합니다. 여기서, Extract Q&A, Sentiment Analysis/Classification, Natural Language Inference (NLI), Text Generation, Translation, Q&A, In Context Q&A, Natural Language Generation (NLG), Flip (Entity Extraction)을 수행합니다.

#### Chaine-of-thought (CoT) prompting

다단계 문제의 해결을 위해 사람의 연쇄적 사고방식을 모방하여 중간 추론 단계를 생성하는 방식으로 기초 모델의 추론능력을 향상시킬수 있습니다. 대형 모델(>100B)이상에서 잘 작동하며 해석능력을 높이기 위해 CoT 추론 데이터셋을 파인튜닝 할 수 있습니다. SageMaker JumptStart의 FLAN-T5 모델은 CoT 추론 훈련을 받았습니다.



### In-context learning

LLM의 파라메터 숫자를 무작정 늘리는것은 메모리와 훈련시간의 한계가 있으므로, 적당한 크기의 LLM 모델에 Fine tuning을 하는것이 좋은 선택일 수 있습니다. Fine tuning은 모델의 Weight를 변경하므로써 좀 더 바람직한 결과를 얻을 수 있지만, 이것 또한 많은 비용이 필요합니다. In-context learning은 모델의 Weight를 변경하지 않으면서, 모델의 성능을 향상 시킬수 있는 방법입니다. 이것은 다음 단어를 예측할때 도움이 되는 Context를 제공하는 방법입니다.

- Prior에 더 많은 정보가 추가되기 때문에 목적에 맞는 Posterior를 만들어주는것 (대열님)

#### Zero-shot

모델이 주어진 자연어 설명(natual language description)만을 이용하여 예측을 수행합니다. 즉, 예시없이 질문만하는 경우입니다.

```text
Task description: Translate English to French:
Prompt: cheese =>
```

[00-prompt-enigeering-flan-t5-xxl.ipynb](https://github.com/kyopark2014/generative-ai-for-text/blob/main/notebook/00-prompt-enigeering-flan-t5-xxl.ipynb)에 아래와 같은 예제가 있습니다.

```text
prompt: "Translate to German:  My name is Arthur"
Response: Ich bin Arthur.
```

#### One-shot

모델에 하나의 예시를 주고 예측을 수행합니다. 

```text
Test description: Translate English to French:
Example: sea otter => loutre de mer
Prompt: cheese =>
```

#### Few-shot

여러개의 예시를 주고 예측을 수행합니다. Prior에 더 많은 정보가 추가되기 때문에 목적에 맞는 Posterior를 만들어줄 수 있습니다. (dy님)

```text
Test description: Translate English to French:
Example: sea otter => loutre de mer
Example: peppermint => menthe poivree
Example: plus girafe => girafe peluche
Prompt: cheese =>
```


## LLM의 성능을 향상시키는 방법


## RAG (Retrieval-Augmented Generation)

[RAG](https://github.com/kyopark2014/generative-ai-for-text/blob/main/rag.md)에서 Hallucination을 방지하고 LLM의 성능을 향상시킬수 있는 RAG에 대해 설명합니다.

vector store를 사용하면 대규모 언어 모델의 token 사이즈를 넘어서는 긴 문장을 활용하여 질문/답변(Question/Answering)과 같은 Task를 수행할 수 있으며 환각(hallucination) 영향을 줄일 수 있습니다.

### Faiss

[Faiss](https://github.com/facebookresearch/faiss)는 Facebook에서 오픈소스로 제공하는 In-memory vector store로서 embedding과 document들을 저장할 수 있으며, LangChain을 지원합니다. 비슷한 역할을 하는 persistent store로는 Amazon OpenSearch, RDS Postgres with pgVector, ChromaDB, Pinecone과 Weaviate가 있습니다. 

faiss.write_index(), faiss.read_index()을 이용해서 local에서 index를 저장하고 읽어올수 있습니다. 그러나 S3에서 직접 로드는 현재 제공하고 있지 않습니다. EFS에서 저장후 S3에 업로드 하는 방식은 레퍼런스가 있습니다.

[Faiss-LangChain](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/faiss)와 같이 save_local(), load_local()을 사용할 수 있고, merge_from()으로 2개의 vector store를 저장할 수 있습니다.

#### 문서 등록

문서를 업로드하면 FAISS를 이용하여 vector store에 저장합니다. 파일을 여러번 업로드할 경우에는 기존 vector store에 추가합니다. 

```python
docs = load_document(file_type, object)

vectorstore_new = FAISS.from_documents(
    docs,
    bedrock_embeddings,
)

vectorstore.merge_from(vectorstore_new)
```

업로드한 문서 파일에 대한 요약(Summerization)을 제공하여 사용자의 파일에 대한 이해를 돕습니다.

```python
query = "summerize the documents"

msg = get_answer(query, vectorstore_new)
print('msg2: ', msg)
```



## LangChain 

[LangChain](https://python.langchain.com/en/latest/index.html)은 언어 모델기반의 어플리케이션을 개발하기 위한 Framework입니다.

LLM의 문장이 긴 경우에 Chunk단위로 요약해서 사용할때 편리하고(추가 확인), Bedrock의 모델 API들이 LangChain과 연동되므로 편리하다고 합니다. (gs님)

"08-RAG-based-question-answering/02-question_answering_langchain_jumpstart.ipynb"을 참조하여 [02-question_answering_langchain_jumpstart.ipynb](https://github.com/sunbc0120/sagemaker-jumpstart-generative-ai-examples/blob/main/08-RAG-based-question-answering/02-question_answering_langchain_jumpstart.ipynb)와 같이 실행하였습니다.

## Hallucination

Hallucination은 환각 또는 거짓정보로 번역되는데, LLM 모델에 학습되지 않은 내용을 물어보거나, 잘못된 context와 함께 질문하는 방식으로 믿을 수 없거나(unfaithful) 터무니 없는(non-sensical) 텍스트를 생성하는 현상을 말합니다. 이것은 유창하고 자연스러워 보여서 그렇듯하게 보여지므로, 사용자에게 잘못된 정보를 전달할 수 있습니다.

### FLAN-T와 AI21의 비교

[Flan-t5-xxl](https://github.com/kyopark2014/generative-ai-for-text/blob/main/notebook/01-question_answering_jumpstart_ai21-apigateway.ipynb)의 경우에 Hallucination 질문에 대하여 아래와 같이 

아래와 같은 답이 없는 context로 제공하고 에펠탑의 높이를 물어보는 질문(What is the height of the Eiffel tower?)을 합니다.

```text
partial_context = "Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest human-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure in the world to surpass both the 200-metre and 300-metre mark in height. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
```

이때의 결과로, Flan-t5-xxl은 "202.5 metres"로 잘못된 대답을 하지만 AI21 모델의 경우에 "Answer not in document"로 답변을 하고 있습니다. 



## Reference 

[sagemaker-jumpstart-generative-ai-examples](https://github.com/sunbc0120/sagemaker-jumpstart-generative-ai-examples)

[Build a powerful question answering bot with Amazon SageMaker, Amazon OpenSearch Service, Streamlit, and LangChai](https://aws.amazon.com/ko/blogs/machine-learning/build-a-powerful-question-answering-bot-with-amazon-sagemaker-amazon-opensearch-service-streamlit-and-langchain/?sc_channel=sm&sc_campaign=Machine_Learning&sc_publisher=LINKEDIN&sc_geo=GLOBAL&sc_outcome=awareness&trk=machine_learning&linkId=219734484)

[Generative AI on Amazon SageMaker - Workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/972fd252-36e5-4eed-8608-743e84957f8e/en-US)

[Large Generative AI model hosting workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/bb62b5d7-313f-4733-88cd-9c1aa41c724d/en-US)

[LLM Instruction Tuning on SageMaker](https://github.com/aws-samples/aws-ml-jp/blob/main/tasks/generative-ai/text-to-text/fine-tuning/instruction-tuning/README_en.md)

[Introducing the Hugging Face LLM Inference Container for Amazon SageMaker](https://huggingface.co/blog/sagemaker-huggingface-llm)

[RAG-ing Success: Guide to choose the right components for your RAG solution on AWS](https://medium.com/@pandey.vikesh/rag-ing-success-guide-to-choose-the-right-components-for-your-rag-solution-on-aws-223b9d4c7280)

[Amazon OpenSearch Service’s vector database capabilities explained](https://aws.amazon.com/ko/blogs/big-data/amazon-opensearch-services-vector-database-capabilities-explained/)

