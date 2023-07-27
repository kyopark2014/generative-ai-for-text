# RAG (Retrieval-Augmented Generation)

사전 학습(Pretrained)된 LLM과 정보 검색을 결합해 더 정확하고 맥락에 맞는 답변을 도출합니다. 외부 지식에 직접 액세스하거나 외부 지식을 통합하여 좀 더 정확한 답변을 생성할 수 있습니다.

- In-context learning처럼 모델의 weight를 변경하지 않습니다.
- Knowledge DB에서 Prompt에 맞는 검색 결과에 대한 임베딩을 뽑아서 Prompt에 추가하여 넣어주는 방식입니다. 따라서 양질의 DB 구축이 중요합니다. (dk님)
- RAG는 모델을 개선하는 것은 아니고, Prior를 더욱 좋게 만들어주는 것이라 보면 될 것 같습니다. 그렇기 때문에 좋은 맥락을 만들어주는 Retriever의 역할이 중요합니다. 더 좋은 결과를 내기 위해서는 후보맥락을 뽑아주는 Retriever를 잘 만드는 것이 중요하고 많은 연구가 이뤄지고 있습니다. 예를들어 LLM의 오류를 Retriever에 Loss값으로 전파해서 Retriever 네트워크만 추가학습하는 기법도 있습니다. (dy님)

[Build a powerful question answering bot with Amazon SageMaker, Amazon OpenSearch Service, Streamlit, and LangChai](https://aws.amazon.com/ko/blogs/machine-learning/build-a-powerful-question-answering-bot-with-amazon-sagemaker-amazon-opensearch-service-streamlit-and-langchain/?sc_channel=sm&sc_campaign=Machine_Learning&sc_publisher=LINKEDIN&sc_geo=GLOBAL&sc_outcome=awareness&trk=machine_learning&linkId=219734484)의 아래 그림을 참조합니다.


![image](https://github.com/kyopark2014/generative-ai-for-text/assets/52392004/ca6ea655-af88-4de5-9c37-b807db6c12da)


#### 프로세스 단계

1) retriever를 사용하여 지식 베이스에서 유관 정보를 검색합니다.
2) generator를 사용해 검색한 정보와 쿼리값을 기반으로 답변을 생성합니다. 


hallucination을 막을 수 있음

#### Lab 3: RAG on SageMaker: Embedding/In-context Q&A/Knowledge Augmentation

여기서는 Flan T5 모델과 AI21의 모델을 비교합니다. Hallucination 문제에 대해 AI21은 "Answer not in document"와 같은 응답이 가능합니다.

"08-RAG-based-question-answering/01-question_answerIng_jumpstart_knn.ipynb"을 수행하여 [01-question_answering_jumpstart_ai21-apigateway.ipynb](https://github.com/kyopark2014/generative-ai-for-text/blob/main/notebook/01-question_answering_jumpstart_ai21-apigateway.ipynb)과 같은 결과를 얻었습니다.
