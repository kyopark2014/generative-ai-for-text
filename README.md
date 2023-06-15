# Text를 위한 Generative AI

Generative AI에서는 레이브리 안된 다양한 데이터로 Pretrain을 수행한 Foundation Model을 이용하여 Text generation, Summarization, Inofomation Extraction, Q&A, Chatbot에 활용할 수 있습니다. 이를 통해 서비스와 인프라를 쉽게 이용하고, 효율적으로 비용을 관리하고, 공통적인 비지니스 작업(Task)에 빠르게 적용할 수 있습니다. 





여기서는 [sagemaker-jumpstart-generative-ai-examples](https://github.com/sunbc0120/sagemaker-jumpstart-generative-ai-examples)의 Text를 이용한 생성 AI에 대해 정리합니다. 먼저 SageMaker Studio에서 Terminal을 열어서 아래와 같이 관련파일을 다운로드 합니다.

```java
git clone https://github.com/sunbc0120/sagemaker-jumpstart-generative-ai-examples
```

## LLM

### Lab 2: LLM (Flan-t5-xl) on SageMaker JumpStart: Prompt Engineering/In-context Q&A

[flan-t5-xxl](https://github.com/kyopark2014/generative-ai-for-text/blob/main/lab2-LLM-Flan.md)은 [00-prompt-enigeering-flan-t5-xxl.ipynb](https://github.com/kyopark2014/generative-ai-for-text/blob/main/notebook/00-prompt-enigeering-flan-t5-xxl.ipynb)와 같이 Zero-shot과 Few-shot learning 결과를 확인합니다. 여기서, Extract Q&A, Sentiment Analysis/Classification, Natural Language Inference (NLI), Text Generation, Translation, Q&A, In Context Q&A, Natural Language Generation (NLG), Flip (Entity Extraction)을 수행합니다.

## RAG

### Lab 3: RAG on SageMaker: Embedding/In-context Q&A/Knowledge Augmentation

여기서는 Flan T5 모델과 AI21의 모델을 비교합니다. Hallucination 문제에 대해 AI21은 "Answer not in document"와 같은 응답이 가능합니다.

"08-RAG-based-question-answering/01-question_answerIng_jumpstart_knn.ipynb"을 수행하여 [01-question_answering_jumpstart_ai21-apigateway.ipynb](https://github.com/kyopark2014/generative-ai-for-text/blob/main/notebook/01-question_answering_jumpstart_ai21-apigateway.ipynb)과 같은 결과를 얻었습니다.

## LangChain 

[LangChain](https://python.langchain.com/en/latest/index.html)은 언어 모델기반의 어플리케이션을 개발하기 위한 Framework입니다.

"08-RAG-based-question-answering/02-question_answering_langchain_jumpstart.ipynb"을 참조하여 [02-question_answering_langchain_jumpstart.ipynb](https://github.com/sunbc0120/sagemaker-jumpstart-generative-ai-examples/blob/main/08-RAG-based-question-answering/02-question_answering_langchain_jumpstart.ipynb)와 같이 실행하였습니다.


## Reference 

[sagemaker-jumpstart-generative-ai-examples](https://github.com/sunbc0120/sagemaker-jumpstart-generative-ai-examples)

[Build a powerful question answering bot with Amazon SageMaker, Amazon OpenSearch Service, Streamlit, and LangChai](https://aws.amazon.com/ko/blogs/machine-learning/build-a-powerful-question-answering-bot-with-amazon-sagemaker-amazon-opensearch-service-streamlit-and-langchain/?sc_channel=sm&sc_campaign=Machine_Learning&sc_publisher=LINKEDIN&sc_geo=GLOBAL&sc_outcome=awareness&trk=machine_learning&linkId=219734484)

![image](https://github.com/kyopark2014/generative-ai-for-text/assets/52392004/72097003-364d-4e4e-9b27-4eafbab4e1ad)

[Generative AI on Amazon SageMaker - Workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/972fd252-36e5-4eed-8608-743e84957f8e/en-US)

[Large Generative AI model hosting workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/bb62b5d7-313f-4733-88cd-9c1aa41c724d/en-US)

[LLM Instruction Tuning on SageMaker](https://github.com/aws-samples/aws-ml-jp/blob/main/tasks/generative-ai/text-to-text/fine-tuning/instruction-tuning/README_en.md)

