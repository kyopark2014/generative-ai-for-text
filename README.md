# Text를 위한 Generative AI

## Text Labs

여기서는 [sagemaker-jumpstart-generative-ai-examples](https://github.com/sunbc0120/sagemaker-jumpstart-generative-ai-examples)의 Text를 이용한 생성 AI에 대해 정리합니다. 먼저 SageMaker Studio에서 Terminal을 열어서 아래와 같이 관련파일을 다운로드 합니다.

```java
git clone https://github.com/sunbc0120/sagemaker-jumpstart-generative-ai-examples
```

## Lab 2: LLM (Flan-t5-xl) on SageMaker JumpStart: Prompt Engineering/In-context Q&A

[flan-t5-xxl](https://github.com/kyopark2014/generative-ai-for-text/blob/main/lab2-LLM-Flan.md)은 [00-prompt-enigeering-flan-t5-xxl.ipynb](https://github.com/kyopark2014/generative-ai-for-text/blob/main/notebook/00-prompt-enigeering-flan-t5-xxl.ipynb)와 같이 Zero-shot과 Few-shot learning 결과를 확인합니다. 여기서, Extract Q&A, Sentiment Analysis/Classification, Natural Language Inference (NLI), Text Generation, Translation, Q&A, In Context Q&A, Natural Language Generation (NLG), Flip (Entity Extraction)을 수행합니다.

## Lab 3: RAG on SageMaker: Embedding/In-context Q&A/Knowledge Augmentation

여기서는 Flan T5 모델과 AI21의 모델을 비교합니다. Hallucination 문제에 대해 AI21은 "Answer not in document"와 같은 응답이 가능합니다.

"08-RAG-based-question-answering/01-question_answerIng_jumpstart_knn.ipynb"을 수행하여 [01-question_answering_jumpstart_ai21-apigateway.ipynb](https://github.com/kyopark2014/generative-ai-for-text/blob/main/notebook/01-question_answering_jumpstart_ai21-apigateway.ipynb)과 같은 결과를 얻었습니다.

