# Text를 위한 Generative AI

## Text Labs

여기서는 [sagemaker-jumpstart-generative-ai-examples](https://github.com/sunbc0120/sagemaker-jumpstart-generative-ai-examples)의 Text를 이용한 생성 AI에 대해 정리합니다. 먼저 SageMaker Studio에서 Terminal을 열어서 아래와 같이 관련파일을 다운로드 합니다.

```java
git clone https://github.com/sunbc0120/sagemaker-jumpstart-generative-ai-examples
```

### Lab 2: LLM (Flan-t5-xl) on SageMaker JumpStart: Prompt Engineering/In-context Q&A

#### 수행 방법 

1) SageMaker Jumpstart의 "Flan-T5 XXL FP16"을 선택하고 아래와 같이 training inference를 "ml.g5.24xlarge"로 설정후에 [Train]을 수행합니다.

![noname](https://github.com/kyopark2014/generative-ai-for-text/assets/52392004/ab98860b-adfd-481a-b89a-60c3754b4125)


2) "04-flan-t5/00-prompt-enigeering-flan-t5-xxl.ipynb"을 열어서 모든 내용을 실행합니다. 실행한 결과는 [00-prompt-enigeering-flan-t5-xxl.ipynb](https://github.com/kyopark2014/generative-ai-for-text/blob/main/notebook/00-prompt-enigeering-flan-t5-xxl.ipynb)와 같습니다

#### 수행 결과

- Parameters
 - max_length: Model generates text until the output length (which includes the input context length) reaches max_length. If specified, it must be a  positive integer.
 - num_return_sequences: Number of output sequences returned. If specified, it must be a positive integer.
 - num_beams: Number of beams used in the greedy search. If specified, it must be integer greater than or equal to num_return_sequences.
 - no_repeat_ngram_size: Model ensures that a sequence of words of no_repeat_ngram_size is not repeated in the output sequence. If specified, it must be a positive integer greater than 1.
 - temperature: Controls the randomness in the output. Higher temperature results in output sequence with low-probability words and lower temperature results in output sequence with high-probability words. If temperature -> 0, it results in greedy decoding. If specified, it must be a positive float.
 - early_stopping: If True, text generation is finished when all beam hypotheses reach the end of sentence token. If specified, it must be boolean.
 do_sample: If True, sample the next word as per the likelihood. If specified, it must be boolean.
 - top_k: In each step of text generation, sample from only the top_k most likely words. If specified, it must be a positive integer.
 - top_p: In each step of text generation, sample from the smallest possible set of words with cumulative probability top_p. If specified, it must be a float between 0 and 1.
 - seed: Fix the randomized state for reproducibility. If specified, it must be an integer.







 

### Lab 3: RAG on SageMaker: Embedding/In-context Q&A/Knowledge Augmentation
