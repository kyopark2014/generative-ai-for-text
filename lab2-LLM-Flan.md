# FLAM-T5-xxl 모델 성능

[flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl)은 아래와 같습니다.

![image](https://github.com/kyopark2014/generative-ai-for-text/assets/52392004/71a4dec4-b511-42bc-88de-d8b10df804af)

## 수행 방법 

1) SageMaker Jumpstart의 "Flan-T5 XXL FP16"을 선택하고 아래와 같이 training inference를 "ml.g5.24xlarge"로 설정후에 [Train]을 수행합니다.

![noname](https://github.com/kyopark2014/generative-ai-for-text/assets/52392004/ab98860b-adfd-481a-b89a-60c3754b4125)


2) "04-flan-t5/00-prompt-enigeering-flan-t5-xxl.ipynb"을 열어서 모든 내용을 실행합니다. 실행한 결과는 [00-prompt-enigeering-flan-t5-xxl.ipynb](https://github.com/kyopark2014/generative-ai-for-text/blob/main/notebook/00-prompt-enigeering-flan-t5-xxl.ipynb)와 같습니다

이때의 주요한 Parameter들은 아래와 같습니다.
- max_length: Model generates text until the output length (which includes the input context length) reaches max_length. If specified, it must be a  positive integer.
- num_return_sequences: Number of output sequences returned. If specified, it must be a positive integer.
- num_beams: Number of beams used in the greedy search. If specified, it must be integer greater than or equal to num_return_sequences.
- no_repeat_ngram_size: Model ensures that a sequence of words of no_repeat_ngram_size is not repeated in the output sequence. If specified, it must be a positive integer greater than 1.
- temperature: Controls the randomness in the output. Higher temperature results in output sequence with low-probability words and lower temperature results in output sequence with high-probability words. If temperature -> 0, it results in greedy decoding. If specified, it must be a positive float.
- early_stopping: If True, text generation is finished when all beam hypotheses reach the end of sentence token. If specified, it must be boolean.
- do_sample: If True, sample the next word as per the likelihood. If specified, it must be boolean.
- top_k: In each step of text generation, sample from only the top_k most likely words. If specified, it must be a positive integer.
- top_p: In each step of text generation, sample from the smallest possible set of words with cumulative probability top_p. If specified, it must be a float between 0 and 1.
- seed: Fix the randomized state for reproducibility. If specified, it must be an integer.

## A Zero-shot Learning

### Extract Q&A

```java
context = """Once, a cunning fox saw a crow with a piece of cheese in its beak sitting on a branch. \
The fox devised a plan and flattered the crow, causing the crow to caw with delight, dropping the \
cheese which the fox quickly snatched up and ran away. The crow learned a valuable lesson and never trusted the fox again."""

question = 'who got cheated?'
answer = 'crow'
```

### Sentiment Analysis/Classification

```java
review = 'I hated the movie. Thoroughly disappointing for a sequel.'
sentiment = 'Sentiment (Good, Bad)'
Response: Bad
```

### Natural Language Inference (NLI)

```java
Prompt: The world cup has kicked off in Los Angeles, United States.
Based on the paragraph above can we conclude that: ”The world cup takes place in United States?"
[”yes”, ”no”]
Response: yes
```

### Text Generation

```java
Prompt: Title: "University has new facility coming up"
Given the above title of an imaginary article, imagine the article.

Response: The University of the West Indies (UWI) has announced that it will be building a new sports facility on its St Augustine's campus. The new facility will be located on the St Augustine's campus and will be a multi-purpose sports complex. The new sports complex will be built on the St Augustine's campus and will be a multi-purpose sports complex. The new sports complex will be built on the St Augustine's campus and will be a multi-purpose sports complex. The new sports complex will be built on the St Augustine's campus and will be a multi-purpose sports complex. The new sports complex will be built on the St Augustine's campus and will be a multi-purpose sports complex. The new sports complex will be built on the St Augustine's campus and will be a multi-purpose sports complex. The new sports complex will be built on the St Augustine
```

### Translation

```java
prompt = "Translate to German:  My name is Arthur"
Response: Ich bin Arthur.
```
### Q&A

```java
Customer: Hi there, I'm having a problem with my iPhone.
Agent: Hi! I'm sorry to hear that. What's happening?
Customer: The phone is not charging properly, and the battery seems to be draining very quickly. I've tried different charging cables and power adapters, but the issue persists.
Agent: Hmm, that's not good. Let's try some troubleshooting steps. Can you go to Settings, then Battery, and see if there are any apps that are using up a lot of battery life?
Customer: Yes, there are some apps that are using up a lot of battery.
Agent: Okay, try force quitting those apps by swiping up from the bottom of the screen and then swiping up on the app to close it.
Customer: I did that, but the issue is still there.
Agent: Alright, let's try resetting your iPhone's settings to their default values. This won't delete any of your data. Go to Settings, then General, then Reset, and then choose Reset All Settings.
Customer: Okay, I did that. What's next?
Agent: Now, let's try restarting your iPhone. Press and hold the power button until you see the "slide to power off" option. Slide to power off, wait a few seconds, and then turn your iPhone back on.
Customer: Alright, I restarted it, but it's still not charging properly.
Agent: I see. It looks like we need to run a diagnostic test on your iPhone. Please visit the nearest Apple Store or authorized service provider to get your iPhone checked out.
Customer: Do I need to make an appointment?
Agent: Yes, it's always best to make an appointment beforehand so you don't have to wait in line. You can make an appointment online or by calling the Apple Store or authorized service provider.
Customer: Okay, will I have to pay for the repairs?
Agent: That depends on whether your iPhone is covered under warranty or not. If it is, you won't have to pay anything. However, if it's not covered under warranty, you will have to pay for the repairs.
Customer: How long will it take to get my iPhone back?
Agent: It depends on the severity of the issue, but it usually takes 1-2 business days.
Customer: Can I track the repair status online?
Agent: Yes, you can track the repair status online or by calling the Apple Store or authorized service provider.
Customer: Alright, thanks for your help.
Agent: No problem, happy to help. Is there anything else I can assist you with?
Customer: No, that's all for now.
Agent: Alright, have a great day and good luck with your iPhone!

What are the customer and agent talking about?

Response: The customer's iPhone isn't charging properly and the battery drains very quickly, regardless of what cable or power adapter he uses. The Customer will need to go to the Apple Store or an authorized service provider to repair it. The repairs usually take between 6 and 10 business days and will be done free of charge while the issue is addressed, depending on the conditions of the warranty and the Customer's status.
```

### In Context Q&A

Prompt: 
Context: 
The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.

In what country is Normandy located?

Response: France
```

## B. One-shot Learning

### Text Summarisation

```text
Prompt: 
article: I love apples especially the large juicy ones. Apples are a great source of vitamins and fiber. An apple a day keeps the doctor away!
summary: I love apples.
--
article: I hate oranges especially the bitter ones. They are high in citric acid and they give me heart burns. Doctor suggests me to avoid them!
summary:

Response: I hate oranges. --
```

### Natural Language Generation (NLG)

```text
Prompt: name[The Punter], eat_type[Indian], price_range[cheap] ==> sentence describing the place: The Punter provides Indian food in the cheap price range. ; name[Blue Spice], eatType[coffee shop], price_range[expensive] ==> sentence describing the place:

Response: Blue Spice is a coffee shop that is expensive and serves sushi.
```

### Flip (Entity Extraction)

```text
Prompt: The Punter provides Indian food in the cheap price range. ==> name[The Punter], eat_type[Indian], price_range[cheap]
--
Blue Spice is a coffee shop that is a bit pricy. ==>
Response: name[Blue Spice], eat_type[coffee shop], price_range[high] --
```

## C. Few-shot Learning

### Translation

```text
prompt = f"""
Translate English to French:
sea otter => loutre de mer
--
peppermint => menthe poivrée
--
plush girafe => girafe en peluche
--
cheese =>
"""
Response: chèvre => chèvre --
```

### Entity Extraction

```text
prompt = f"""
Extract main person:

s: John is playing basketball
p: John
##
s: Jeff and Phil are chatting about GAI. Phil has to run. He is in a rush
p: Phil
##
s: Max is older than Emma
p: Max
##
s: Susan misses the bus this morning but still get in time for her meeting with Sara
p:
"""

Response: Susan
```

### Classification

```text
prompt = f"""
Classify the topic of the following paragraph: Carlyle Looks Toward Commercial Aerospace (Reuters) Reuters - Private investment firm Carlyle Group, which has a reputation for making well-timed and occasionally controversial plays in the defense industry, has quietly placed its bets on another part of the market.
Label: Business.

##

Classify the topic of the following paragraph: Some People Not Eligible to Get in on Google IPO Google has billed its IPO as a way for everyday people to get in on the process, denying Wall Street the usual stranglehold it's had on IPOs. Public bidding, a minimum of just five shares, an open process with 28 underwriters - all this pointed to a new level of public participation. But this isn't the case.
Label: Technology.

##

Classify the topic of the following paragraph: Indians Mount Charge The Cleveland Indians pulled within one game of the AL Central lead by beating the Minnesota Twins, 7-1, Saturday night with home runs by Travis Hafner and Victor Martinez.
Label: Sports.

##

Classify the topic of the following paragraph: Uptown girl, she's been living in her uptown world, I bet she never had a backstreet guy, I bet her mother never told her why, I'm gonna try.
Label:
"""

Response: Song
```

```text
prompt = f"""

Classify the below use-case description to one of the following NLP tasks: Short form generation, Long form generation, Summarization, Classification, Question answering, Paraphrasing, Conversational agent, Information extraction, Generate code

Use case: My native language is not English, I have blogs and I write my own articles. I also get articles from outsource writers, so I want to use it to re-write such articles, so i will create a tool for that
NLP Task: Paraphrasing
##

Classify the below use-case description to one of the following NLP tasks: Short form generation, Long form generation, Summarization, Classification, Question answering, Paraphrasing, Conversational agent, Information extraction, Generate code

Use case: Just experimenting with content generation
NLP task: Long form generation
##

Classify the below use-case description to one of the following NLP tasks: Short form generation, Long form generation, Summarization, Classification, Question answering, Paraphrasing, Conversational agent, Information extraction, Generate code

Use case: My company MetaDialog provides human in the loop automated support for costumers, we are currently using GPT3 to generate answers using our custom search engine to clients questions, and then provide the answers as suggestions to human agents to use or reject them as real answers to clients.
NLP task: Conversational agent
##

Classify the below use-case description to one of the following NLP tasks: Short form generation, Long form generation, Summarization, Classification, Question answering, Paraphrasing, Conversational agent, Information extraction, Generate code

Use case: Receipt extraction including line items from plain text (sources being OCR-ed images, PDFs and HTML Emails)
NLP task: Information extraction
##

Classify the below use-case description to one of the following NLP tasks: Short form generation, Long form generation, Summarization, Classification, Question answering, Paraphrasing, Conversational agent, Information extraction, Generate code

Use case: I have a lot of legacy documentation which needs to be cleaned, summarized, and queried. I think AI21 would help.
NLP task: Summarization

##

Classify the below use-case description to one of the following NLP tasks: Short form generation, Long form generation, Summarization, Classification, Question answering, Paraphrasing, Conversational agent, Information extraction, Generate code

Use case: Answer questions based on a given corpus of information
NLP task: Question answering
##

Classify the below use-case description to one of the following NLP tasks: Short form generation, Long form generation, Summarization, Classification, Question answering, Paraphrasing, Conversational agent, Information extraction, Generate code

Use case: creating useful content for companies websites articles
NLP task:
"""

Response: Long form generation
```
