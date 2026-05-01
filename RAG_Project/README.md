# RAG System Project – Eagle Document

## Description of Selected Document
For the project I selected a Wikipidea article on eagles (https://en.wikipedia.org/wiki/Eagle), This article gives the basics of what an eagle is, where they live, their habitat, and types of eagles that exist.

---

## Project Overview 
**This project implements a RAG system that:**
* Extracts text from a hard-coded document
* Splits the text into readable/processable chuncks that the system can take out the semantics and understand better
* Stores embeddings in a FAISS vector database
* Retrieves the relevant chunks based on the user input question
* Generates answers using the OpenAI language model

---

## Q&A of RAG Model

``` text
1. Your question: How fast can eagles go?
   Answer: I don’t know from the provided context. It mentions that eagles have more direct, faster flight than some other raptors, but gives no specific speeds.

2. Your question: What is an eagle?
   Answer: An eagle is a large, powerfully built bird of prey in the family Accipitridae. The term is a common name without strict taxonomic weight: “true eagles” are in the subfamily Aquilinae, but many large raptors (e.g., the bald eagle) are also called eagles. They have heavy heads and beaks and hunt sizeable vertebrates.

3. Your question: What are the eagles habitat?
   Answer: Eagles inhabit nearly all parts of the world and almost every habitat—ranging from northern tundra to deserts and tropical rainforests. Depending on species, they use open country (Aquila), woodlands and forests (Spizaetus), and dense tropical forests (harpy eagles); fish eagles live near water on every continent except South America.
```


### **Analysis of how retrieval quality varied with different chunk_size and chunk_overlap**

When changing the **Chunk_size** the model provided longer or shorter answers in correspondance with the value it was set.

The **Chuck_ovelap** changed what words and embeddings in the question could overlap leading to more accurate and well put together responses the higher this value was and less quality answers the lower this value was.

---

## Deep-Dive Questions & Answers 

### 1. What is the purpose of chunking in a RAG system?
**Answer:**  
Chunking breaks large documents into smaller pieces so they can be processed within model limits and retrieved more accurately.

---

### 2. How do embeddings help retrieve relevant information?
**Answer:**  
Embeddings convert text into numerical vectors that capture semantic meaning, allowing similarity-based retrieval instead of keyword matching.

---

### 3. What role does FAISS play in the system?
**Answer:**  
FAISS enables efficient similarity search over embeddings, allowing the system to quickly find the most relevant chunks for a query.

---

### 4. Why is a cross-encoder used for re-ranking?
**Answer:**  
A cross-encoder evaluates the relationship between the question and each chunk more precisely, improving ranking accuracy before generating the final answer.

---

### 5. How does chunk overlap affect answer quality?
**Answer:**  
Chunk overlap preserves context between chunks, improving continuity and retrieval accuracy, but too much overlap can introduce redundancy.

---

## Reflection Report
The project helped me understand the inner workings of AI and how they it was able to come up with answers based on questions. What essentially happened under the hood, was the system seperated text or inputs from a user into embeddings and chunks of readable text that can be used to figure out the context of other words in the sentence. After it figures out the meaning or intention of the user input it searches a database or its "knowledge" to figure out the best possible answer and provides it to the user. The project was extremely useful as I was able to delve deeper into the inner components of AI generated responses.
