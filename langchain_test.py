# from dotenv import load_dotenv, find_dotenv

import os
import pinecone
from langchain.vectorstores import Pinecone

from langchain import PromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import SimpleSequentialChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# 
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
llm = OpenAI(model_name="text-davinci-003", temperature=0.3)
# load_dotenv(find_dotenv())
# PromptTemplate.input_variables = ["concept"]    

# print(llm("explain large language models in one sentence"))

# messages = [
#     SystemMessage(content="You are an expert data scientist"),
#     HumanMessage(content="Write a Python script that trains a neural network on simulaled data")
# ]
# response = chat(messages)
# print(response.content, end="\n")

template = """
You are an expert data scientist with expertise in building deep learning models. 
Explain the concept of {concept} in a couple of lines.
"""
PromptTemplate.template = ["concept"]

prompt = PromptTemplate(
    input_variables = ["concept"],
    template=template
)

# print(llm(prompt.format(concept="autoencoder")))

chain = LLMChain(llm=llm, prompt=prompt)

# run the chain only specifying the input variables
# print(chain.run("autoencoder"))

second_prompt = PromptTemplate(
    input_variables = ["ml_concept"],
    template="Turn the concept description of {ml_concept} and explain it to me like I'm five in 500 words",
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True) 

# Run the chain specifying only the input variable for the first chain
explanation = overall_chain.run("autoencoder")
# print(explanation)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

texts = text_splitter.create_documents([explanation])
# print(texts)

embeddings = OpenAIEmbeddings(chunk_size=1)

query_result = embeddings.embed_query(texts[0].page_content)
# print(query_result)
                              
# Pinecone

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), 
    environment=os.getenv("PINECONE_ENV"))

index_name = "langchain-quickstart"

search = Pinecone.from_documents(texts, embeddings, index_name=index_name)


query = "What is magical about an autoencoder?"
result = search.similarity_search(query)    
print(result)
