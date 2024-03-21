from flask import Flask, render_template, request, jsonify
from langchain import HuggingFacePipeline
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain.embeddings import HuggingFaceInstructEmbeddings
import os
import torch
from langchain import PromptTemplate

app = Flask(__name__)

prompt_template = """
    I'm your friendly NLP chatbot named ChakyBot, here to assist Chaky and Gun with any questions they have about Natural Language Processing (NLP). 
    If you're curious about how probability works in the context of NLP, feel free to ask any questions you may have. 
    Whether it's about probabilistic models, language models, or any other related topic, 
    I'm here to help break down complex concepts into easy-to-understand explanations.
    Just let me know what you're wondering about, and I'll do my best to guide you through it!
    {context}
    Question: {question}
    Answer:
    """.strip()

PROMPT = PromptTemplate.from_template(
    template = prompt_template
)

embed_model = 'hkunlp/instructor-base'

# Load Hugging Face model for the chatbot
# model_id = '../A7/model_gpt2/gpt2-span-head-few-shot-k-16-finetuned-squad-seed-0'
model_id = 'D:/AIT-2023/NLP/A7/model_gpt2/gpt2-span-head-few-shot-k-16-finetuned-squad-seed-0'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    model_kwargs={"temperature": 0, "repetition_penalty": 1.5}
)
llm = HuggingFacePipeline(pipeline=pipe)

# Define question generator and document chain
question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)
doc_chain = load_qa_chain(llm=llm, chain_type='stuff', prompt=PROMPT, verbose=True)

# Load vector store for retrieval
# vector_path = '../A7/vector-store'
vector_path = 'D:/AIT-2023/NLP/A7/vector-store'
db_file_name = 'nlp_stanford'
embedding_model = HuggingFaceInstructEmbeddings(model_name=embed_model, model_kwargs={"device": torch.device('cpu')})
vectordb = FAISS.load_local(folder_path=os.path.join(vector_path, db_file_name),embeddings=embedding_model, index_name='nlp')
retriever = vectordb.as_retriever()

# Initialize memory for conversation
memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True, output_key='answer')

# Define Conversational Retrieval Chain
chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    return_source_documents=True,
    memory=memory,
    verbose=True,
    get_chat_history=lambda h: h
)

# Store chat history
chat_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json['message']
    # Get bot response using the chatbot model
    bot_response = chain({"question": user_message})['answer']
    # Store user message and bot response in chat history
    chat_history.append({'user': user_message, 'bot': bot_response})
    return jsonify({'bot_response': bot_response})

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    return jsonify(chat_history)

if __name__ == '__main__':
    app.run(debug=True)
