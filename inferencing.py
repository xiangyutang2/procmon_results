from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.retrievers import MultiVectorRetriever
#%%
embedding_model='sentence-transformers/all-MiniLM-L6-v2'
llm_model ="llama3.2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

#%%
def get_conversation_chain(retriever):
    llm = Ollama(model=llm_model)
    contextualize_q_system_prompt = (
        "Given the chat history and the latest user question, "
        "provide a response that directly addresses the user's query based on the provided  documents. "
        "Do not rephrase the question or ask follow-up questions."
    )
 

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    ### Answer question ###
    system_prompt = (
        "As a personal chat assistant, provide accurate and relevant information based on the provided document in 2-3 sentences. "
        "Answe should be limited to 50 words and 2-3 sentences.  do not prompt to select answers or do not formualate a stand alone question. do not ask questions in the response. "
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    ### Statefully manage chat history ###
    store = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    print("Conversational chain created")
    return conversational_rag_chain


#%%
persist_directory = "C:/research/RAG/vectorstore/chroma"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
local_vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
doc_store_dir = 'C:/research/RAG/store_location'
fs = LocalFileStore(doc_store_dir)
local_doc_store = create_kv_docstore(fs)

big_chunks_retriever = MultiVectorRetriever(
    vectorstore=local_vectordb,
    docstore=local_doc_store,
    )

#%%
conversational_rag_chain=get_conversation_chain(big_chunks_retriever)
