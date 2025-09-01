from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def create_chain(store: FAISS):
    model = ChatOllama(model="deepseek-r1:8b")

    prompt = ChatPromptTemplate.from_template("""
        Answer the question thruthfully as possible using the provided context, and if the answer is not container within the text below, say "I don't know".
        Context: {context}
        Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)

    retriever = store.as_retriever(search_kwargs={"k": 3})
    retriever_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain,
    )

    return retriever_chain


def query(store):
    chain = create_chain(store)

    question = input("Ask something about offerapi code: ")
    response = chain.stream({"input": question})

    for part in response:
        if "answer" in part:
            print(part["answer"], end="", flush=True)
        else:
            print(part, end="")
