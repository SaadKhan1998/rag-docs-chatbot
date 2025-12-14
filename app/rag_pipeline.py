from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

from .config import (
    OPENAI_API_KEY,
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    VECTORSTORE_DIR,
)


def get_vectorstore():
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL_NAME,
    )

    return Chroma(
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )


def build_rag_chain():
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=0.1,
    )

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know."

Question:
{question}

Context:
{context}

Answer:
""".strip(),
    )

    def rag_call(question: str):
        # Retrieve relevant documents
        docs = retriever.invoke(question)

        # Build context
        context = "\n\n".join(d.page_content for d in docs)

        # Format prompt
        formatted_prompt = prompt.format(
            question=question,
            context=context,
        )

        # Call LLM
        response = llm.invoke(formatted_prompt)
        answer = response.content

        # Extract sources
        sources = list({d.metadata.get("source", "unknown") for d in docs})

        return {
            "answer": answer,
            "sources": sources,
        }

    return rag_call
