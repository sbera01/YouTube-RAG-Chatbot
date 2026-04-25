from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from youtube_rag.config import Settings


PROMPT_TEMPLATE = """
You are a helpful assistant.
Answer using only the provided transcript context.
If the context is insufficient, say that politely and do not hallucinate.

Context:
{context}

Question:
{question}
""".strip()


def _format_docs(retrieved_docs) -> str:
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def build_rag_chain(retriever, settings: Settings):
    llm = HuggingFaceEndpoint(
        repo_id=settings.llm_repo_id,
        temperature=settings.llm_temperature,
    )
    model = ChatHuggingFace(llm=llm)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    retrieval_stage = RunnableParallel(
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
    )

    return retrieval_stage | prompt | model | StrOutputParser()
