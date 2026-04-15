from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .config import AppSettings
from .ingestion import ChromaDocumentStore


@dataclass(slots=True)
class RagResult:
    answer: str
    documents: list[Document]


class RagPipeline:
    """Single entrypoint service keeps retrieval and generation flow easy to inspect."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._store = ChromaDocumentStore(settings)
        self._llm = ChatOllama(
            model=self.settings.model_name,
            base_url=self.settings.ollama_base_url,
            temperature=0,
        )

    def answer_question(self, question: str) -> RagResult:
        docs = self._store.similarity_search(question, k=self.settings.retrieval_k)

        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an assistant. Use the following context to answer the user. "
                    "If the answer is not in the context, say you do not know.\n\nContext:\n{context}",
                ),
                ("human", "Question: {question}"),
            ]
        )

        chain = prompt | self._llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return RagResult(answer=answer, documents=docs)
