from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(temperature=0)

prompt = ChatPromptTemplate.from_template("""
다음은 팀 프로젝트 문서입니다:
{context}

사용자의 질문: {question}

문서를 참고하여 정확하고 간결하게 답변해주세요.
""")

parser = StrOutputParser()

def generate_node(state):
    question = state.question
    docs = state.retrieved_docs

    print(f"[Generate Node] 질문: {question}")
    print(f"[Generate Node] 문서 수: {len(docs)}")

    context = "\n\n".join([doc.page_content for doc in docs])
    
    # ✅ 여기에서 chain을 정의해야 해!
    chain = prompt | llm | parser

    answer = chain.invoke({"question": question, "context": context})
    print(f"[Generate Node] 응답: {answer}")
    return {**state.dict(), "answer": answer}
