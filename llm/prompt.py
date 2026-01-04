from langchain.prompts import PromptTemplate

def get_prompt():
    template = """
You are an AI assistant.
Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
