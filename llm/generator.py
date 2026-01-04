from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from llm.prompt import get_prompt

def generate_answer(context, question):
    llm = ChatOpenAI(temperature=0)
    prompt = get_prompt()

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(context=context, question=question)
