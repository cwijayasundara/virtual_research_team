from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Get API key
load_dotenv()


def summary(content):
    llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=10000,
        chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a detailed summary of the following text for a research purpose:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, )

    return output

# text = search("What is ads/cft?")
# print(summary(text['knowledgeGraph']['description']))
