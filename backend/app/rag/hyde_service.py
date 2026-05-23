from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.app.utils.common import loadPrompt


def create_hyde_chain(chat_model):
    hyde_prompt = PromptTemplate.from_template(loadPrompt("hyde_prompt.txt"))
    hyde_chain = hyde_prompt | chat_model | StrOutputParser()

class HyDEService:
    @staticmethod
    def generate_hypothetical_doc(query:str) -> str:
        