from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

class City(BaseModel):
    city:str=Field(description="Give details about the city.")
    information:str=Field(description="give description about city ")

    
parser=PydanticOutputParser(pydantic_object=City)


template = PromptTemplate(
    template="Provide details about the city {city}.\n{format_instructions}",
    input_variables=["city"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite',
                             google_api_key=os.getenv("GEMINI_API_KEY"))

model_chain=template | model | parser
result=model_chain.invoke({"city":"Bangalore"})


print(result)
# print(result.city)
# print(result.information)