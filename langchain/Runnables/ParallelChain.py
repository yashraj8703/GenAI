from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from typing import Literal
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel

import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

model1= ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite',
                             google_api_key=os.getenv("GEMINI_API_KEY"))
model2= ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite',
                             google_api_key=os.getenv("GEMINI_API_KEY"))


prompt1=PromptTemplate(
    template="Create a detailed notes on given article:\n {topic}",
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template="Create 5 question and answer on given article:\n {topic}",
    input_variables=['topic']
)
class StudyDoc(BaseModel):
    title: str
    notes: str
    quiz: str

parser = PydanticOutputParser(pydantic_object=StudyDoc)

prompt3 = PromptTemplate(
    template="""Merge the provided notes and quiz into a structured JSON.
Return strictly in this JSON format:
{format_instructions}
notes: {notes}
quiz: {quiz}""",
    input_variables=["notes", "quiz"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

parallelChain=RunnableParallel(
    {
        'notes':RunnableSequence(prompt1,model1,StrOutputParser()),
        'quiz':RunnableSequence(prompt2,model2,StrOutputParser())
    }
)

sequenceChain=RunnableSequence(prompt3,model1,parser)

chain=RunnableSequence(parallelChain,sequenceChain)
result=chain.invoke("""
The Power of Photosynthesis: How Plants Make Food
Life on Earth, in all its complexity and variety, is fundamentally powered by the sun. But how does the energy from a star millions of miles away get into the food we eat? The answer lies in a remarkable and elegant process called photosynthesis. Performed by plants, algae, and some bacteria, photosynthesis converts light energy into chemical energy, creating the foundation for nearly all life on our planet.

What Is Photosynthesis?
At its core, photosynthesis is a chemical process where light energy is used to convert carbon dioxide and water into glucose (a type of sugar) and oxygen. The glucose serves as food or energy for the plant, while the oxygen is released into the atmosphere as a byproduct.

The overall chemical equation for photosynthesis is:

6CO₂ (Carbon Dioxide) + 6H₂O (Water) + Light Energy → C₆H₁₂O₆ (Glucose) + 6O₂ (Oxygen)

This simple equation summarizes a highly complex series of reactions occurring within the plant's cells.

The Site of Photosynthesis: Chloroplasts
Photosynthesis takes place inside specialized organelles called chloroplasts, which are found in the cells of plant leaves and stems. These chloroplasts contain a green pigment called chlorophyll. Chlorophyll is crucial because it is responsible for absorbing the light energy from the sun. The green color we see in plants is due to chlorophyll reflecting green light while absorbing red and blue light.

The Two Stages of the Process
Photosynthesis is not a single event but is divided into two main stages:

The Light-Dependent Reactions: As the name suggests, this stage requires sunlight. When sunlight strikes the chlorophyll in the chloroplasts, the light energy is captured and used to split water molecules (H₂O). This process releases oxygen (O₂) into the atmosphere. The energy captured from the sunlight is temporarily stored in two energy-carrying molecules: ATP (adenosine triphosphate) and NADPH (nicotinamide adenine dinucleotide phosphate).

The Light-Independent Reactions (The Calvin Cycle): This second stage does not directly require light but depends on the products from the first stage. Using the energy from ATP and NADPH, this cycle captures carbon dioxide (CO₂) from the air. Through a series of chemical reactions, the CO₂ is "fixed" or converted into glucose (C₆H₁₂O₆). This sugar molecule can be used immediately by the plant for energy or stored for later use, often as starch.

Why Is Photosynthesis So Important?
The significance of photosynthesis extends far beyond plants. It is arguably the most important biological process on Earth for several reasons:

Foundation of Food Webs: Photosynthetic organisms, known as producers, form the base of most of the world's food webs. Herbivores get their energy by eating plants, and carnivores get their energy by eating herbivores.

Production of Oxygen: The oxygen we breathe is almost entirely a byproduct of photosynthesis. Billions of years ago, photosynthetic microbes transformed Earth's atmosphere into one that could support complex life.

Role in the Carbon Cycle: By absorbing carbon dioxide, a major greenhouse gas, photosynthesis plays a critical role in regulating Earth's climate.

In conclusion, photosynthesis is a vital process that sustains life by converting sunlight into usable energy, producing the food we eat and the air we breathe. It is a perfect example of nature's efficiency and a reminder of our planet's delicate balance.
""")
print(result.notes)
print(result.quiz)