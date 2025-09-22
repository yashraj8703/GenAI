from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
generator_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
evaluator_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
optimizer_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

class TweetEvaluation(BaseModel):
    evaluation:Literal['aproved','needs_improvement'] = Field(...,description="The final verdict on the tweet, either 'approved' or 'needs_improvement'.")
    feedback:str=Field(..., description="feedback for tweet")

struct_llm=evaluator_llm.with_structured_output(TweetEvaluation)


class TweetState(TypedDict):
    topic:str
    tweet:str
    evaluation:Literal['approved','needs_improvement']
    iteration:int
    max_iteration:int
    feedback:str

def generate_tweet(state:TweetState):
    messages = [
                    SystemMessage(
                        content="You are a funny and clever Twitter/X influencer."
                    ),
                    HumanMessage(
                        content=f"""
                Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

                Rules:
                - Do NOT use question-answer format.
                - Max 280 characters.
                - Use observational humor, irony, sarcasm, or cultural references.
                - Think in meme logic, punchlines, or relatable takes.
                - Use simple, day to day english
                - This is version {state['iteration'] + 1}.
                """
                    ),
                ]
    result=generator_llm.invoke(messages).content

    return {'tweet':result}
    
def evaluate_tweet(state:TweetState):
    messages = [
                    SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
                    HumanMessage(content=f"""
                Evaluate the following tweet:

                Tweet: "{state['tweet']}"

                Use the criteria below to evaluate the tweet:

                1. Originality - Is this fresh, or have you seen it a hundred times before?
                2. Humor - Did it genuinely make you smile, laugh, or chuckle?
                3. Punchiness - Is it short, sharp, and scroll-stopping?
                4. Virality Potential - Would people retweet or share it?
                5. Format - Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

                Auto-reject if:
                - It's written in question-answer format (e.g., "Why did..." or "what happens when...")
                - It exceeds 280 characters
                - It reads like a traditional setup-punchline joke
                - Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., "Masterpieces of the auntie-uncle universe" or vague summaries)

                ### Respond ONLY in structured format:
                - evaluation: "approved" or "needs_improvement"
                - feedback: One paragraph explaining the strengths and weaknesses
                """)
                ]
    result=struct_llm.invoke(messages)
    return {'evaluation':result.evaluation,'feedback':result.feedback}


def optimize_tweet(state:TweetState):
    messages = [
                    SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
                    HumanMessage(content=f"""
                Improve the tweet based on this feedback:
                "{state['feedback']}"

                Topic: "{state['topic']}"
                Original Tweet:
                {state['tweet']}

                Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
                """)
                ]
    result=optimizer_llm.invoke(messages).content
    iteration=state['iteration']+1
    return {'tweet':result,'iteration':iteration}


def Evaluation(state:TweetState):
    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'needs_improvement'

graph=StateGraph(TweetState)

graph.add_node('generation',generate_tweet)
graph.add_node('evaluate',evaluate_tweet)
graph.add_node('optimize',optimize_tweet)

graph.add_edge(START,'generation')
graph.add_edge('generation','evaluate')
graph.add_conditional_edges('evaluate',Evaluation,{'approved':END,'needs_improvement':'optimize'})
graph.add_edge('evaluate',END)

workflow=graph.compile()

initial_state={
    'topic':'Laravel',
    'iteration':1,
    'max_iteration':5
}
answer=workflow.invoke(initial_state)
print(answer['topic'])
print(answer['tweet'])
print(answer['iteration'])
print(answer['feedback'])
# print(answer)