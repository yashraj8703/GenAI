from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int
    sr: float
    bpb: float
    boundary_percentage: float
    summary: str
    

def calculate_sr(state: BatsmanState):
    sr = (state['runs'] / state['balls']) * 100 
    return {'sr':sr}

def calculate_bpb(state: BatsmanState):
    bpb = state['balls'] / (state['fours'] + state['sixes'])
    return {'bpb':bpb}

def calculate_boundary_percentage(state: BatsmanState):
    bp = (((state['fours'] * 4) + (state['sixes'] * 6)) / state['runs']) * 100
    return {'boundary_percentage':bp}

def summary(state: BatsmanState):
    summary_text = (
        f"The batsman scored {state['runs']} runs from {state['balls']} balls "
        f"with a strike rate of {state['sr']:.2f}. "
        f"He hit {state['fours']} fours and {state['sixes']} sixes, "
        f"facing {state['bpb']:.2f} balls per boundary. "
        f"Boundaries contributed {state['boundary_percentage']:.2f}% of his runs."
    )
    return {'summary':summary_text}


graph = StateGraph(BatsmanState)

graph.add_node('calculate_sr', calculate_sr)
graph.add_node('calculate_bpb', calculate_bpb)
graph.add_node('calculate_boundary_percentage', calculate_boundary_percentage)
graph.add_node('summary', summary)

graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundary_percentage')

graph.add_edge('calculate_sr', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_boundary_percentage', 'summary')
graph.add_edge('summary', END)

workflow = graph.compile()
print(workflow)


initial_state={
    "runs": 60,
    "balls": 45,
    "fours": 6,
    "sixes": 2,
}

res=workflow.invoke(initial_state)
print(res['summary']) 