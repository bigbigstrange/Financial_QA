from importlib import import_module

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.consts import (
    GENERATE,
    GIVE_UP,
    GRADE_DOCUMENTS,
    MAX_ITERATIONS,
    RETRIEVE,
    WEBSEARCH,
)
from graph.state import GraphState

load_dotenv()


class _LazyChain:
    def __init__(self, module_name: str, attribute: str):
        self.module_name = module_name
        self.attribute = attribute

    def invoke(self, payload):
        chain = getattr(import_module(self.module_name), self.attribute)
        return chain.invoke(payload)


question_router = _LazyChain("graph.chains.router", "question_router")
hallucination_grader = _LazyChain("graph.chains.hallucination_grader", "hallucination_grader")
answer_grader = _LazyChain("graph.chains.answer_grader", "answer_grader")


def retrieve(state):
    return import_module("graph.nodes.retrieve").retrieve(state)


def grade_documents(state):
    return import_module("graph.nodes.grade_documents").grade_documents(state)


def generate(state):
    return import_module("graph.nodes.generate").generate(state)


def web_search(state):
    return import_module("graph.nodes.web_search").web_search(state)


def give_up(state):
    return import_module("graph.nodes.give_up").give_up(state)


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        iterations = int(state.get("iterations", 0))
        if iterations >= MAX_ITERATIONS:
            print(f"---DECISION: NOT GROUNDED AFTER {iterations} ATTEMPTS, GIVE UP---")
            return "give_up"
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE


def build_app():
    workflow = StateGraph(GraphState)
    workflow.add_node(RETRIEVE, retrieve)
    workflow.add_node(GRADE_DOCUMENTS, grade_documents)
    workflow.add_node(GENERATE, generate)
    workflow.add_node(WEBSEARCH, web_search)
    workflow.add_node(GIVE_UP, give_up)

    workflow.set_conditional_entry_point(
        route_question,
        {WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE},
    )
    workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
    workflow.add_conditional_edges(
        GRADE_DOCUMENTS,
        decide_to_generate,
        {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
    )
    workflow.add_conditional_edges(
        GENERATE,
        grade_generation_grounded_in_documents_and_question,
        {
            "not supported": GENERATE,
            "useful": END,
            "not useful": WEBSEARCH,
            "give_up": GIVE_UP,
        },
    )
    workflow.add_edge(WEBSEARCH, GENERATE)
    workflow.add_edge(GIVE_UP, END)
    return workflow.compile()


app = build_app()
