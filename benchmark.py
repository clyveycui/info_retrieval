import interactive_llm

def ask(question: str) -> str:
    response = sim_user.generate(question)
    return response.str

def check_answers(answers: list) -> float:
    
INSTRUCTION_PROMPT


def main():
    sim_user = interactive_llm.model