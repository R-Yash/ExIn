import os
import json
from httpx import ResponseNotRead
import logfire
from typing import List, Set

from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel
from langgraph.graph import StateGraph
from google import genai

from dotenv import load_dotenv
from prompts import INTERVIEW_COORDINATOR_PROMPT, QUESTION_TOOL_PROMPT, EVALUATOR_PROMPT, SUMMARIZER_PROMPT
from models import (
    SkillLevel, InterviewResponse, CandidateInput, GraphState, QuestionOutput,
    EvaluationInput, EvaluationResult, ScoreBreakdown, SummaryInput, SummaryResult
)

from langgraph.graph import StateGraph, END

logfire.configure()  
logfire.instrument_pydantic_ai()

load_dotenv()

previous_topics: Set[str] = set()

provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel('gemini-2.0-flash', provider=provider)
client = genai.Client()

interview_coordinator = Agent[CandidateInput,InterviewResponse](
    model=model,
    deps_type=CandidateInput,
    output_type=InterviewResponse,
    system_prompt=INTERVIEW_COORDINATOR_PROMPT,
)

evaluator_agent = Agent[EvaluationInput, EvaluationResult](
    model=model,
    deps_type=EvaluationInput,
    output_type=EvaluationResult,
    system_prompt=EVALUATOR_PROMPT,
)

summarizer_agent = Agent[SummaryInput, SummaryResult](
    model=model,
    deps_type=SummaryInput,
    output_type=SummaryResult,
    system_prompt=SUMMARIZER_PROMPT,
)

@interview_coordinator.tool
def generate_excel_question(ctx: RunContext[CandidateInput], skill_level: SkillLevel) -> QuestionOutput:
    """Generate an Excel question with topics.

    Inputs:
        skill_level: The candidate's current `SkillLevel`.

    Output:
        QuestionOutput: contains `question` (str) and `topics` (List[str]).
    """
    prompt = QUESTION_TOOL_PROMPT.format(
        skill_level=skill_level.value,
        previous_topics=', '.join(sorted(previous_topics)) if previous_topics else 'None'
    )

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        text = response.text.strip()
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end+1])
        else:
            raise

    question = str(data.get("question", "")).strip()
    topics_list: List[str] = [str(t).strip() for t in data.get("topics", []) if str(t).strip()]

    if topics_list:
        previous_topics.update(topics_list)

    return QuestionOutput(question=question, topics=topics_list)

@evaluator_agent.tool
def score_latest(ctx: RunContext[EvaluationInput], questions: List[str], responses: List[str]) -> EvaluationResult:
    """Score the latest Q/A pair.

    Inputs:
        questions: Ordered list of questions asked.
        responses: Ordered list of candidate responses.

    Output:
        EvaluationResult with a single-item breakdown and totals.
    """
    if not questions or not responses:
        return EvaluationResult(total_points=0.0, max_points=0.0, percentage=0.0, breakdown=[])

    latest_q = questions[-1].strip()
    latest_a = responses[-1].strip()

    # Naive difficulty heuristic from keywords
    q_lower = latest_q.lower()
    if any(k in q_lower for k in ["array", "automation", "optimiz", "power", "index/match", "xlookup", "dynamic array"]):
        difficulty = "advanced"
        base = 12.0
    elif any(k in q_lower for k in ["pivot", "vlookup", "index", "match", "conditional", "what-if", "data table"]):
        difficulty = "intermediate"
        base = 8.0
    else:
        difficulty = "beginner"
        base = 5.0

    # Ask LLM for accuracy and relevance in strict JSON
    eval_prompt = (
        "Evaluate the response to the question. Return STRICT JSON with keys accuracy and relevance, both between 0 and 1.\n\n"
        f"Question: {latest_q}\nResponse: {latest_a}"
    )
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=eval_prompt)
    try:
        eval_data = json.loads(resp.text)
        acc = float(eval_data.get("accuracy", 0))
        rel = float(eval_data.get("relevance", 0))
    except Exception:
        # Fallback heuristic if parsing fails
        acc = 0.5
        rel = 0.5

    weighted = base * (0.5 * acc + 0.5 * rel)
    breakdown = [
        ScoreBreakdown(
            question=latest_q,
            response=latest_a,
            difficulty=difficulty,
            accuracy_score=acc,
            relevance_score=rel,
            weighted_points=weighted,
        )
    ]
    total = sum(b.weighted_points for b in breakdown)
    max_points = base
    pct = (total / max_points) * 100 if max_points > 0 else 0.0

    return EvaluationResult(total_points=total, max_points=max_points, percentage=pct, breakdown=breakdown)

@summarizer_agent.tool
def persist_summary(
    ctx: RunContext[SummaryInput],
    candidate_name: str,
    role: str,
    summary: str,
    feedback: str,
    overall_percentage: float,
    strengths: List[str],
    improvements: List[str],
) -> str:
    """Persist summary and scores to a JSONL file for internal evaluation.

    Appends a single JSON record per run into summaries.jsonl in the working directory.
    Returns the file path on success.
    """
    record = {
        "candidate_name": candidate_name,
        "role": role,
        "summary": summary,
        "feedback": feedback,
        "overall_percentage": overall_percentage,
        "strengths": strengths,
        "improvements": improvements,
    }
    out_path = os.path.join(os.getcwd(), "summaries.jsonl")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path

def done_condition(state: GraphState)->str:
    if state.done or state.current_question>=state.total_questions:
        return "summarizer"
    return "interviewer"

graph = StateGraph(GraphState)
graph.add_node("interviewer",node_interviewer)
graph.add_node("evaluator",node_evaluator)
graph.add_node("summarizer",node_summarizer)

graph.set_entry_point("interviewer")
graph.add_edge("interviewer","evaluator")
graph.add_edge("evaluator","interviewer")
graph.add_conditional_edges("interviewer",done_condition)
graph.add_edge("summarizer",END)

if __name__ == "__main__":
    app = graph.compile()
