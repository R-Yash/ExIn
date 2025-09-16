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

def node_interviewer(state: GraphState) -> GraphState:
    """
    Interviewer node that generates questions and processes responses.
    """
    # Check if this is the first interaction
    if not state.get("interviewer_messages"):
        # Initial greeting and setup
        candidate_input = CandidateInput(response="Start interview")
        response = interview_coordinator.run_sync(deps=candidate_input)
        
        state["interviewer_messages"] = state.get("interviewer_messages", [])
        state["candidate_messages"] = state.get("candidate_messages", [])
        
        # Add the initial message
        if response.data.message:
            state["interviewer_messages"].append(response.data.message)
        
        # Add the first question if generated
        if response.data.question:
            state["interviewer_messages"].append(response.data.question.question)
            state["current_question"] = 1
            state["skill_level"] = state.get("skill_level", SkillLevel.INTERMEDIATE)
    else:
        # Process the latest candidate response
        latest_response = state["candidate_messages"][-1] if state["candidate_messages"] else ""
        candidate_input = CandidateInput(response=latest_response)
        
        # Check if candidate wants to end interview
        if any(phrase in latest_response.lower() for phrase in ["end interview", "stop", "quit", "finish"]):
            state["done"] = True
            return state
        
        # Generate next question or message
        response = interview_coordinator.run_sync(deps=candidate_input)
        
        # Add any message from the coordinator
        if response.data.message:
            state["interviewer_messages"].append(response.data.message)
        
        # Add the question if generated
        if response.data.question:
            state["interviewer_messages"].append(response.data.question.question)
            state["current_question"] = state.get("current_question", 0) + 1
            
            # Update skill level based on performance (you might want to adjust this logic)
            if state["current_question"] > 3:
                # Simple heuristic: could be enhanced with actual evaluation scores
                state["skill_level"] = SkillLevel.INTERMEDIATE
    
    # Check if we've reached the question limit
    if state.get("current_question", 0) >= state.get("total_questions", 10):
        state["done"] = True
    
    return state

def node_evaluator(state: GraphState) -> GraphState:
    """
    Evaluator node that scores the latest question-response pair.
    """
    # Extract questions and responses from messages
    questions = []
    responses = []
    
    # Parse interviewer messages for questions
    for msg in state.get("interviewer_messages", []):
        # Assuming questions are stored directly as strings
        # You might need to adjust this based on your actual message format
        if "?" in msg:  # Simple heuristic to identify questions
            questions.append(msg)
    
    # Get candidate responses
    responses = state.get("candidate_messages", [])
    
    # Only evaluate if we have matching questions and responses
    if questions and responses and len(questions) <= len(responses):
        eval_input = EvaluationInput(
            questions=questions[:len(responses)],  # Match the number of responses
            responses=responses[:len(questions)]
        )
        
        # Run evaluation
        result = evaluator_agent.run_sync(deps=eval_input)
        
        state["evaluations"].append(result.data)
        
        if result.data.percentage > 80:
            state["skill_level"] = SkillLevel.ADVANCED
        elif result.data.percentage < 40:
            state["skill_level"] = SkillLevel.BEGINNER
        else:
            state["skill_level"] = SkillLevel.INTERMEDIATE
    
    return state


def node_summarizer(state: GraphState) -> GraphState:
    """
    Summarizer node that generates final interview summary and feedback.
    """
    # Extract questions and responses
    questions = []
    responses = state.get("candidate_messages", [])
    
    # Parse questions from interviewer messages
    for msg in state.get("interviewer_messages", []):
        if "?" in msg:  # Simple heuristic
            questions.append(msg)
    
    # Get the latest evaluation or create a comprehensive one
    evaluations = state.get("evaluations", [])
    
    if evaluations:
        # Use the most recent evaluation
        latest_eval = evaluations[-1]
    else:
        # Create a final evaluation if none exists
        if questions and responses:
            eval_input = EvaluationInput(
                questions=questions[:len(responses)],
                responses=responses[:len(questions)]
            )
            eval_result = evaluator_agent.run_sync(deps=eval_input)
            latest_eval = eval_result.data
        else:
            # Default evaluation if no Q&A pairs
            latest_eval = EvaluationResult(
                total_points=0.0,
                max_points=0.0,
                percentage=0.0,
                breakdown=[]
            )
    
    # Create summary input
    summary_input = SummaryInput(
        questions=questions[:len(responses)],
        responses=responses[:len(questions)],
        evaluation=latest_eval
    )
    
    # Generate summary
    summary_result = summarizer_agent.run_sync(deps=summary_input)
    
    # Store summary in state
    state["summary"] = summary_result.data
    
    # Persist to file using the tool
    candidate_name = state.get("name", "Unknown Candidate")
    role = "Excel Specialist"  # You might want to make this configurable
    
    # The persist_summary tool is called within the summarizer_agent
    # So we just need to ensure the summary is saved
    
    # Mark interview as complete
    state["done"] = True
    
    # Add final message to interviewer messages
    final_message = f"""
Interview Complete! 

Summary: {summary_result.data.summary}

Feedback: {summary_result.data.feedback}

Overall Score: {summary_result.data.overall_percentage:.1f}%

Thank you for participating in this Excel skills assessment!
"""
    state["interviewer_messages"].append(final_message)
    
    return state


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
