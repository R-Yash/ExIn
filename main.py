import os
import json
from typing import List, Set, Optional
from datetime import datetime

from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel
from langgraph.graph import StateGraph, END
from google import genai
import logfire

from dotenv import load_dotenv
from prompts import INTERVIEW_COORDINATOR_PROMPT, QUESTION_TOOL_PROMPT, EVALUATOR_PROMPT, SUMMARIZER_PROMPT
from models import (
    SkillLevel, InterviewResponse, CandidateInput, GraphState, QuestionOutput,
    EvaluationInput, EvaluationResult, ScoreBreakdown, SummaryInput, SummaryResult
)

# Configure logging
logfire.configure()  
logfire.instrument_pydantic_ai()

# Load environment variables
load_dotenv()

# Global state for tracking topics
previous_topics: Set[str] = set()

# Initialize providers and models
provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel('gemini-2.0-flash', provider=provider)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize agents
interview_coordinator = Agent[CandidateInput, InterviewResponse](
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

    Args:
        skill_level: The candidate's current SkillLevel.

    Returns:
        QuestionOutput: contains question (str) and topics (List[str]).
    """
    prompt = QUESTION_TOOL_PROMPT.format(
        skill_level=skill_level.value,
        previous_topics=', '.join(sorted(previous_topics)) if previous_topics else 'None'
    )

    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    
    try:
        # Clean the response text
        text = response.text.strip()
        # Replace single quotes with double quotes for valid JSON
        text = text.replace("'", '"')
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from the text
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1].replace("'", '"')
            data = json.loads(json_str)
        else:
            # Fallback
            data = {
                "question": "What is your experience with Excel formulas?",
                "topics": ["formulas", "basic"]
            }

    question = str(data.get("question", "")).strip()
    topics_list: List[str] = [str(t).strip() for t in data.get("topics", []) if str(t).strip()]

    if topics_list:
        previous_topics.update(topics_list)

    return QuestionOutput(question=question, topics=topics_list)

@evaluator_agent.tool
def score_latest(ctx: RunContext[EvaluationInput], questions: List[str], responses: List[str]) -> EvaluationResult:
    """Score the latest Q/A pair.

    Args:
        questions: Ordered list of questions asked.
        responses: Ordered list of candidate responses.

    Returns:
        EvaluationResult with a single-item breakdown and totals.
    """
    if not questions or not responses:
        return EvaluationResult(total_points=0.0, max_points=0.0, percentage=0.0, breakdown=[])

    latest_q = questions[-1].strip()
    latest_a = responses[-1].strip()

    # Difficulty heuristic from keywords
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

    # Ask LLM for accuracy and relevance
    eval_prompt = (
        "Evaluate the response to the question. Return STRICT JSON with keys accuracy and relevance, both between 0 and 1.\n\n"
        f"Question: {latest_q}\nResponse: {latest_a}"
    )
    
    try:
        resp = client.models.generate_content(model="gemini-2.0-flash", contents=eval_prompt)
        eval_text = resp.text.strip().replace("'", '"')
        eval_data = json.loads(eval_text)
        acc = float(eval_data.get("accuracy", 0.5))
        rel = float(eval_data.get("relevance", 0.5))
    except Exception as e:
        print(f"Evaluation parsing error: {e}")
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
    """Persist summary and scores to a JSONL file."""
    record = {
        "timestamp": datetime.now().isoformat(),
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
    """Interviewer node that generates questions and processes responses."""
    
    # Initialize fields if not present
    if "interviewer_messages" not in state:
        state["interviewer_messages"] = []
    if "candidate_messages" not in state:
        state["candidate_messages"] = []
    if "evaluations" not in state:
        state["evaluations"] = []
    if "current_question" not in state:
        state["current_question"] = 0
    if "skill_level" not in state:
        state["skill_level"] = SkillLevel.INTERMEDIATE
    if "done" not in state:
        state["done"] = False
    
    # Check if this is the first interaction
    if state["current_question"] == 0:
        # Initial greeting
        greeting = (
            f"Hello {state.get('name', 'Candidate')}! Welcome to the Excel Skills Assessment Interview.\n"
            "I'll be asking you 10 questions to evaluate your Excel proficiency.\n"
            "Let's start with our first question.\n"
        )
        state["interviewer_messages"].append(greeting)
        
        # Generate first question
        candidate_input = CandidateInput(response="Start interview")
        response = interview_coordinator.run_sync(deps=candidate_input.dict())
        
        if response.output.question:
            state["interviewer_messages"].append(response.output.question.question)
            state["current_question"] = 1
    else:
        # Process the latest candidate response
        if state["candidate_messages"]:
            latest_response = state["candidate_messages"][-1]
            
            # Check if candidate wants to end
            if any(phrase in latest_response.lower() for phrase in ["end interview", "stop", "quit", "finish"]):
                state["done"] = True
                return state
            
            # Generate next question
            candidate_input = CandidateInput(response=latest_response)
            response = interview_coordinator.run_sync(deps=candidate_input.dict())
            
            # Add any feedback message
            if response.output.message:
                state["interviewer_messages"].append(response.output.message)
            
            # Add the next question
            if response.output.question and state["current_question"] < state.get("total_questions", 10):
                state["interviewer_messages"].append(response.output.question.question)
                state["current_question"] += 1
    
    # Check if we've reached the question limit
    if state["current_question"] >= state.get("total_questions", 10):
        state["done"] = True
    
    return state

def node_evaluator(state: GraphState) -> GraphState:
    """Evaluator node that scores the latest question-response pair."""
    
    # Extract questions from interviewer messages
    questions = []
    for msg in state.get("interviewer_messages", []):
        if "?" in msg and not msg.startswith("Hello") and not msg.startswith("Welcome"):
            questions.append(msg)
    
    responses = state.get("candidate_messages", [])
    
    # Only evaluate if we have matching Q&A pairs
    if questions and responses:
        # Take only the latest pair for evaluation
        eval_input = EvaluationInput(
            questions=[questions[-1]] if len(questions) > 0 else [],
            responses=[responses[-1]] if len(responses) > 0 else []
        )
        
        if eval_input.questions and eval_input.responses:
            result = evaluator_agent.run_sync(deps=eval_input.dict())
            state["evaluations"].append(result.data)
            
            # Adjust skill level based on performance
            if result.data.percentage > 80:
                state["skill_level"] = SkillLevel.ADVANCED
            elif result.data.percentage < 40:
                state["skill_level"] = SkillLevel.BEGINNER
    
    return state

def node_summarizer(state: GraphState) -> GraphState:
    """Summarizer node that generates final interview summary."""
    
    # Extract all questions and responses
    questions = []
    for msg in state.get("interviewer_messages", []):
        if "?" in msg and not msg.startswith("Hello") and not msg.startswith("Welcome"):
            questions.append(msg)
    
    responses = state.get("candidate_messages", [])
    
    # Calculate overall evaluation
    if state.get("evaluations"):
        total_points = sum(e.total_points for e in state["evaluations"])
        max_points = sum(e.max_points for e in state["evaluations"])
        overall_percentage = (total_points / max_points * 100) if max_points > 0 else 0
        
        # Create comprehensive evaluation
        all_breakdowns = []
        for e in state["evaluations"]:
            all_breakdowns.extend(e.breakdown)
        
        final_eval = EvaluationResult(
            total_points=total_points,
            max_points=max_points,
            percentage=overall_percentage,
            breakdown=all_breakdowns
        )
    else:
        final_eval = EvaluationResult(
            total_points=0.0,
            max_points=0.0,
            percentage=0.0,
            breakdown=[]
        )
    
    # Generate summary
    summary_input = SummaryInput(
        questions=questions,
        responses=responses,
        evaluation=final_eval
    )
    
    summary_result = summarizer_agent.run_sync(deps=summary_input.dict())
    state["summary"] = summary_result.data
    
    # Create final message
    final_message = f"""
Interview Complete! 

Summary: {summary_result.data.summary}

Feedback: {summary_result.data.feedback}

Overall Score: {summary_result.data.overall_percentage:.1f}%

Thank you for participating in this Excel skills assessment!
"""
    state["interviewer_messages"].append(final_message)
    state["done"] = True
    
    return state

def should_continue(state: GraphState) -> str:
    """Determine next node based on state."""
    if state.get("done", False):
        return "summarizer"
    elif state.get("current_question", 0) >= state.get("total_questions", 10):
        return "summarizer"
    elif len(state.get("candidate_messages", [])) > len(state.get("evaluations", [])):
        return "evaluator"
    else:
        return "interviewer"

def create_interview_graph():
    """Create and compile the interview graph."""
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("interviewer", node_interviewer)
    graph.add_node("evaluator", node_evaluator)
    graph.add_node("summarizer", node_summarizer)
    
    # Set entry point
    graph.set_entry_point("interviewer")
    
    # Add edges with proper flow
    graph.add_conditional_edges(
        "interviewer",
        should_continue,
        {
            "evaluator": "evaluator",
            "summarizer": "summarizer",
            "interviewer": "interviewer"
        }
    )
    
    graph.add_conditional_edges(
        "evaluator",
        lambda x: "interviewer" if not x.get("done", False) else "summarizer",
        {
            "interviewer": "interviewer",
            "summarizer": "summarizer"
        }
    )
    
    graph.add_edge("summarizer", END)
    
    return graph.compile()

# Testing Interface
class InterviewTester:
    """Interactive testing interface for the Excel Interview system."""
    
    def __init__(self):
        self.app = create_interview_graph()
        self.state = None
        self.reset_interview()
    
    def reset_interview(self):
        """Reset the interview state."""
        global previous_topics
        previous_topics.clear()
        
        self.state = {
            "name": "Test Candidate",
            "interviewer_messages": [],
            "candidate_messages": [],
            "current_question": 0,
            "total_questions": 10,
            "skill_level": SkillLevel.INTERMEDIATE,
            "done": False,
            "evaluations": [],
            "summary": None
        }
    
    def process_response(self, user_input: str) -> str:
        """Process a user response and return the next interviewer message."""
        if user_input and user_input.strip():
            self.state["candidate_messages"].append(user_input)
        
        # Run the graph
        result = self.app.invoke(self.state)
        self.state = result
        
        # Get the latest interviewer message
        if self.state["interviewer_messages"]:
            return self.state["interviewer_messages"][-1]
        return "Interview processing..."
    
    def run_interactive(self):
        """Run the interactive interview session."""
        print("\n" + "="*60)
        print("Excel Interview System - Interactive Testing Mode")
        print("="*60)
        print("\nCommands:")
        print("  'quit' - Exit the testing interface")
        print("  'reset' - Start a new interview")
        print("  'status' - Show current interview status")
        print("  'end interview' - Complete the current interview")
        print("\n" + "="*60)
        
        name = input("\nEnter candidate name (or press Enter for 'Test Candidate'): ").strip()
        if name:
            self.state["name"] = name
        
        # Start the interview
        initial_response = self.process_response("")
        print(f"\n[INTERVIEWER]: {initial_response}")
        
        while True:
            user_input = input("\n[YOU]: ").strip()
            
            if user_input.lower() == 'quit':
                print("\nExiting interview system...")
                break
            elif user_input.lower() == 'reset':
                self.reset_interview()
                print("\nInterview reset. Starting new session...")
                initial_response = aself.process_response("")
                print(f"\n[INTERVIEWER]: {initial_response}")
                continue
            elif user_input.lower() == 'status':
                print(f"\nInterview Status:")
                print(f"  Questions Asked: {self.state.get('current_question', 0)}")
                print(f"  Total Questions: {self.state.get('total_questions', 10)}")
                print(f"  Skill Level: {self.state.get('skill_level', 'Unknown')}")
                print(f"  Completed: {self.state.get('done', False)}")
                continue
            
            # Process the response
            interviewer_response = self.process_response(user_input)
            print(f"\n[INTERVIEWER]: {interviewer_response}")
            
            # Check if interview is done
            if self.state.get("done", False):
                print("\n" + "="*60)
                print("Interview completed! Check 'summaries.jsonl' for detailed results.")
                print("="*60)
                
                again = input("\nWould you like to start another interview? (yes/no): ").strip().lower()
                if again == 'yes':
                    self.reset_interview()
                    initial_response = self.process_response("")
                    print(f"\n[INTERVIEWER]: {initial_response}")
                else:
                    break
    
    def run_automated_test(self, responses: List[str]):
        """Run an automated test with predefined responses."""
        print("\n" + "="*60)
        print("Running Automated Test")
        print("="*60)
        
        # Start interview
        initial = self.process_response("")
        print(f"\n[INTERVIEWER]: {initial}")
        
        for i, response in enumerate(responses):
            print(f"\n[CANDIDATE]: {response}")
            interviewer_msg = self.process_response(response)
            print(f"\n[INTERVIEWER]: {interviewer_msg}")
            
            if self.state.get("done", False):
                break
        
        return self.state

# Main execution
def main():
    """Main function to run the testing interface."""
    tester = InterviewTester()
    
    print("\nSelect mode:")
    print("1. Interactive Interview")
    print("2. Automated Test")
    print("3. Quick Demo")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        tester.run_interactive()
    elif choice == "2":
        # Automated test with sample responses
        test_responses = [
            "I use SUM and AVERAGE functions regularly for basic calculations.",
            "VLOOKUP helps me find data in large tables by matching a lookup value.",
            "Pivot tables allow me to summarize and analyze large datasets quickly.",
            "I use conditional formatting to highlight cells based on certain criteria.",
            "INDEX and MATCH together provide more flexibility than VLOOKUP.",
            "I've used macros to automate repetitive tasks in Excel.",
            "Data validation helps ensure data integrity by restricting input types.",
            "I use array formulas for complex calculations across multiple cells.",
            "Power Query helps me import and transform data from various sources.",
            "I create dashboards using charts and slicers for interactive reporting."
        ]
        tester.run_automated_test(test_responses)
    elif choice == "3":
        # Quick demo with 3 questions
        tester.state["total_questions"] = 3
        demo_responses = [
            "I mainly use basic formulas like SUM and AVERAGE.",
            "I've heard of VLOOKUP but haven't used it much.",
            "I sometimes create simple charts for presentations."
        ]
        tester.run_automated_test(demo_responses)
    else:
        print("Invalid choice. Exiting...")

if __name__ == "__main__":
    main()