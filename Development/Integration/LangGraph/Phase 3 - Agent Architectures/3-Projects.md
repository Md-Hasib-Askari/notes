# Phase 3 Projects

## Project 1: Research and Writing Pipeline

### Overview
Build a sophisticated pipeline with specialized agents: Research Agent → Analysis Agent → Writer Agent, implementing review and revision cycles.

```python
from typing import TypedDict, List, Dict, Optional, Any
from langgraph.graph import StateGraph
from datetime import datetime

class PipelineState(TypedDict):
    topic: str
    research_query: str
    research_results: List[Dict[str, Any]]
    analysis_report: Dict[str, Any]
    draft_content: str
    review_scores: Dict[str, float]
    revision_notes: List[str]
    final_content: str
    pipeline_stage: str
    revision_count: int
    max_revisions: int

def research_agent(state: PipelineState) -> PipelineState:
    """Comprehensive research agent with multiple data sources."""
    topic = state["topic"]
    
    # Simulate multi-source research
    research_results = [
        {
            "source": "academic_papers",
            "title": f"Academic Analysis of {topic}",
            "content": f"Scholarly research indicates {topic} has significant implications...",
            "credibility": 0.9,
            "date": "2024-01-15"
        },
        {
            "source": "industry_reports", 
            "title": f"Industry Report: {topic} Trends",
            "content": f"Market analysis shows {topic} is experiencing rapid growth...",
            "credibility": 0.8,
            "date": "2024-02-01"
        },
        {
            "source": "expert_interviews",
            "title": f"Expert Opinions on {topic}",
            "content": f"Leading experts believe {topic} will transform the industry...",
            "credibility": 0.85,
            "date": "2024-02-10"
        },
        {
            "source": "statistical_data",
            "title": f"Statistical Overview of {topic}",
            "content": f"Data shows 40% growth in {topic} adoption over the past year...",
            "credibility": 0.95,
            "date": "2024-02-15"
        }
    ]
    
    return {
        **state,
        "research_results": research_results,
        "pipeline_stage": "research_completed"
    }

def analysis_agent(state: PipelineState) -> PipelineState:
    """Advanced analysis agent that synthesizes research."""
    research = state["research_results"]
    
    if not research:
        return {**state, "pipeline_stage": "analysis_failed"}
    
    # Analyze research quality and themes
    total_credibility = sum(item["credibility"] for item in research)
    avg_credibility = total_credibility / len(research)
    
    # Extract key themes
    themes = []
    for item in research:
        if "growth" in item["content"]:
            themes.append("market_expansion")
        if "transform" in item["content"]:
            themes.append("industry_disruption")
        if "implications" in item["content"]:
            themes.append("broad_impact")
    
    # Generate analysis report
    analysis_report = {
        "total_sources": len(research),
        "average_credibility": avg_credibility,
        "key_themes": list(set(themes)),
        "source_breakdown": {
            item["source"]: item["credibility"] for item in research
        },
        "recommendations": [
            "Focus on market expansion trends",
            "Highlight expert consensus",
            "Include statistical evidence",
            "Address potential challenges"
        ],
        "quality_score": avg_credibility * (len(research) / 5.0)  # Normalized quality
    }
    
    return {
        **state,
        "analysis_report": analysis_report,
        "pipeline_stage": "analysis_completed"
    }

def writer_agent(state: PipelineState) -> PipelineState:
    """Content generation agent that creates structured articles."""
    analysis = state["analysis_report"]
    research = state["research_results"]
    
    if not analysis or not research:
        return {**state, "pipeline_stage": "writing_failed"}
    
    # Generate structured content
    content = f"""# {state['topic']}: A Comprehensive Analysis

## Executive Summary
Based on analysis of {analysis['total_sources']} high-quality sources (average credibility: {analysis['average_credibility']:.2f}), this report examines the current state and future prospects of {state['topic']}.

## Key Findings
The research reveals several critical themes:
"""
    
    for theme in analysis['key_themes']:
        content += f"- {theme.replace('_', ' ').title()}\n"
    
    content += "\n## Detailed Analysis\n"
    
    # Add content from each research source
    for item in research:
        content += f"\n### {item['title']}\n"
        content += f"Source: {item['source']} (Credibility: {item['credibility']:.1f})\n"
        content += f"{item['content']}\n"
    
    content += f"\n## Recommendations\n"
    for rec in analysis['recommendations']:
        content += f"- {rec}\n"
    
    content += f"\n## Conclusion\n"
    content += f"The analysis demonstrates that {state['topic']} represents a significant opportunity with strong supporting evidence from multiple credible sources.\n"
    
    content += f"\n---\n*Report generated on {datetime.now().strftime('%Y-%m-%d')}*"
    content += f"\n*Quality Score: {analysis['quality_score']:.2f}/1.0*"
    
    return {
        **state,
        "draft_content": content,
        "pipeline_stage": "draft_completed"
    }

def reviewer_agent(state: PipelineState) -> PipelineState:
    """Comprehensive review agent with scoring system."""
    content = state["draft_content"]
    analysis = state["analysis_report"]
    
    if not content:
        return {**state, "pipeline_stage": "review_failed"}
    
    # Review criteria scoring
    review_scores = {}
    revision_notes = []
    
    # Content structure score
    sections = ["Executive Summary", "Key Findings", "Detailed Analysis", "Recommendations", "Conclusion"]
    structure_score = sum(1 for section in sections if section in content) / len(sections)
    review_scores["structure"] = structure_score
    
    if structure_score < 0.8:
        revision_notes.append("Improve content structure - missing key sections")
    
    # Content depth score
    word_count = len(content.split())
    depth_score = min(word_count / 500, 1.0)  # Target 500+ words
    review_scores["depth"] = depth_score
    
    if depth_score < 0.6:
        revision_notes.append("Content needs more depth and detail")
    
    # Source integration score
    research_count = len(state["research_results"])
    mentions = sum(1 for item in state["research_results"] if item["title"] in content)
    integration_score = mentions / research_count if research_count > 0 else 0
    review_scores["source_integration"] = integration_score
    
    if integration_score < 0.7:
        revision_notes.append("Better integration of research sources needed")
    
    # Clarity score (simplified heuristic)
    avg_sentence_length = word_count / max(content.count('.'), 1)
    clarity_score = max(0, 1.0 - (avg_sentence_length - 20) / 30)
    review_scores["clarity"] = clarity_score
    
    if clarity_score < 0.7:
        revision_notes.append("Improve content clarity and readability")
    
    # Overall score
    overall_score = sum(review_scores.values()) / len(review_scores)
    review_scores["overall"] = overall_score
    
    if overall_score < 0.75:
        revision_notes.append("Overall quality needs improvement")
    
    return {
        **state,
        "review_scores": review_scores,
        "revision_notes": revision_notes,
        "pipeline_stage": "review_completed"
    }

def revision_controller(state: PipelineState) -> str:
    """Control revision workflow."""
    scores = state.get("review_scores", {})
    overall_score = scores.get("overall", 0)
    
    # Check if revision is needed
    if overall_score >= 0.75:
        return "finalize"
    elif state["revision_count"] >= state["max_revisions"]:
        return "max_revisions_reached" 
    else:
        return "revise"

def revision_agent(state: PipelineState) -> PipelineState:
    """Agent that handles content revisions."""
    revision_notes = state["revision_notes"]
    current_content = state["draft_content"]
    
    # Simple revision simulation (in practice, use LLM for intelligent revision)
    revised_content = current_content
    
    # Address specific revision notes
    for note in revision_notes:
        if "structure" in note.lower():
            revised_content += "\n\n## Additional Analysis\nFurther structural improvements implemented."
        elif "depth" in note.lower():
            revised_content += "\n\nThis section provides additional depth and comprehensive coverage of the topic."
        elif "integration" in note.lower():
            revised_content += "\n\nAdditional research sources have been better integrated throughout the content."
    
    return {
        **state,
        "draft_content": revised_content,
        "revision_count": state["revision_count"] + 1,
        "pipeline_stage": "revision_completed"
    }

def finalizer_agent(state: PipelineState) -> PipelineState:
    """Finalize the content pipeline."""
    return {
        **state,
        "final_content": state["draft_content"],
        "pipeline_stage": "pipeline_completed"
    }

# Build the research and writing pipeline
def create_research_writing_pipeline():
    workflow = StateGraph(PipelineState)
    
    # Add agent nodes
    workflow.add_node("research", research_agent)
    workflow.add_node("analyze", analysis_agent)
    workflow.add_node("write", writer_agent)
    workflow.add_node("review", reviewer_agent)
    workflow.add_node("revise", revision_agent)
    workflow.add_node("finalize", finalizer_agent)
    
    # Set up workflow
    workflow.set_entry_point("research")
    workflow.add_edge("research", "analyze")
    workflow.add_edge("analyze", "write")
    workflow.add_edge("write", "review")
    
    # Conditional revision cycle
    workflow.add_conditional_edges(
        "review",
        revision_controller,
        {
            "revise": "revise",
            "finalize": "finalize",
            "max_revisions_reached": "finalize"
        }
    )
    
    workflow.add_edge("revise", "review")  # Back to review after revision
    workflow.add_edge("finalize", "__end__")
    
    return workflow.compile()

## Project 2: Multi-Agent Debate System

### Overview
Create agents with different perspectives that engage in structured debate and build consensus.

```python
class DebateState(TypedDict):
    topic: str
    debate_positions: Dict[str, str]
    arguments: List[Dict[str, Any]]
    counterarguments: List[Dict[str, Any]]
    current_round: int
    max_rounds: int
    consensus_score: float
    final_consensus: str
    debate_summary: str

def position_agent(position_name: str, stance: str):
    """Factory for creating position-specific debate agents."""
    def agent_logic(state: DebateState) -> DebateState:
        topic = state["topic"]
        current_round = state["current_round"]
        
        # Generate position-specific argument
        if stance == "supportive":
            argument = f"Round {current_round}: Strong support for {topic} due to numerous benefits including economic growth and innovation."
        elif stance == "critical":
            argument = f"Round {current_round}: Serious concerns about {topic} including potential risks and unintended consequences."
        else:  # neutral
            argument = f"Round {current_round}: Balanced perspective on {topic} considering both benefits and challenges."
        
        new_argument = {
            "agent": position_name,
            "stance": stance,
            "round": current_round,
            "content": argument,
            "strength": 0.7 + (current_round * 0.1)  # Arguments get stronger over time
        }
        
        return {
            **state,
            "arguments": state["arguments"] + [new_argument]
        }
    
    return agent_logic

def moderator_agent(state: DebateState) -> DebateState:
    """Moderator that facilitates debate and builds consensus."""
    arguments = state["arguments"]
    current_round = state["current_round"]
    
    # Analyze arguments from current round
    round_arguments = [arg for arg in arguments if arg["round"] == current_round]
    
    if len(round_arguments) >= 3:  # All agents have spoken
        # Calculate consensus score
        stances = [arg["stance"] for arg in round_arguments]
        strengths = [arg["strength"] for arg in round_arguments]
        
        # Simple consensus calculation
        avg_strength = sum(strengths) / len(strengths)
        stance_diversity = len(set(stances)) / 3.0  # Normalized diversity
        consensus_score = avg_strength * (1 - stance_diversity * 0.5)
        
        # Generate summary for this round
        summary = f"Round {current_round} Summary:\n"
        for arg in round_arguments:
            summary += f"- {arg['agent']} ({arg['stance']}): {arg['content']}\n"
        
        return {
            **state,
            "current_round": current_round + 1,
            "consensus_score": consensus_score,
            "debate_summary": state.get("debate_summary", "") + summary + "\n"
        }
    
    return state

def consensus_builder(state: DebateState) -> DebateState:
    """Build final consensus from debate."""
    arguments = state["arguments"]
    topic = state["topic"]
    
    # Analyze all arguments
    supportive_args = [arg for arg in arguments if arg["stance"] == "supportive"]
    critical_args = [arg for arg in arguments if arg["stance"] == "critical"]
    neutral_args = [arg for arg in arguments if arg["stance"] == "neutral"]
    
    # Build consensus
    consensus = f"Consensus on {topic}:\n\n"
    consensus += f"After {state['current_round']-1} rounds of debate, the following consensus emerged:\n\n"
    
    if len(supportive_args) > len(critical_args):
        consensus += "The debate leans toward a generally positive view, while acknowledging legitimate concerns.\n"
    elif len(critical_args) > len(supportive_args):
        consensus += "The debate highlights significant concerns that need to be addressed.\n"
    else:
        consensus += "The debate reveals a balanced perspective with valid points on multiple sides.\n"
    
    consensus += f"\nKey points of agreement:\n"
    consensus += f"- Importance of careful consideration\n"
    consensus += f"- Need for evidence-based decision making\n"
    consensus += f"- Value of multiple perspectives\n"
    
    consensus += f"\nFinal consensus score: {state['consensus_score']:.2f}/1.0"
    
    return {
        **state,
        "final_consensus": consensus
    }

def debate_controller(state: DebateState) -> str:
    """Control debate flow."""
    if state["current_round"] > state["max_rounds"]:
        return "build_consensus"
    elif state.get("consensus_score", 0) >= 0.8:
        return "build_consensus"
    else:
        return "continue_debate"

# Build debate system
def create_debate_system():
    workflow = StateGraph(DebateState)
    
    # Add agent nodes
    workflow.add_node("supporter", position_agent("Supporter", "supportive"))
    workflow.add_node("critic", position_agent("Critic", "critical"))
    workflow.add_node("neutral", position_agent("Neutral", "neutral"))
    workflow.add_node("moderator", moderator_agent)
    workflow.add_node("consensus", consensus_builder)
    
    # Debate flow
    workflow.set_entry_point("supporter")
    workflow.add_edge("supporter", "critic")
    workflow.add_edge("critic", "neutral")
    workflow.add_edge("neutral", "moderator")
    
    workflow.add_conditional_edges(
        "moderator",
        debate_controller,
        {
            "continue_debate": "supporter",
            "build_consensus": "consensus"
        }
    )
    
    workflow.add_edge("consensus", "__end__")
    
    return workflow.compile()

# Test both projects
def test_phase3_projects():
    # Test research pipeline
    pipeline_app = create_research_writing_pipeline()
    pipeline_result = pipeline_app.invoke({
        "topic": "Artificial Intelligence in Education",
        "research_query": "",
        "research_results": [],
        "analysis_report": {},
        "draft_content": "",
        "review_scores": {},
        "revision_notes": [],
        "final_content": "",
        "pipeline_stage": "starting",
        "revision_count": 0,
        "max_revisions": 2
    })
    
    print("Research Pipeline Result:")
    print(f"Stage: {pipeline_result['pipeline_stage']}")
    print(f"Final content length: {len(pipeline_result['final_content'])} characters")
    print(f"Revisions made: {pipeline_result['revision_count']}")
    
    # Test debate system
    debate_app = create_debate_system()
    debate_result = debate_app.invoke({
        "topic": "Remote Work Policy",
        "debate_positions": {},
        "arguments": [],
        "counterarguments": [],
        "current_round": 1,
        "max_rounds": 3,
        "consensus_score": 0.0,
        "final_consensus": "",
        "debate_summary": ""
    })
    
    print(f"\nDebate System Result:")
    print(f"Rounds completed: {debate_result['current_round']-1}")
    print(f"Arguments presented: {len(debate_result['arguments'])}")
    print(f"Consensus score: {debate_result['consensus_score']:.2f}")
    print(f"Final consensus:\n{debate_result['final_consensus']}")

if __name__ == "__main__":
    test_phase3_projects()
```

These projects demonstrate sophisticated agent coordination patterns essential for building production-ready multi-agent systems.
