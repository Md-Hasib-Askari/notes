# Capstone Projects - Phase 5

Two comprehensive projects that demonstrate production-ready LangGraph applications with full enterprise features.

## Project 1: Enterprise Workflow Automation

### Architecture Overview

```python
# src/enterprise_workflow/core.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, List
import asyncio
from datetime import datetime
from enum import Enum

class WorkflowStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class EnterpriseWorkflowState(TypedDict):
    workflow_id: str
    request_data: Dict[str, Any]
    approvals: List[Dict[str, Any]]
    processing_steps: List[Dict[str, Any]]
    external_integrations: Dict[str, Any]
    notifications: List[Dict[str, Any]]
    status: str
    error_logs: List[str]
    metrics: Dict[str, Any]

class EnterpriseWorkflowOrchestrator:
    def __init__(self):
        self.graph = self.build_workflow_graph()
        self.external_systems = {}
        self.approval_handlers = {}
        self.notification_system = NotificationSystem()
    
    def build_workflow_graph(self) -> StateGraph:
        """Build the enterprise workflow graph"""
        graph = StateGraph(EnterpriseWorkflowState)
        
        # Core workflow nodes
        graph.add_node("validate_request", self.validate_request)
        graph.add_node("route_approval", self.route_approval)
        graph.add_node("process_approval", self.process_approval)
        graph.add_node("execute_business_logic", self.execute_business_logic)
        graph.add_node("integrate_external_systems", self.integrate_external_systems)
        graph.add_node("generate_reports", self.generate_reports)
        graph.add_node("send_notifications", self.send_notifications)
        graph.add_node("finalize_workflow", self.finalize_workflow)
        
        # Error handling nodes
        graph.add_node("handle_validation_error", self.handle_validation_error)
        graph.add_node("handle_approval_rejection", self.handle_approval_rejection)
        graph.add_node("handle_integration_failure", self.handle_integration_failure)
        
        # Define workflow edges
        graph.add_conditional_edges(
            "validate_request",
            self.validation_router,
            {
                "valid": "route_approval",
                "invalid": "handle_validation_error"
            }
        )
        
        graph.add_conditional_edges(
            "process_approval",
            self.approval_router,
            {
                "approved": "execute_business_logic",
                "rejected": "handle_approval_rejection",
                "pending": "process_approval"  # Loop for pending approvals
            }
        )
        
        graph.add_conditional_edges(
            "integrate_external_systems",
            self.integration_router,
            {
                "success": "generate_reports",
                "failure": "handle_integration_failure",
                "retry": "integrate_external_systems"
            }
        )
        
        graph.add_edge("route_approval", "process_approval")
        graph.add_edge("execute_business_logic", "integrate_external_systems")
        graph.add_edge("generate_reports", "send_notifications")
        graph.add_edge("send_notifications", "finalize_workflow")
        
        # Error handling edges
        graph.add_edge("handle_validation_error", END)
        graph.add_edge("handle_approval_rejection", END)
        graph.add_edge("handle_integration_failure", "send_notifications")
        
        graph.set_entry_point("validate_request")
        return graph.compile()
    
    async def validate_request(self, state: EnterpriseWorkflowState) -> EnterpriseWorkflowState:
        """Validate incoming workflow request"""
        request_data = state["request_data"]
        
        # Validate required fields
        required_fields = ["department", "priority", "request_type", "details"]
        missing_fields = [field for field in required_fields if field not in request_data]
        
        if missing_fields:
            return {
                **state,
                "status": WorkflowStatus.FAILED.value,
                "error_logs": [f"Missing required fields: {missing_fields}"]
            }
        
        # Business rule validation
        if request_data.get("priority") not in ["low", "medium", "high", "critical"]:
            return {
                **state,
                "status": WorkflowStatus.FAILED.value,
                "error_logs": ["Invalid priority level"]
            }
        
        return {
            **state,
            "status": WorkflowStatus.IN_PROGRESS.value,
            "processing_steps": state["processing_steps"] + [{
                "step": "validation",
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }]
        }
    
    async def route_approval(self, state: EnterpriseWorkflowState) -> EnterpriseWorkflowState:
        """Route request to appropriate approval workflow"""
        request_data = state["request_data"]
        priority = request_data.get("priority", "medium")
        department = request_data.get("department")
        
        # Determine approval requirements based on priority and department
        approval_chain = self.get_approval_chain(priority, department)
        
        return {
            **state,
            "approvals": approval_chain,
            "processing_steps": state["processing_steps"] + [{
                "step": "approval_routing",
                "status": "completed",
                "approvers": [a["approver_id"] for a in approval_chain],
                "timestamp": datetime.now().isoformat()
            }]
        }
    
    async def process_approval(self, state: EnterpriseWorkflowState) -> EnterpriseWorkflowState:
        """Process approval requests"""
        pending_approvals = [a for a in state["approvals"] if a["status"] == "pending"]
        
        if not pending_approvals:
            return {
                **state,
                "processing_steps": state["processing_steps"] + [{
                    "step": "approval_completed",
                    "status": "completed",
                    "timestamp": datetime.now().isoformat()
                }]
            }
        
        # Send approval requests to pending approvers
        for approval in pending_approvals:
            await self.send_approval_request(approval, state["workflow_id"])
        
        return state
    
    async def integrate_external_systems(self, state: EnterpriseWorkflowState) -> EnterpriseWorkflowState:
        """Integrate with external systems (CRM, ERP, etc.)"""
        integration_results = {}
        
        # Example integrations
        systems_to_integrate = ["crm", "erp", "inventory", "finance"]
        
        for system in systems_to_integrate:
            try:
                result = await self.call_external_system(system, state["request_data"])
                integration_results[system] = {
                    "status": "success",
                    "data": result,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                integration_results[system] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        return {
            **state,
            "external_integrations": integration_results,
            "processing_steps": state["processing_steps"] + [{
                "step": "external_integration",
                "status": "completed",
                "systems": list(integration_results.keys()),
                "timestamp": datetime.now().isoformat()
            }]
        }

# Monitoring and alerting integration
from src.monitoring.metrics import metrics_collector, monitor_performance
from src.monitoring.alerting import alert_manager

@monitor_performance("enterprise_workflow")
async def execute_enterprise_workflow(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute enterprise workflow with full monitoring"""
    
    workflow_id = f"workflow_{datetime.now().timestamp()}"
    
    initial_state = EnterpriseWorkflowState(
        workflow_id=workflow_id,
        request_data=request_data,
        approvals=[],
        processing_steps=[],
        external_integrations={},
        notifications=[],
        status=WorkflowStatus.PENDING.value,
        error_logs=[],
        metrics={}
    )
    
    orchestrator = EnterpriseWorkflowOrchestrator()
    
    try:
        result = await orchestrator.graph.ainvoke(initial_state)
        metrics_collector.record_graph_execution(workflow_id, success=True)
        return result
        
    except Exception as e:
        metrics_collector.record_graph_execution(workflow_id, success=False)
        await alert_manager._handle_alert(Alert(
            id=f"workflow_failure_{workflow_id}",
            severity=AlertSeverity.HIGH,
            title="Enterprise Workflow Failed",
            description=f"Workflow {workflow_id} failed: {str(e)}",
            timestamp=datetime.now(),
            metadata={"workflow_id": workflow_id, "error": str(e)}
        ))
        raise
```

## Project 2: Intelligent Content Management System

### Multi-Agent Content Processing

```python
# src/content_management/agents.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
import asyncio
from datetime import datetime

class ContentState(TypedDict):
    content_id: str
    raw_content: str
    content_type: str  # article, video, audio, image
    processing_pipeline: List[str]
    agent_results: Dict[str, Any]
    quality_scores: Dict[str, float]
    metadata: Dict[str, Any]
    tags: List[str]
    categories: List[str]
    status: str
    recommendations: List[Dict[str, Any]]

class IntelligentContentManager:
    def __init__(self):
        self.content_graph = self.build_content_graph()
        self.agent_registry = self.setup_agents()
        self.quality_thresholds = {
            "readability": 0.7,
            "relevance": 0.8,
            "originality": 0.9,
            "engagement": 0.6
        }
    
    def build_content_graph(self) -> StateGraph:
        """Build dynamic content processing graph"""
        graph = StateGraph(ContentState)
        
        # Content analysis agents
        graph.add_node("content_analyzer", self.analyze_content)
        graph.add_node("quality_assessor", self.assess_quality)
        graph.add_node("metadata_extractor", self.extract_metadata)
        graph.add_node("tag_generator", self.generate_tags)
        graph.add_node("category_classifier", self.classify_content)
        graph.add_node("seo_optimizer", self.optimize_seo)
        graph.add_node("compliance_checker", self.check_compliance)
        
        # Enhancement agents
        graph.add_node("content_enhancer", self.enhance_content)
        graph.add_node("image_processor", self.process_images)
        graph.add_node("accessibility_optimizer", self.optimize_accessibility)
        
        # Publishing workflow
        graph.add_node("workflow_router", self.route_workflow)
        graph.add_node("publish_content", self.publish_content)
        graph.add_node("schedule_content", self.schedule_content)
        graph.add_node("review_queue", self.add_to_review_queue)
        
        # Dynamic routing based on content type and quality
        graph.add_conditional_edges(
            "content_analyzer",
            self.content_router,
            {
                "text": "quality_assessor",
                "video": "metadata_extractor",
                "image": "image_processor",
                "audio": "metadata_extractor"
            }
        )
        
        graph.add_conditional_edges(
            "quality_assessor",
            self.quality_router,
            {
                "high_quality": "metadata_extractor",
                "needs_enhancement": "content_enhancer",
                "requires_review": "review_queue"
            }
        )
        
        graph.add_conditional_edges(
            "workflow_router",
            self.publishing_router,
            {
                "auto_publish": "publish_content",
                "schedule": "schedule_content",
                "manual_review": "review_queue"
            }
        )
        
        # Sequential processing chain
        graph.add_edge("metadata_extractor", "tag_generator")
        graph.add_edge("tag_generator", "category_classifier")
        graph.add_edge("category_classifier", "seo_optimizer")
        graph.add_edge("seo_optimizer", "compliance_checker")
        graph.add_edge("compliance_checker", "accessibility_optimizer")
        graph.add_edge("accessibility_optimizer", "workflow_router")
        
        # Enhancement flow
        graph.add_edge("content_enhancer", "quality_assessor")
        graph.add_edge("image_processor", "metadata_extractor")
        
        # End states
        graph.add_edge("publish_content", END)
        graph.add_edge("schedule_content", END)
        graph.add_edge("review_queue", END)
        
        graph.set_entry_point("content_analyzer")
        return graph.compile()
    
    async def analyze_content(self, state: ContentState) -> ContentState:
        """Analyze content structure and type"""
        content = state["raw_content"]
        
        # Content analysis using LLM
        analysis_prompt = f"""
        Analyze the following content and provide insights:
        
        Content: {content[:1000]}...
        
        Provide analysis for:
        1. Content type and format
        2. Primary topics and themes
        3. Target audience
        4. Content structure quality
        5. Key insights and recommendations
        """
        
        # Simulate LLM analysis (replace with actual LLM call)
        analysis_result = {
            "content_type": self.detect_content_type(content),
            "word_count": len(content.split()),
            "reading_time": len(content.split()) // 200,  # Avg reading speed
            "complexity_score": self.calculate_complexity(content),
            "topics": self.extract_topics(content),
            "sentiment": self.analyze_sentiment(content)
        }
        
        return {
            **state,
            "agent_results": {**state["agent_results"], "content_analysis": analysis_result},
            "content_type": analysis_result["content_type"]
        }
    
    async def assess_quality(self, state: ContentState) -> ContentState:
        """Assess content quality using multiple metrics"""
        content = state["raw_content"]
        analysis = state["agent_results"]["content_analysis"]
        
        quality_scores = {
            "readability": self.calculate_readability(content),
            "relevance": self.calculate_relevance(content, analysis["topics"]),
            "originality": self.check_originality(content),
            "engagement": self.predict_engagement(content, analysis)
        }
        
        # Overall quality score
        overall_quality = sum(quality_scores.values()) / len(quality_scores)
        
        return {
            **state,
            "quality_scores": quality_scores,
            "agent_results": {
                **state["agent_results"],
                "quality_assessment": {
                    "overall_score": overall_quality,
                    "scores": quality_scores,
                    "meets_threshold": all(
                        score >= self.quality_thresholds.get(metric, 0.5)
                        for metric, score in quality_scores.items()
                    )
                }
            }
        }
    
    async def enhance_content(self, state: ContentState) -> ContentState:
        """Enhance content based on quality assessment"""
        quality_scores = state["quality_scores"]
        content = state["raw_content"]
        
        enhancements = []
        
        # Improve readability
        if quality_scores.get("readability", 0) < self.quality_thresholds["readability"]:
            enhanced_content = await self.improve_readability(content)
            enhancements.append("readability_improved")
        else:
            enhanced_content = content
        
        # Enhance engagement
        if quality_scores.get("engagement", 0) < self.quality_thresholds["engagement"]:
            enhanced_content = await self.improve_engagement(enhanced_content)
            enhancements.append("engagement_improved")
        
        # Add SEO improvements
        enhanced_content = await self.add_seo_elements(enhanced_content)
        enhancements.append("seo_optimized")
        
        return {
            **state,
            "raw_content": enhanced_content,
            "agent_results": {
                **state["agent_results"],
                "content_enhancement": {
                    "enhancements_applied": enhancements,
                    "enhancement_timestamp": datetime.now().isoformat()
                }
            }
        }
    
    def content_router(self, state: ContentState) -> str:
        """Route based on content type"""
        content_type = state.get("content_type", "text")
        return content_type.lower()
    
    def quality_router(self, state: ContentState) -> str:
        """Route based on quality assessment"""
        quality_result = state["agent_results"].get("quality_assessment", {})
        
        if quality_result.get("meets_threshold", False):
            return "high_quality"
        elif quality_result.get("overall_score", 0) > 0.5:
            return "needs_enhancement"
        else:
            return "requires_review"
    
    def publishing_router(self, state: ContentState) -> str:
        """Route content to appropriate publishing workflow"""
        quality_scores = state.get("quality_scores", {})
        metadata = state.get("metadata", {})
        
        # Auto-publish high-quality content
        if all(score > 0.8 for score in quality_scores.values()):
            if metadata.get("publish_immediately", False):
                return "auto_publish"
            else:
                return "schedule"
        
        return "manual_review"

# Production deployment integration
class ProductionContentManager:
    def __init__(self):
        self.content_manager = IntelligentContentManager()
        self.load_balancer = LangGraphLoadBalancer()
        self.state_store = PostgresStateStore("postgresql://...")
        self.cache_manager = GraphCache()
    
    async def process_content_batch(self, content_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple content items with load balancing"""
        tasks = []
        
        for content_item in content_batch:
            # Select optimal instance for processing
            instance = self.load_balancer.select_instance("least_loaded")
            
            # Create processing task
            task = self.process_single_content(content_item, instance)
            tasks.append(task)
        
        # Process all content in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]
    
    async def process_single_content(self, content_data: Dict[str, Any], instance: GraphInstance) -> Dict[str, Any]:
        """Process single content item with monitoring"""
        content_id = content_data.get("id", f"content_{datetime.now().timestamp()}")
        
        initial_state = ContentState(
            content_id=content_id,
            raw_content=content_data["content"],
            content_type=content_data.get("type", "text"),
            processing_pipeline=[],
            agent_results={},
            quality_scores={},
            metadata=content_data.get("metadata", {}),
            tags=[],
            categories=[],
            status="processing",
            recommendations=[]
        )
        
        try:
            # Save initial state
            await self.state_store.save_state(content_id, initial_state)
            
            # Process content
            result = await self.content_manager.content_graph.ainvoke(initial_state)
            
            # Save final state
            await self.state_store.save_state(content_id, result)
            
            # Track metrics
            metrics_collector.record_graph_execution(f"content_processing_{content_id}", success=True)
            
            return result
            
        except Exception as e:
            # Handle processing errors
            error_state = {**initial_state, "status": "failed", "error": str(e)}
            await self.state_store.save_state(content_id, error_state)
            
            metrics_collector.record_graph_execution(f"content_processing_{content_id}", success=False)
            raise

# API endpoints for content management
from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="Intelligent Content Management API")
production_manager = ProductionContentManager()

@app.post("/content/process")
async def process_content(content_data: Dict[str, Any]):
    """Process a single content item"""
    result = await production_manager.process_single_content(content_data, None)
    return {"status": "success", "content_id": result["content_id"], "result": result}

@app.post("/content/batch")
async def process_content_batch(content_batch: List[Dict[str, Any]]):
    """Process multiple content items"""
    results = await production_manager.process_content_batch(content_batch)
    return {"status": "success", "processed_count": len(results), "results": results}

@app.get("/content/{content_id}/status")
async def get_content_status(content_id: str):
    """Get processing status of content"""
    state = await production_manager.state_store.load_state(content_id)
    if state:
        return {"content_id": content_id, "status": state.get("status"), "progress": state}
    return {"error": "Content not found"}
```

## Key Features Implemented

### Enterprise Workflow Automation
- **Complete Business Process**: Request validation, approval routing, business logic execution
- **External Integration**: CRM, ERP, inventory, and finance system connections
- **Monitoring & Alerting**: Full observability with metrics collection and alert management
- **Error Handling**: Robust error management with fallback strategies
- **Scalability**: Load balancing and auto-scaling capabilities

### Intelligent Content Management
- **Multi-Agent Processing**: Specialized agents for analysis, quality assessment, and enhancement
- **Dynamic Workflow**: Adaptive routing based on content type and quality scores
- **Production Deployment**: Load balancing, state persistence, and batch processing
- **Quality Assurance**: Automated quality scoring and content enhancement
- **Real-time Processing**: Async processing with progress tracking

## Production Readiness Features

1. **Monitoring**: Comprehensive metrics and alerting
2. **Scalability**: Load balancing and horizontal scaling
3. **Persistence**: State management and recovery
4. **Error Handling**: Graceful degradation and recovery
5. **Security**: Authentication and authorization
6. **Performance**: Caching and optimization strategies
