# Responsible AI

## Learning Objectives
- Understand the principles and frameworks of responsible AI development
- Learn to implement AI governance, accountability, and transparency measures
- Apply responsible AI practices throughout the ML lifecycle
- Design AI systems with ethical considerations and societal impact in mind
- Establish monitoring and governance frameworks for responsible AI deployment

## 1. Introduction to Responsible AI

### What is Responsible AI?
Responsible AI refers to the practice of developing and deploying artificial intelligence systems in a way that is ethical, transparent, accountable, and beneficial to society. It encompasses principles that ensure AI systems are fair, reliable, safe, and respect human values and rights.

### Core Principles of Responsible AI

#### 1. Fairness and Non-Discrimination
- Ensure AI systems treat all individuals and groups equitably
- Prevent bias and discrimination in AI decisions
- Promote inclusive and representative AI development

#### 2. Transparency and Explainability
- Make AI decision processes understandable
- Provide clear explanations for AI-driven outcomes
- Enable stakeholders to understand how AI systems work

#### 3. Accountability and Governance
- Establish clear responsibility for AI system outcomes
- Implement oversight and audit mechanisms
- Create governance frameworks for AI development and deployment

#### 4. Privacy and Security
- Protect personal data and user privacy
- Implement robust security measures
- Ensure data governance and compliance

#### 5. Human-Centered Design
- Keep humans in control of AI systems
- Design AI to augment rather than replace human judgment
- Ensure AI systems serve human needs and values

#### 6. Robustness and Safety
- Build reliable and resilient AI systems
- Test for edge cases and failure modes
- Implement fail-safe mechanisms

## 2. Responsible AI Framework Implementation

### AI Ethics Assessment Framework

```python
import pandas as pd
import numpy as np
from datetime import datetime
import json

class ResponsibleAIFramework:
    """Comprehensive framework for responsible AI implementation"""
    
    def __init__(self, project_name, stakeholders):
        self.project_name = project_name
        self.stakeholders = stakeholders
        self.assessments = {}
        self.governance_log = []
        
    def ethical_impact_assessment(self, use_case, data_sources, target_population, potential_harms):
        """Conduct ethical impact assessment"""
        
        assessment = {
            'timestamp': datetime.now(),
            'use_case': use_case,
            'data_sources': data_sources,
            'target_population': target_population,
            'potential_harms': potential_harms,
            'risk_levels': {},
            'mitigation_strategies': {},
            'approval_status': 'pending'
        }
        
        # Risk assessment categories
        risk_categories = [
            'discrimination_bias',
            'privacy_violation',
            'security_breach',
            'transparency_lack',
            'autonomy_reduction',
            'societal_impact'
        ]
        
        print(f"=== ETHICAL IMPACT ASSESSMENT: {self.project_name} ===")
        print(f"Use Case: {use_case}")
        print(f"Target Population: {target_population}")
        
        for category in risk_categories:
            risk_level = self._assess_risk_category(category, use_case, potential_harms)
            mitigation = self._suggest_mitigation(category, risk_level)
            
            assessment['risk_levels'][category] = risk_level
            assessment['mitigation_strategies'][category] = mitigation
            
            print(f"\n{category.replace('_', ' ').title()}:")
            print(f"  Risk Level: {risk_level}")
            print(f"  Mitigation: {mitigation}")
        
        self.assessments['ethical_impact'] = assessment
        return assessment
    
    def _assess_risk_category(self, category, use_case, potential_harms):
        """Assess risk level for specific category"""
        # Simplified risk assessment logic
        high_risk_indicators = {
            'discrimination_bias': ['hiring', 'lending', 'criminal justice', 'healthcare'],
            'privacy_violation': ['personal data', 'biometric', 'location', 'health'],
            'security_breach': ['financial', 'healthcare', 'government', 'infrastructure'],
            'transparency_lack': ['automated decision', 'black box', 'complex model'],
            'autonomy_reduction': ['autonomous', 'automated', 'replacement'],
            'societal_impact': ['employment', 'democracy', 'social services']
        }
        
        indicators = high_risk_indicators.get(category, [])
        risk_score = 0
        
        # Check use case against indicators
        for indicator in indicators:
            if indicator.lower() in use_case.lower():
                risk_score += 1
        
        # Check potential harms
        for harm in potential_harms:
            for indicator in indicators:
                if indicator.lower() in harm.lower():
                    risk_score += 1
        
        # Categorize risk level
        if risk_score >= 3:
            return 'high'
        elif risk_score >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _suggest_mitigation(self, category, risk_level):
        """Suggest mitigation strategies based on risk category and level"""
        mitigation_strategies = {
            'discrimination_bias': {
                'high': 'Implement bias testing, diverse datasets, fairness constraints',
                'medium': 'Regular bias monitoring, diverse review team',
                'low': 'Basic fairness testing, documentation'
            },
            'privacy_violation': {
                'high': 'Data minimization, anonymization, privacy-preserving techniques',
                'medium': 'Data governance policies, consent management',
                'low': 'Basic privacy controls, user notification'
            },
            'security_breach': {
                'high': 'Advanced security measures, regular audits, incident response',
                'medium': 'Standard security protocols, monitoring',
                'low': 'Basic security measures, access controls'
            },
            'transparency_lack': {
                'high': 'Explainable AI, decision audit trails, user interfaces',
                'medium': 'Model documentation, basic explanations',
                'low': 'Simple documentation, basic transparency'
            },
            'autonomy_reduction': {
                'high': 'Human oversight, manual override, user control',
                'medium': 'Human-in-the-loop design, user feedback',
                'low': 'User notification, basic controls'
            },
            'societal_impact': {
                'high': 'Stakeholder engagement, impact monitoring, gradual rollout',
                'medium': 'Community consultation, impact assessment',
                'low': 'Basic impact documentation, feedback collection'
            }
        }
        
        return mitigation_strategies.get(category, {}).get(risk_level, 'Standard best practices')
    
    def governance_checklist(self):
        """AI governance checklist"""
        
        checklist = {
            'governance_structure': [
                'AI ethics committee established',
                'Clear roles and responsibilities defined',
                'Escalation procedures documented',
                'Regular governance reviews scheduled'
            ],
            'development_practices': [
                'Ethical guidelines integrated into development',
                'Bias testing implemented',
                'Privacy by design principles followed',
                'Security measures implemented'
            ],
            'deployment_controls': [
                'Pre-deployment testing completed',
                'Impact assessment conducted',
                'Monitoring systems in place',
                'Rollback procedures defined'
            ],
            'ongoing_monitoring': [
                'Performance monitoring active',
                'Bias monitoring implemented',
                'User feedback collection',
                'Regular audits scheduled'
            ],
            'stakeholder_engagement': [
                'Affected communities consulted',
                'User education provided',
                'Feedback mechanisms established',
                'Transparency reports published'
            ]
        }
        
        print("=== AI GOVERNANCE CHECKLIST ===")
        completion_scores = {}
        
        for category, items in checklist.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            completed = 0
            for item in items:
                # In practice, this would check actual implementation status
                status = "✓" if np.random.random() > 0.3 else "✗"
                print(f"  {status} {item}")
                if status == "✓":
                    completed += 1
            
            completion_scores[category] = completed / len(items)
            print(f"  Completion: {completion_scores[category]:.1%}")
        
        overall_score = np.mean(list(completion_scores.values()))
        print(f"\nOverall Governance Score: {overall_score:.1%}")
        
        return checklist, completion_scores
    
    def stakeholder_impact_analysis(self):
        """Analyze impact on different stakeholders"""
        
        impact_analysis = {}
        
        for stakeholder in self.stakeholders:
            print(f"\n=== STAKEHOLDER IMPACT: {stakeholder.upper()} ===")
            
            # Define impact dimensions
            impact_dimensions = [
                'economic_impact',
                'social_impact',
                'privacy_impact',
                'autonomy_impact',
                'fairness_impact'
            ]
            
            stakeholder_impacts = {}
            
            for dimension in impact_dimensions:
                # Simplified impact assessment
                impact_level = np.random.choice(['positive', 'neutral', 'negative'], p=[0.4, 0.3, 0.3])
                magnitude = np.random.choice(['low', 'medium', 'high'], p=[0.4, 0.4, 0.2])
                
                stakeholder_impacts[dimension] = {
                    'level': impact_level,
                    'magnitude': magnitude,
                    'description': self._generate_impact_description(stakeholder, dimension, impact_level, magnitude)
                }
                
                print(f"{dimension.replace('_', ' ').title()}: {impact_level} ({magnitude})")
                print(f"  {stakeholder_impacts[dimension]['description']}")
            
            impact_analysis[stakeholder] = stakeholder_impacts
        
        return impact_analysis
    
    def _generate_impact_description(self, stakeholder, dimension, level, magnitude):
        """Generate impact description"""
        descriptions = {
            'users': {
                'economic_impact': f"{level.title()} {magnitude} effect on user costs/benefits",
                'social_impact': f"{level.title()} {magnitude} impact on social interactions",
                'privacy_impact': f"{level.title()} {magnitude} effect on personal privacy",
                'autonomy_impact': f"{level.title()} {magnitude} impact on user control and choice",
                'fairness_impact': f"{level.title()} {magnitude} effect on fair treatment"
            },
            'employees': {
                'economic_impact': f"{level.title()} {magnitude} impact on job security/wages",
                'social_impact': f"{level.title()} {magnitude} effect on workplace dynamics",
                'privacy_impact': f"{level.title()} {magnitude} impact on employee privacy",
                'autonomy_impact': f"{level.title()} {magnitude} effect on job autonomy",
                'fairness_impact': f"{level.title()} {magnitude} impact on workplace fairness"
            },
            'society': {
                'economic_impact': f"{level.title()} {magnitude} broader economic implications",
                'social_impact': f"{level.title()} {magnitude} societal changes",
                'privacy_impact': f"{level.title()} {magnitude} impact on societal privacy norms",
                'autonomy_impact': f"{level.title()} {magnitude} effect on collective autonomy",
                'fairness_impact': f"{level.title()} {magnitude} impact on social equity"
            }
        }
        
        return descriptions.get(stakeholder, {}).get(dimension, f"{level.title()} {magnitude} impact")

# Example usage
framework = ResponsibleAIFramework(
    project_name="AI Recruitment System",
    stakeholders=['job_candidates', 'hr_teams', 'hiring_managers', 'society']
)

# Conduct ethical impact assessment
ethical_assessment = framework.ethical_impact_assessment(
    use_case="Automated resume screening and candidate ranking for hiring decisions",
    data_sources=["resumes", "job descriptions", "historical hiring data"],
    target_population="Job candidates across all demographics",
    potential_harms=["discrimination against protected groups", "privacy violations", "reduced human judgment"]
)

# Check governance readiness
checklist, scores = framework.governance_checklist()

# Analyze stakeholder impacts
stakeholder_impacts = framework.stakeholder_impact_analysis()
```

## 3. AI Transparency and Explainability

### Transparency Implementation

```python
class AITransparencyFramework:
    """Framework for implementing AI transparency and explainability"""
    
    def __init__(self, model, model_type, use_case):
        self.model = model
        self.model_type = model_type
        self.use_case = use_case
        self.transparency_report = {}
    
    def generate_model_card(self, intended_use, limitations, performance_metrics, training_data_info):
        """Generate comprehensive model card for transparency"""
        
        model_card = {
            'model_details': {
                'name': f"{self.model_type} for {self.use_case}",
                'version': '1.0',
                'date_created': datetime.now().strftime('%Y-%m-%d'),
                'model_type': self.model_type,
                'intended_use': intended_use,
                'primary_use_cases': [self.use_case],
                'out_of_scope_uses': self._generate_out_of_scope_uses()
            },
            'training_data': training_data_info,
            'performance_metrics': performance_metrics,
            'limitations': limitations,
            'ethical_considerations': self._generate_ethical_considerations(),
            'recommendations': self._generate_recommendations()
        }
        
        self.transparency_report['model_card'] = model_card
        
        print("=== MODEL CARD ===")
        print(f"Model: {model_card['model_details']['name']}")
        print(f"Intended Use: {intended_use}")
        print(f"Limitations: {', '.join(limitations)}")
        
        return model_card
    
    def _generate_out_of_scope_uses(self):
        """Generate out-of-scope use cases"""
        common_out_of_scope = [
            "Use on populations not represented in training data",
            "High-stakes decisions without human oversight",
            "Use in domains requiring perfect accuracy",
            "Real-time applications without proper validation"
        ]
        return common_out_of_scope
    
    def _generate_ethical_considerations(self):
        """Generate ethical considerations"""
        return [
            "Potential for discriminatory outcomes",
            "Privacy implications of data use",
            "Need for human oversight in decision-making",
            "Regular monitoring for performance degradation"
        ]
    
    def _generate_recommendations(self):
        """Generate recommendations for responsible use"""
        return [
            "Implement bias monitoring and testing",
            "Provide explanations for important decisions",
            "Maintain human oversight and control",
            "Regular retraining and validation",
            "User education and feedback collection"
        ]
    
    def explainability_interface(self, instance_data, explanation_method='feature_importance'):
        """Create user-friendly explanation interface"""
        
        explanations = {
            'prediction': "Model prediction would be displayed here",
            'confidence': 0.85,
            'key_factors': [
                "Most important feature contributing to decision",
                "Second most important factor",
                "Third contributing factor"
            ],
            'what_if_scenarios': [
                "If feature X changed, prediction would be...",
                "If feature Y increased by 10%, outcome would be..."
            ],
            'similar_cases': [
                "Similar case 1 with different outcome",
                "Similar case 2 with same outcome"
            ]
        }
        
        print("=== EXPLANATION INTERFACE ===")
        print(f"Prediction: {explanations['prediction']}")
        print(f"Confidence: {explanations['confidence']:.2%}")
        print("\nKey Factors:")
        for factor in explanations['key_factors']:
            print(f"  • {factor}")
        
        return explanations
    
    def transparency_dashboard(self):
        """Create transparency dashboard for system monitoring"""
        
        dashboard_data = {
            'system_performance': {
                'accuracy': 0.87,
                'precision': 0.85,
                'recall': 0.89,
                'fairness_metrics': {
                    'demographic_parity': 0.92,
                    'equal_opportunity': 0.88
                }
            },
            'data_quality': {
                'completeness': 0.95,
                'consistency': 0.92,
                'timeliness': 0.98
            },
            'usage_statistics': {
                'total_predictions': 10000,
                'user_feedback_score': 4.2,
                'human_override_rate': 0.03
            },
            'alerts': [
                "Minor drift detected in feature distribution",
                "Fairness metric below threshold for Group A"
            ]
        }
        
        print("=== TRANSPARENCY DASHBOARD ===")
        print("System Performance:")
        for metric, value in dashboard_data['system_performance'].items():
            if isinstance(value, dict):
                print(f"  {metric}:")
                for sub_metric, sub_value in value.items():
                    print(f"    {sub_metric}: {sub_value:.3f}")
            else:
                print(f"  {metric}: {value:.3f}")
        
        if dashboard_data['alerts']:
            print("\nAlerts:")
            for alert in dashboard_data['alerts']:
                print(f"  ⚠️ {alert}")
        
        return dashboard_data

# Example transparency implementation
transparency_framework = AITransparencyFramework(
    model=None,  # Would be actual model
    model_type="Random Forest Classifier",
    use_case="Credit Risk Assessment"
)

# Generate model card
model_card = transparency_framework.generate_model_card(
    intended_use="Assess credit risk for loan applications",
    limitations=["Limited to specific demographic groups", "Requires manual review for edge cases"],
    performance_metrics={"accuracy": 0.87, "precision": 0.85, "recall": 0.89},
    training_data_info={"size": "100,000 records", "time_period": "2020-2023", "sources": "Bank records"}
)

# Create explanation interface
explanation = transparency_framework.explainability_interface(
    instance_data={"income": 50000, "credit_score": 720},
    explanation_method="SHAP"
)

# Generate transparency dashboard
dashboard = transparency_framework.transparency_dashboard()
```

## 4. Human-AI Collaboration

### Human-in-the-Loop Design

```python
class HumanAICollaboration:
    """Framework for human-AI collaboration and oversight"""
    
    def __init__(self, decision_type, risk_level):
        self.decision_type = decision_type
        self.risk_level = risk_level
        self.human_oversight_config = self._configure_oversight()
    
    def _configure_oversight(self):
        """Configure human oversight based on risk level"""
        
        oversight_configs = {
            'high': {
                'human_review_required': True,
                'ai_recommendation_only': True,
                'multiple_reviewer_required': True,
                'explanation_required': True,
                'audit_trail_required': True,
                'override_allowed': True
            },
            'medium': {
                'human_review_required': True,
                'ai_recommendation_only': False,
                'multiple_reviewer_required': False,
                'explanation_required': True,
                'audit_trail_required': True,
                'override_allowed': True
            },
            'low': {
                'human_review_required': False,
                'ai_recommendation_only': False,
                'multiple_reviewer_required': False,
                'explanation_required': False,
                'audit_trail_required': True,
                'override_allowed': True
            }
        }
        
        return oversight_configs.get(self.risk_level, oversight_configs['high'])
    
    def decision_workflow(self, ai_prediction, confidence, human_reviewer=None):
        """Implement human-AI decision workflow"""
        
        decision_log = {
            'timestamp': datetime.now(),
            'ai_prediction': ai_prediction,
            'confidence': confidence,
            'risk_level': self.risk_level,
            'human_review_required': self.human_oversight_config['human_review_required']
        }
        
        print(f"=== HUMAN-AI DECISION WORKFLOW ===")
        print(f"AI Prediction: {ai_prediction}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Risk Level: {self.risk_level}")
        
        # Check if human review is required
        if self.human_oversight_config['human_review_required']:
            print("\n🔍 Human review required")
            
            if self.human_oversight_config['explanation_required']:
                explanation = self._generate_explanation()
                print(f"Explanation: {explanation}")
                decision_log['explanation'] = explanation
            
            # Simulate human review
            if human_reviewer:
                human_decision = self._simulate_human_review(ai_prediction, confidence)
                decision_log['human_reviewer'] = human_reviewer
                decision_log['human_decision'] = human_decision
                decision_log['final_decision'] = human_decision
                
                print(f"Human Decision: {human_decision}")
                
                if human_decision != ai_prediction:
                    print("⚠️ Human override detected")
                    decision_log['override'] = True
                    decision_log['override_reason'] = "Human expert judgment"
            else:
                decision_log['final_decision'] = 'pending_human_review'
                print("⏳ Awaiting human review")
        else:
            decision_log['final_decision'] = ai_prediction
            print("✅ Automated decision approved")
        
        # Log decision
        if self.human_oversight_config['audit_trail_required']:
            self._log_decision(decision_log)
        
        return decision_log
    
    def _generate_explanation(self):
        """Generate explanation for AI decision"""
        explanations = [
            "Based on historical patterns and key risk factors",
            "Primary factors: income stability, credit history, debt ratio",
            "Confidence based on similarity to training examples"
        ]
        return "; ".join(explanations)
    
    def _simulate_human_review(self, ai_prediction, confidence):
        """Simulate human review process"""
        # In practice, this would involve actual human input
        # Simplified simulation: humans more likely to agree with high-confidence predictions
        if confidence > 0.9:
            return ai_prediction  # High confidence, likely agreement
        elif confidence > 0.7:
            return ai_prediction if np.random.random() > 0.2 else not ai_prediction
        else:
            return ai_prediction if np.random.random() > 0.4 else not ai_prediction
    
    def _log_decision(self, decision_log):
        """Log decision for audit trail"""
        print(f"📝 Decision logged: {decision_log['timestamp']}")
        # In practice, this would write to a database or audit system
    
    def feedback_loop(self, decision_id, outcome, user_feedback):
        """Implement feedback loop for continuous improvement"""
        
        feedback_data = {
            'decision_id': decision_id,
            'outcome': outcome,
            'user_feedback': user_feedback,
            'timestamp': datetime.now()
        }
        
        print(f"=== FEEDBACK COLLECTION ===")
        print(f"Decision ID: {decision_id}")
        print(f"Actual Outcome: {outcome}")
        print(f"User Feedback: {user_feedback}")
        
        # Analyze feedback for model improvement
        improvement_suggestions = self._analyze_feedback(feedback_data)
        
        return feedback_data, improvement_suggestions
    
    def _analyze_feedback(self, feedback_data):
        """Analyze feedback for improvement opportunities"""
        suggestions = []
        
        if feedback_data['user_feedback']['satisfaction'] < 3:
            suggestions.append("Consider improving explanation quality")
        
        if feedback_data['user_feedback']['trust'] < 3:
            suggestions.append("Enhance transparency and human oversight")
        
        if feedback_data['outcome'] != feedback_data.get('predicted_outcome'):
            suggestions.append("Review model performance and retraining needs")
        
        return suggestions

# Example human-AI collaboration
collaboration = HumanAICollaboration(
    decision_type="loan_approval",
    risk_level="high"
)

# Simulate decision workflow
decision_result = collaboration.decision_workflow(
    ai_prediction=True,  # Approve loan
    confidence=0.78,
    human_reviewer="Senior Loan Officer"
)

# Simulate feedback collection
feedback_result, improvements = collaboration.feedback_loop(
    decision_id="LOAN_2024_001",
    outcome="successful_repayment",
    user_feedback={
        "satisfaction": 4,
        "trust": 3,
        "explanation_clarity": 4
    }
)

print(f"\nImprovement Suggestions: {improvements}")
```

## 5. Responsible AI Monitoring and Governance

### Continuous Monitoring Framework

```python
class ResponsibleAIMonitoring:
    """Continuous monitoring for responsible AI deployment"""
    
    def __init__(self, model_name, monitoring_config):
        self.model_name = model_name
        self.config = monitoring_config
        self.monitoring_history = []
        self.alerts = []
    
    def performance_monitoring(self, predictions, actuals, sensitive_attributes):
        """Monitor model performance including fairness metrics"""
        
        monitoring_result = {
            'timestamp': datetime.now(),
            'model_name': self.model_name,
            'metrics': {}
        }
        
        # Performance metrics
        accuracy = np.mean(predictions == actuals)
        monitoring_result['metrics']['accuracy'] = accuracy
        
        # Fairness metrics by group
        fairness_metrics = {}
        for group in np.unique(sensitive_attributes):
            group_mask = sensitive_attributes == group
            group_accuracy = np.mean(predictions[group_mask] == actuals[group_mask])
            fairness_metrics[f'accuracy_{group}'] = group_accuracy
        
        monitoring_result['metrics']['fairness'] = fairness_metrics
        
        # Check thresholds
        alerts = self._check_performance_thresholds(monitoring_result['metrics'])
        monitoring_result['alerts'] = alerts
        
        self.monitoring_history.append(monitoring_result)
        self.alerts.extend(alerts)
        
        print(f"=== PERFORMANCE MONITORING: {self.model_name} ===")
        print(f"Overall Accuracy: {accuracy:.3f}")
        print("Group-wise Performance:")
        for metric, value in fairness_metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        if alerts:
            print("⚠️ Alerts:")
            for alert in alerts:
                print(f"  - {alert}")
        
        return monitoring_result
    
    def _check_performance_thresholds(self, metrics):
        """Check if metrics exceed defined thresholds"""
        alerts = []
        
        # Check overall performance
        if metrics['accuracy'] < self.config.get('min_accuracy', 0.8):
            alerts.append(f"Accuracy below threshold: {metrics['accuracy']:.3f}")
        
        # Check fairness
        fairness_metrics = metrics.get('fairness', {})
        accuracies = [v for k, v in fairness_metrics.items() if k.startswith('accuracy_')]
        
        if len(accuracies) > 1:
            max_diff = max(accuracies) - min(accuracies)
            if max_diff > self.config.get('max_fairness_gap', 0.1):
                alerts.append(f"Fairness gap detected: {max_diff:.3f}")
        
        return alerts
    
    def data_drift_detection(self, current_data, reference_data):
        """Detect data drift in model inputs"""
        
        drift_results = {}
        
        for column in current_data.columns:
            if column in reference_data.columns:
                # Simple drift detection using statistical tests
                from scipy.stats import ks_2samp
                
                current_values = current_data[column].dropna()
                reference_values = reference_data[column].dropna()
                
                if len(current_values) > 0 and len(reference_values) > 0:
                    # Kolmogorov-Smirnov test
                    statistic, p_value = ks_2samp(reference_values, current_values)
                    
                    drift_detected = p_value < 0.05  # 5% significance level
                    drift_results[column] = {
                        'drift_detected': drift_detected,
                        'p_value': p_value,
                        'statistic': statistic
                    }
        
        print(f"=== DATA DRIFT DETECTION ===")
        drift_columns = [col for col, result in drift_results.items() if result['drift_detected']]
        
        if drift_columns:
            print(f"⚠️ Drift detected in: {', '.join(drift_columns)}")
        else:
            print("✅ No significant drift detected")
        
        return drift_results
    
    def compliance_monitoring(self, regulations=['GDPR', 'CCPA', 'FCRA']):
        """Monitor compliance with regulations"""
        
        compliance_status = {}
        
        for regulation in regulations:
            compliance_checks = self._get_compliance_checks(regulation)
            regulation_status = {}
            
            for check_name, check_func in compliance_checks.items():
                try:
                    result = check_func()
                    regulation_status[check_name] = {
                        'compliant': result,
                        'status': 'PASS' if result else 'FAIL'
                    }
                except Exception as e:
                    regulation_status[check_name] = {
                        'compliant': False,
                        'status': 'ERROR',
                        'error': str(e)
                    }
            
            compliance_status[regulation] = regulation_status
        
        print(f"=== COMPLIANCE MONITORING ===")
        for regulation, status in compliance_status.items():
            print(f"\n{regulation}:")
            for check, result in status.items():
                status_symbol = "✅" if result.get('compliant') else "❌"
                print(f"  {status_symbol} {check}: {result['status']}")
        
        return compliance_status
    
    def _get_compliance_checks(self, regulation):
        """Get compliance checks for specific regulation"""
        
        if regulation == 'GDPR':
            return {
                'data_minimization': lambda: True,  # Simplified check
                'consent_management': lambda: True,
                'right_to_explanation': lambda: True,
                'data_protection_impact_assessment': lambda: True
            }
        elif regulation == 'CCPA':
            return {
                'privacy_notice': lambda: True,
                'opt_out_mechanism': lambda: True,
                'data_deletion_capability': lambda: True
            }
        elif regulation == 'FCRA':
            return {
                'adverse_action_notice': lambda: True,
                'accuracy_requirements': lambda: True,
                'dispute_resolution': lambda: True
            }
        else:
            return {}
    
    def generate_responsible_ai_report(self):
        """Generate comprehensive responsible AI report"""
        
        if not self.monitoring_history:
            return "No monitoring data available"
        
        latest_metrics = self.monitoring_history[-1]['metrics']
        
        report = {
            'executive_summary': {
                'model_name': self.model_name,
                'monitoring_period': f"{self.monitoring_history[0]['timestamp']} to {self.monitoring_history[-1]['timestamp']}",
                'total_alerts': len(self.alerts),
                'current_performance': latest_metrics.get('accuracy', 'N/A'),
                'compliance_status': 'Under Review'
            },
            'performance_trends': self._analyze_performance_trends(),
            'fairness_assessment': self._analyze_fairness_trends(),
            'risk_assessment': self._assess_current_risks(),
            'recommendations': self._generate_recommendations()
        }
        
        print("=== RESPONSIBLE AI REPORT ===")
        print(f"Model: {report['executive_summary']['model_name']}")
        print(f"Current Performance: {report['executive_summary']['current_performance']}")
        print(f"Total Alerts: {report['executive_summary']['total_alerts']}")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        return report
    
    def _analyze_performance_trends(self):
        """Analyze performance trends over time"""
        if len(self.monitoring_history) < 2:
            return "Insufficient data for trend analysis"
        
        accuracies = [m['metrics'].get('accuracy', 0) for m in self.monitoring_history]
        
        if accuracies[-1] < accuracies[0]:
            return "Declining performance trend detected"
        else:
            return "Stable or improving performance"
    
    def _analyze_fairness_trends(self):
        """Analyze fairness trends over time"""
        return "Fairness metrics within acceptable range"
    
    def _assess_current_risks(self):
        """Assess current risks"""
        risk_level = "Medium" if len(self.alerts) > 5 else "Low"
        return f"Current risk level: {risk_level}"
    
    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        recommendations = [
            "Continue regular monitoring and evaluation",
            "Implement bias testing in production pipeline",
            "Enhance user feedback collection mechanisms",
            "Schedule quarterly ethics review meetings"
        ]
        
        if len(self.alerts) > 0:
            recommendations.insert(0, "Address current alerts and performance issues")
        
        return recommendations

# Example monitoring implementation
monitoring_config = {
    'min_accuracy': 0.85,
    'max_fairness_gap': 0.08,
    'drift_threshold': 0.05
}

monitor = ResponsibleAIMonitoring(
    model_name="Credit Risk Model v2.1",
    monitoring_config=monitoring_config
)

# Simulate monitoring data
np.random.seed(42)
predictions = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
actuals = np.random.choice([0, 1], size=1000, p=[0.65, 0.35])
sensitive_attrs = np.random.choice(['Group_A', 'Group_B'], size=1000, p=[0.6, 0.4])

# Run performance monitoring
perf_result = monitor.performance_monitoring(predictions, actuals, sensitive_attrs)

# Simulate compliance monitoring
compliance_result = monitor.compliance_monitoring()

# Generate comprehensive report
ai_report = monitor.generate_responsible_ai_report()
```

## Summary

Responsible AI implementation requires:

### Core Components
1. **Ethical Impact Assessment**: Systematic evaluation of potential harms and risks
2. **Governance Framework**: Clear accountability, oversight, and decision-making processes
3. **Transparency Measures**: Model cards, explanations, and public reporting
4. **Human-AI Collaboration**: Appropriate human oversight and control mechanisms
5. **Continuous Monitoring**: Performance, fairness, and compliance tracking

### Key Practices
- **Early Integration**: Consider responsible AI from project inception
- **Stakeholder Engagement**: Include diverse perspectives in development and deployment
- **Continuous Improvement**: Regular assessment and refinement of AI systems
- **Documentation**: Comprehensive recording of decisions, trade-offs, and impacts
- **Compliance**: Adherence to relevant regulations and industry standards

### Implementation Strategy
- Establish ethics committees and governance structures
- Implement bias testing and fairness monitoring
- Design human-centered AI interfaces and workflows
- Create transparency and accountability mechanisms
- Develop incident response and corrective action procedures

Responsible AI is not a one-time implementation but an ongoing commitment to ethical, transparent, and accountable AI development and deployment.