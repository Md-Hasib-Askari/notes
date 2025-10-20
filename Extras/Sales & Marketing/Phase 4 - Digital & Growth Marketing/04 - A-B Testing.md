# A/B Testing

## 1. Core Concepts

- Hypothesis-driven: Predefine expected impact and how you’ll measure success.
- Minimum detectable effect: Ensure your sample size and duration can realistically observe the change.
- Guardrails: Monitor negative side effects (e.g., retention, error rates) during tests.
- Ethics: No deceptive patterns; be transparent where it affects user trust.

## 2. Frameworks & Examples

- Test plan:
  - Hypothesis, metric, audience, timeline.
  - Variants: Control (A) vs Variant (B); keep to one meaningful change.
  - Analysis: Statistical significance vs. business significance.
- Example tests:
  - Landing page headline; onboarding checklist step order; pricing page labels.

## 3. Actionable Techniques & Tools

- Calculators: Use online sample size calculators; aim for at least one business cycle.
- Segmentation: Analyze new vs returning users and top channels separately.
- Platform: Optimizely, Visual Website Optimizer (VWO), Google Optimize (legacy) or built-in in-app frameworks.
- Documentation: Log hypotheses, results, and decisions in a shared doc.

## 4. Common Mistakes & Fixes

- Mistake: Stopping early. Fix: Pre-commit to test length; use sequential testing if needed.
- Mistake: Multiple changes per variant. Fix: Isolate variables.
- Mistake: Chasing tiny wins. Fix: Prioritize by impact and confidence (ICE score).

## 5. Hands-on Exercises

- Guided: Draft a test plan for a high-impact page with expected effect size and timeline.
- Guided: Run a 2-week test and report primary metric and guardrail metrics.
- Challenge: Build a simple internal testing checklist and require it for all experiments.
