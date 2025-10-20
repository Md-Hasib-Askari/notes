# Growth Loops and Viral Mechanics

## 1. Core Concepts

- Loops vs funnels: Loops create compounding effects as outputs feed inputs (e.g., content → shares → more content creation).
- Types of loops: Content loops, viral loops (invites, sharing), paid loops (reinvest margin), sales loops (customer referrals), product-led loops (templates, marketplaces).
- Viral coefficient (K): Average invites × invite acceptance rate; K > 1 leads to exponential growth, but even < 1 can be valuable.
- Quality of growth: Idle signups don’t help; measure Activation and Retention alongside invites.

## 2. Frameworks & Examples

- Loop design:
  - Trigger → Action → Reward → Investment (hook model applied to loops).
  - Example (collaboration app): Create document → invite teammate → real-time edit → store templates → more documents created.
- Incentives: Credits, premium features, or recognition; avoid abuse.
- Instrumentation: Track invites sent, acceptance, multi-player sessions, content created per user.

## 3. Actionable Techniques & Tools

- Start with one loop: Make it measurable and tie it to retained usage.
- Reduce friction: One-click invites, clear value for invitee, seeded content/templates.
- Anti-abuse: Rate limits, invite caps, and monitoring.
- Tools: Amplitude/Mixpanel (analytics), PostHog (product analytics), in-app frameworks for invites and rewards.

## 4. Common Mistakes & Fixes

- Mistake: Counting raw invites. Fix: Track accepted invites and retained multi-player use.
- Mistake: Incentives that attract the wrong users. Fix: Reward actions tied to value (e.g., completed projects).
- Mistake: Ignoring governance. Fix: Add controls and audit logs for enterprise.

## 5. Hands-on Exercises

- Guided: Define one growth loop and the 3 core events you’ll measure.
- Guided: Design an invite flow with clear value for the invitee and a single CTA (Call To Action).
- Challenge: Run a 4-week loop experiment and report changes in Activation, Retention, and invites accepted.
