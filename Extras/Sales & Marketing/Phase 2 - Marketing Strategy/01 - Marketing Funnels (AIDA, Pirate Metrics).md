# Marketing Funnels (AIDA, Pirate Metrics)

## 1. Core Concepts

- Funnel: A staged model describing how people move from unaware → paying → loyal advocates. Each stage has a clear success metric and owner.
- AIDA — Attention, Interest, Desire, Action: Classic communication funnel useful for campaign messaging.
- Pirate Metrics (AARRR) — Acquisition, Activation, Retention, Revenue, Referral: Product-centric lifecycle view for software and online products.
- North Star Metric (NSM): The single metric that best captures the value delivered to customers; aligns teams across the funnel.
- Conversion Rate (CR) and Drop-off: The percentage moving to the next stage vs. exiting; absolute numbers matter as much as rates.

## 2. Frameworks & Examples

- Stage definitions with example metrics:
  - Acquisition: New visitors/leads from channels (Source tracking via Urchin Tracking Module (UTM) parameters).
  - Activation: First key outcome achieved (e.g., time-to-first-value, onboarding completion).
  - Retention: Returning usage over time (e.g., Day-7/Week-4 retention).
  - Revenue: Trial → paid conversion, Average Revenue Per Account (ARPA), Annual Contract Value (ACV).
  - Referral: Share rate, invite acceptance, viral coefficient.
- AIDA in practice:
  - Attention: Thumb-stopping hook in ads; clear audience targeting.
  - Interest: Useful, credible content that addresses Jobs-to-be-Done (JTBD).
  - Desire: Proof (logos, reviews, quantified outcomes).
  - Action: Clear Call To Action (CTA) and frictionless path (no unnecessary fields).
- Example mapping (Business-to-Business (B2B) SaaS):
  - Landing page → signup → onboarding checklist → in-app aha → upgrade → invite teammates.

## 3. Actionable Techniques & Tools

- Instrumentation: Define events and properties; implement with Segment or RudderStack; visualize in Mixpanel/Amplitude.
- Funnel analysis: Create cohorted funnel views for new vs returning users; compare by channel and persona.
- Quick wins: Shorten forms, improve page speed, add social proof near CTAs, test a single CTA per page.
- Experiment cadence: Weekly small bets; track Confidence, Impact, Ease (ICE) scores; document learnings.
- Tooling: Google Analytics 4 (GA4), Looker Studio dashboards, Hotjar session replays, Optimizely/Visual Website Optimizer (VWO) for experiments.

## 4. Common Mistakes & Fixes

- Mistake: Optimizing a stage without a baseline. Fix: Establish current CR and absolute counts first.
- Mistake: Channel-only focus. Fix: Balance acquisition with activation/retention initiatives.
- Mistake: Wrong North Star Metric. Fix: Choose an NSM tightly correlated with retention and revenue.
- Mistake: Leaky handoffs. Fix: Document Service Level Agreements (SLAs) between Marketing → Sales and Product → Sales on Product Qualified Leads (PQLs).

## 5. Hands-on Exercises

- Guided: Define your AARRR stages with one metric and one owner each; list top 3 drop-offs.
- Guided: Build a funnel in Amplitude/Mixpanel for last 30 days; segment by top 3 channels (via UTM source/medium).
- Challenge: Run a 2-week A/B test on your onboarding CTA and report changes in Activation and Day-7 retention.
