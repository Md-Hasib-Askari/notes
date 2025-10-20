# Analytics (Google Analytics, UTM, Funnel Tracking)

## 1. Core Concepts

- Source of truth: Define which tool is authoritative for which metric (CRM vs. product analytics vs. ad platforms).
- Attribution: Use Urchin Tracking Module (UTM) parameters consistently; agree on primary vs. assist rules.
- Funnel tracking: Standardize events, properties, and identities across tools.
- Privacy and governance: Consent, data retention, and access controls.

## 2. Frameworks & Examples

- Tracking plan:
  - Events: signup_started, signup_completed, project_created, invite_sent, plan_upgraded.
  - Properties: plan, role, segment, source/medium/campaign (UTMs).
  - Identity: user_id, account_id; anonymous_id for pre-login.
- Dashboards:
  - Acquisition: Sessions, Cost Per Lead (CPL), MQLs (Marketing Qualified Leads), channel ROI (Return on Investment).
  - Activation: Time-to-First-Value (TTFV), onboarding completion.
  - Retention: Day-7/Week-4 retention, churn.
  - Revenue: Trial-to-paid, ARPA (Average Revenue Per Account), ACV (Annual Contract Value), LTV (Lifetime Value), CAC (Customer Acquisition Cost).

## 3. Actionable Techniques & Tools

- Implement: Google Tag Manager + GA4 (Google Analytics 4) + Amplitude/Mixpanel; server-side tagging for reliability.
- Identity resolution: Stitch web, app, and backend events with stable IDs.
- QA: Use debug modes and staging; sample events in a data warehouse if possible.
- Tools: GA4, Looker Studio (reporting), Amplitude/Mixpanel (product analytics), BigQuery/Snowflake (warehouse).

## 4. Common Mistakes & Fixes

- Mistake: Tracking everything. Fix: Track what you’ll use; delete noise.
- Mistake: Inconsistent UTMs. Fix: Create a shared UTM builder and naming standard.
- Mistake: No identity strategy. Fix: Decide how/when to set user/account IDs.

## 5. Hands-on Exercises

- Guided: Build a basic tracking plan (events/properties/IDs) and share with stakeholders.
- Guided: Create one acquisition and one activation dashboard.
- Challenge: Implement identity resolution and show a cohort chart from signup to upgrade by channel.
