# GDPR & Responsible AI Compliance Notes

---

## Data Protection Principles

### 1. Lawful Basis for Processing
**Legal Basis:** Legitimate Interest (Article 6(1)(f) GDPR)
- Business has legitimate interest in customer retention
- Processing necessary for marketing purposes
- Customer interests not overridden

**Alternative Basis:** Consent
- Customers opted in to marketing communications
- Can withdraw consent at any time

---

### 2. Data Minimization
**Data Collected:**
- Customer ID (pseudonymized)
- Purchase history (transactional data)
- Behavioral metrics (RFM, velocity)

**NOT Collected:**
- Names, addresses (not required for modeling)
- Demographics (age, gender not available)
- Payment details (not used in analysis)

---

### 3. Right to Explanation (Article 22)
**SHAP Explanations Provided:**
- Individual customer churn predictions explained
- Top 5 factors driving each prediction shown
- Explanations available on request

**Transparency:**
- Model logic documented
- Feature importance rankings published
- No "black box" decisions

---

### 4. Automated Decision-Making
**Human Oversight Required:**
- Model outputs are **recommendations**, not final decisions
- Marketing team reviews before action
- Customers can request human review

**No Solely Automated Decisions:**
- Campaign targeting reviewed by team
- Discount offers approved manually
- No automatic account suspensions

---

### 5. Data Retention
**Storage Duration:**
- Transaction data: 2 years (business records)
- Model predictions: 1 year (refresh annually)
- Aggregated insights: Indefinite (anonymized)

**Right to Erasure:**
- Customers can request data deletion
- Removal from marketing lists honored
- Model retraining excludes deleted customers

---

### 6. Security Measures
**Data Protection:**
- Customer IDs pseudonymized
- Model files encrypted at rest
- Access controls on prediction outputs

**No Public Exposure:**
- Individual predictions not published
- Aggregated segments only in reports
- Customer lists secured (CRM access only)

---

## Responsible AI Considerations

### Fairness
**No Protected Characteristics Used:**
- No age, gender, ethnicity data
- No proxy variables for protected groups
- Purchase behavior only

**Bias Monitoring:**
- SHAP analysis checks for unexpected correlations
- Regular audits of segment composition
- Fairness metrics tracked

---

### Accountability
**Model Governance:**
- Model owner: Data Science Team
- Business owner: Marketing Director
- Escalation path defined for edge cases

**Audit Trail:**
- Model version tracked
- Training data snapshots saved
- Prediction logs maintained

---

### Transparency
**Customer Communication:**
- Privacy policy includes retention modeling
- Opt-out available for marketing
- Explanation provided on request

**Internal Documentation:**
- README provides complete methodology
- Feature definitions published
- Limitations acknowledged

---

## Compliance Checklist

- [x] Lawful basis for processing identified
- [x] Data minimization applied
- [x] Explanations available (SHAP)
- [x] Human oversight in decision-making
- [x] Data retention policy defined
- [x] Right to erasure procedure in place
- [x] Security measures implemented
- [x] Fairness audit conducted
- [x] Accountability structure defined
- [x] Transparency documentation complete

---

**Status:** Compliant  
**Review Date:** Quarterly  
**Contact:** DPO (Data Protection Officer)
