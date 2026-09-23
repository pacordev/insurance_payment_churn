-- ============================================================
-- DDL Script: Create tables from db_tables.xlsx
-- Database:   PostgreSQL
-- Generated:  2026-05-18
-- ============================================================

CREATE TABLE IF NOT EXISTS tbl_customer (
    customer_id     TEXT            NOT NULL,
    tax_id          TEXT,
    c_first_name    TEXT,
    c_last_name     TEXT,
    CONSTRAINT pk_tbl_customer PRIMARY KEY (customer_id)
);

CREATE TABLE IF NOT EXISTS tbl_policies (
    insurance_product       TEXT,
    customer_id             TEXT            NOT NULL,
    policy_num              TEXT            NOT NULL,
    effective_start_date    DATE,
    effective_end_date      DATE,
    payment_frequency       TEXT,
    policy_status           TEXT,
    cancel_date             DATE,
    CONSTRAINT pk_tbl_policies PRIMARY KEY (policy_num),
    CONSTRAINT fk_policies_customer FOREIGN KEY (customer_id)
        REFERENCES tbl_customer (customer_id)
);

CREATE TABLE IF NOT EXISTS tbl_invoice (
    policy_num              TEXT            NOT NULL,
    invoice_num             TEXT            NOT NULL,
    invoice_type            TEXT,
    invoice_start_date      DATE,
    invoice_end_date        DATE,
    days_arrears_allowed    INTEGER,
    installment_num         INTEGER,
    premium                 NUMERIC(12, 2),
    CONSTRAINT pk_tbl_invoice PRIMARY KEY (invoice_num),
    CONSTRAINT fk_invoice_policy FOREIGN KEY (policy_num)
        REFERENCES tbl_policies (policy_num)
);

CREATE TABLE IF NOT EXISTS tbl_payhistory (
    transaction_id          TEXT            NOT NULL,
    invoice_num             TEXT            NOT NULL,
    premium                 NUMERIC(12, 2),
    pay_transaction_date    DATE,
    pay_transaction_status  TEXT,
    CONSTRAINT pk_tbl_payhistory PRIMARY KEY (transaction_id),
    CONSTRAINT fk_payhistory_invoice FOREIGN KEY (invoice_num)
        REFERENCES tbl_invoice (invoice_num)
);

-- Batch-scoring output (churn_predict.py). One row per scoring run per
-- policy, not just current-state, so the dashboard can show a risk trend
-- over time instead of only the latest number.
CREATE TABLE IF NOT EXISTS tbl_policy_risk_score (
    policy_num          TEXT            NOT NULL,
    scored_at           TIMESTAMP       NOT NULL DEFAULT now(),
    churn_probability   NUMERIC(5, 4),
    segment_label       TEXT,           -- unsupervised cluster from segmentation.py
    top_factors         JSONB,          -- SHAP top contributors, e.g. [{"feature": "failed_invoice_ratio", "value": 0.41}, ...]
    recommended_action  TEXT,           -- from recommend.py's driver -> action mapping
    CONSTRAINT pk_policy_risk_score PRIMARY KEY (policy_num, scored_at),
    CONSTRAINT fk_risk_score_policy FOREIGN KEY (policy_num)
        REFERENCES tbl_policies (policy_num)
);
