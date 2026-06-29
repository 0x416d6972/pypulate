# KPI Guide

The `KPI` class provides a comprehensive suite of methods for calculating and tracking business metrics. It maintains state to provide health assessments and trend analysis.

## Basic Usage

```python
from pypulate import KPI

# Initialize KPI tracker
kpi = KPI()

# Calculate basic metrics
churn = kpi.churn_rate(customers_start=1000, customers_end=950, new_customers=50)
retention = kpi.retention_rate(customers_start=1000, customers_end=950, new_customers=50)

# Get health assessment
health = kpi.health
```

## Customer Metrics

### Churn and Retention
```python
# Calculate churn rate
churn = kpi.churn_rate(
    customers_start=1000,  # Starting customer count
    customers_end=950,     # Ending customer count
    new_customers=50       # New customers acquired
)

# Calculate retention rate
retention = kpi.retention_rate(
    customers_start=1000,
    customers_end=950,
    new_customers=50
)
```

### Customer Lifetime Value
```python
clv = kpi.customer_lifetime_value(
    avg_revenue_per_customer=100,  # Monthly revenue per customer
    gross_margin=70,              # Gross margin percentage
    churn_rate_value=5,          # Monthly churn rate
    discount_rate=10             # Annual discount rate
)
```

## Financial Metrics

### Revenue Metrics
```python
# Calculate MRR
mrr = kpi.monthly_recurring_revenue(
    paying_customers=1000,
    avg_revenue_per_customer=50
)

# Calculate total MRR by summing individual subscription revenues
total_mrr = kpi.total_monthly_recurring_revenue(
    subscription_revenues=[50, 75, 120, 30]
)

# Calculate ARR
arr = kpi.annual_recurring_revenue(
    paying_customers=1000,
    avg_revenue_per_customer=50
)
```

### Cost Metrics
```python
# Calculate CAC
cac = kpi.customer_acquisition_cost(
    marketing_costs=50000,
    sales_costs=30000,
    new_customers=100
)

# Calculate ROI
roi = kpi.roi(
    revenue=150000,
    costs=100000
)
```

### Profitability Metrics
```python
# Operating margin (EBIT / revenue)
op_margin = kpi.operating_margin(
    operating_income=2000,
    revenue=10000
)

# Net profit margin (net income / revenue)
net_margin = kpi.net_profit_margin(
    net_income=1500,
    revenue=10000
)

# EBITDA margin
ebitda = kpi.ebitda_margin(
    ebitda=3000,
    revenue=10000
)

# Days Sales Outstanding (collection efficiency)
dso = kpi.days_sales_outstanding(
    accounts_receivable=50000,
    total_credit_sales=500000,
    days=365
)
```

## Growth & Retention Metrics

### Revenue Retention
```python
# Net Revenue Retention (includes expansion, net of churn/contraction)
nrr = kpi.net_revenue_retention(
    starting_revenue=10000,
    expansion_revenue=2000,
    contraction_revenue=500,
    churned_revenue=1000
)

# Gross Revenue Retention (excludes expansion)
grr = kpi.gross_revenue_retention(
    starting_revenue=10000,
    contraction_revenue=500,
    churned_revenue=1000
)
```

### Growth Efficiency
```python
# Revenue growth rate (period over period)
growth = kpi.revenue_growth_rate(
    current_revenue=12000,
    previous_revenue=10000
)

# SaaS Quick Ratio (revenue gained vs. revenue lost)
quick_ratio = kpi.saas_quick_ratio(
    new_mrr=500,
    expansion_mrr=200,
    churned_mrr=100,
    contraction_mrr=75
)

# Rule of 40 (growth rate + profit margin should be >= 40)
rule40 = kpi.rule_of_40(
    revenue_growth_rate=25,
    profit_margin=20
)

# Magic Number (sales & marketing efficiency)
magic = kpi.magic_number(
    current_quarter_revenue=1300,
    previous_quarter_revenue=1000,
    sales_marketing_spend=600
)
```

### Account & Risk Metrics
```python
# Average Revenue Per Account (ARPA)
arpa = kpi.average_revenue_per_account(
    total_revenue=50000,
    total_accounts=250
)

# Customer concentration risk (share of revenue from top customers)
concentration = kpi.customer_concentration(
    top_customer_revenue=4000,
    total_revenue=10000
)
```

## Conversion & Web Metrics
```python
# Bounce rate (single-page sessions / total sessions)
bounce = kpi.bounce_rate(
    single_page_sessions=400,
    total_sessions=1000
)

# Cart abandonment rate
abandonment = kpi.cart_abandonment_rate(
    completed_purchases=300,
    initiated_checkouts=1000
)
```

## Engagement Metrics

### Net Promoter Score
```python
nps = kpi.net_promoter_score(
    promoters=70,        # Customers rating 9-10
    detractors=10,       # Customers rating 0-6
    total_respondents=100
)
```

### Customer Satisfaction
```python
# Calculate CSAT
csat = kpi.customer_satisfaction_score(
    satisfaction_ratings=[4, 5, 3, 5, 4],
    max_rating=5
)

# Calculate Customer Effort Score
ces = kpi.customer_effort_score(
    effort_ratings=[2, 3, 1, 2, 4],
    max_rating=7
)
```

## Health Assessment

The `health` property provides a comprehensive assessment of business health based on all tracked metrics:

```python
health = kpi.health

# Health assessment structure
{
    'overall_score': 85.5,
    'status': 'Good',
    'components': {
        'churn_rate': {
            'score': 90.0,
            'status': 'Excellent'
        },
        'retention_rate': {
            'score': 85.0,
            'status': 'Good'
        },
        # ... other metrics
    }
}
```

### Health Score Components

The health score is calculated based on weighted components:

- **Customer Health (30%)**
  - Churn Rate
  - Retention Rate
  - LTV/CAC Ratio

- **Financial Health (30%)**
  - Gross Margin
  - Net Profit Margin
  - ROI
  - Revenue Growth

- **Revenue Retention**
  - Net Revenue Retention (NRR)

- **Engagement Health (40%)**
  - NPS
  - CSAT
  - Feature Adoption

Each component is scored from 0-100 and assigned a status:
- Excellent: ≥ 90
- Good: ≥ 75
- Fair: ≥ 60
- Poor: ≥ 45
- Critical: < 45

## State Management

The KPI class maintains state for all calculated metrics in the `_state` dictionary. This allows for:
- Trend analysis
- Health assessment
- Historical comparison
- Metric correlation

```python
# Access stored metrics
stored_churn = kpi._state['churn_rate']
stored_retention = kpi._state['retention_rate']
```

## Best Practices

### 1. Data Collection and Management
- 1.1. **Initialize Early**: Create the KPI instance at the start of your analysis
- 1.2. **Regular Updates**: Update metrics consistently for accurate trending
- 1.3. **Store History**: Consider saving state for long-term analysis

### 2. Analysis and Monitoring
- 2.1. **Monitor Health**: Regularly check the health assessment
- 2.2. **Validate Inputs**: Ensure input data quality for accurate metrics
- 2.3. **Compare Trends**: Analyze metric changes over time rather than isolated values

### 3. Reporting and Decision Making
- 3.1. **Focus on Key Metrics**: Prioritize metrics most relevant to your business model
- 3.2. **Set Thresholds**: Establish alert thresholds for critical metrics
- 3.3. **Contextualize Results**: Consider market conditions when interpreting metrics
``` 