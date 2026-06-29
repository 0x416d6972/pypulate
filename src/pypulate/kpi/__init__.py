"""
Key Performance Indicators (KPI) module.

This module provides functions for calculating various business and financial KPIs.
"""

from .business_kpi import (
    churn_rate, retention_rate,
    customer_lifetime_value, customer_acquisition_cost, ltv_cac_ratio, payback_period,
    monthly_recurring_revenue, total_monthly_recurring_revenue, annual_recurring_revenue,
    revenue_churn_rate, expansion_revenue_rate,
    customer_satisfaction_score, customer_effort_score, net_promoter_score,
    daily_active_users_ratio, monthly_active_users_ratio, stickiness_ratio,
    burn_rate, runway, gross_margin,
    conversion_rate, virality_coefficient, feature_adoption_rate, roi,
    average_revenue_per_user, average_revenue_per_paying_user, customer_engagement_score, time_to_value,
    net_revenue_retention, gross_revenue_retention, revenue_growth_rate,
    saas_quick_ratio, rule_of_40, magic_number,
    operating_margin, net_profit_margin, ebitda_margin,
    average_revenue_per_account, customer_concentration,
    bounce_rate, cart_abandonment_rate, days_sales_outstanding
)


__all__ = [
    'churn_rate', 'retention_rate',
    'customer_lifetime_value', 'customer_acquisition_cost', 'ltv_cac_ratio', 'payback_period',
    'monthly_recurring_revenue', 'total_monthly_recurring_revenue', 'annual_recurring_revenue',
    'revenue_churn_rate', 'expansion_revenue_rate',
    'customer_satisfaction_score', 'customer_effort_score', 'net_promoter_score',
    'daily_active_users_ratio', 'monthly_active_users_ratio', 'stickiness_ratio',
    'burn_rate', 'runway', 'gross_margin',
    'conversion_rate', 'virality_coefficient', 'feature_adoption_rate', 'roi',
    'average_revenue_per_user', 'average_revenue_per_paying_user', 'customer_engagement_score', 'time_to_value',
    'net_revenue_retention', 'gross_revenue_retention', 'revenue_growth_rate',
    'saas_quick_ratio', 'rule_of_40', 'magic_number',
    'operating_margin', 'net_profit_margin', 'ebitda_margin',
    'average_revenue_per_account', 'customer_concentration',
    'bounce_rate', 'cart_abandonment_rate', 'days_sales_outstanding'
]
