"""
Business KPIs Module

This module provides functions for calculating various business metrics
commonly used in SaaS and subscription-based businesses.
"""

import numpy as np
from typing import Union, Optional, cast
from numpy.typing import ArrayLike, NDArray


def _ensure_array(value: ArrayLike) -> NDArray[np.float64]:
    """
    Ensure the input is either a scalar or numpy array.
    
    Parameters
    ----------
    value : array-like
        Input value to convert
        
    Returns
    -------
    numpy.ndarray
        Converted value as numpy array
    """
    if np.isscalar(value):
        return np.array([float(value)], dtype=np.float64)
    return np.array(value, dtype=np.float64)

def churn_rate(
    customers_start: ArrayLike, 
    customers_end: ArrayLike, 
    new_customers: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate customer churn rate.
    
    Churn rate is the percentage of customers who stop using your product or service
    during a given time period.
    
    Parameters
    ----------
    customers_start : array-like
        Number of customers at the start of the period
    customers_end : array-like
        Number of customers at the end of the period
    new_customers : array-like
        Number of new customers acquired during the period
        
    Returns
    -------
    float or numpy.ndarray
        Churn rate as a percentage
        
    Examples
    --------
    >>> churn_rate(100, 90, 10)
    20.0
    """
    customers_start_arr = _ensure_array(customers_start)
    customers_end_arr = _ensure_array(customers_end)
    new_customers_arr = _ensure_array(new_customers)
    
    lost_customers = customers_start_arr + new_customers_arr - customers_end_arr
    
    result = np.zeros_like(customers_start_arr, dtype=np.float64)
    
    non_zero_mask = customers_start_arr != 0
    result[non_zero_mask] = (lost_customers[non_zero_mask] / customers_start_arr[non_zero_mask]) * 100.0
    
    if (np.isscalar(customers_start) and 
        np.isscalar(customers_end) and 
        np.isscalar(new_customers)):
        return float(result[0])
    
    return result

def retention_rate(
    customers_start: ArrayLike, 
    customers_end: ArrayLike, 
    new_customers: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate customer retention rate.
    
    Retention rate is the percentage of customers who remain with your product or service
    over a given time period.
    
    Parameters
    ----------
    customers_start : array-like
        Number of customers at the start of the period
    customers_end : array-like
        Number of customers at the end of the period
    new_customers : array-like
        Number of new customers acquired during the period
        
    Returns
    -------
    float or numpy.ndarray
        Retention rate as a percentage
        
    Examples
    --------
    >>> retention_rate(100, 90, 10)
    80.0
    """
    customers_start_arr = _ensure_array(customers_start)
    customers_end_arr = _ensure_array(customers_end)
    new_customers_arr = _ensure_array(new_customers)
    
    churn = churn_rate(customers_start_arr, customers_end_arr, new_customers_arr)
    
    if np.isscalar(churn):
        return 100.0 - churn
    
    return 100.0 - churn

def customer_lifetime_value(
    avg_revenue_per_customer: ArrayLike,
    gross_margin: ArrayLike,
    churn_rate_value: ArrayLike,
    discount_rate: ArrayLike = 10.0
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Customer Lifetime Value (CLV).
    
    CLV is the total worth to a business of a customer over the whole period of their relationship.
    
    Parameters
    ----------
    avg_revenue_per_customer : array-like
        Average revenue per customer per period (e.g., monthly)
    gross_margin : array-like
        Gross margin percentage (0-100)
    churn_rate_value : array-like
        Churn rate percentage (0-100)
    discount_rate : array-like, default 10.0
        Annual discount rate for future cash flows (0-100)
        
    Returns
    -------
    float or numpy.ndarray
        Customer Lifetime Value
        
    Examples
    --------
    >>> customer_lifetime_value(100, 70, 5, 10)
    466.67
    """
    avg_revenue = _ensure_array(avg_revenue_per_customer)
    gross_margin_arr = _ensure_array(gross_margin)
    churn_rate_arr = _ensure_array(churn_rate_value)
    discount_rate_arr = _ensure_array(discount_rate)
    
    gross_margin_decimal = gross_margin_arr / 100.0
    churn_rate_decimal = churn_rate_arr / 100.0
    discount_rate_decimal = discount_rate_arr / 100.0
    
    result = np.zeros_like(avg_revenue, dtype=np.float64)
    
    non_zero_churn = churn_rate_decimal > 0
    result[non_zero_churn] = (avg_revenue[non_zero_churn] * gross_margin_decimal[non_zero_churn]) / (
        churn_rate_decimal[non_zero_churn] + discount_rate_decimal[non_zero_churn]
    )
    
    zero_churn = ~non_zero_churn
    if np.any(zero_churn):
        max_periods = 240
        for i in range(max_periods):
            discount_factor = np.power(1 + discount_rate_decimal[zero_churn]/12, i)
            result[zero_churn] += (avg_revenue[zero_churn] * gross_margin_decimal[zero_churn]) / discount_factor
    
    if (np.isscalar(avg_revenue_per_customer) and 
        np.isscalar(gross_margin) and 
        np.isscalar(churn_rate_value) and 
        np.isscalar(discount_rate)):
        return float(result[0])
    
    return result

def customer_acquisition_cost(
    marketing_costs: ArrayLike,
    sales_costs: ArrayLike,
    new_customers: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Customer Acquisition Cost (CAC).
    
    CAC is the cost of convincing a potential customer to buy a product or service.
    
    Parameters
    ----------
    marketing_costs : array-like
        Total marketing costs for the period
    sales_costs : array-like
        Total sales costs for the period
    new_customers : array-like
        Number of new customers acquired during the period
        
    Returns
    -------
    float or numpy.ndarray
        Customer Acquisition Cost
        
    Examples
    --------
    >>> customer_acquisition_cost(5000, 3000, 100)
    80.0
    """
    marketing_costs_arr = _ensure_array(marketing_costs)
    sales_costs_arr = _ensure_array(sales_costs)
    new_customers_arr = _ensure_array(new_customers)
    
    total_costs = marketing_costs_arr + sales_costs_arr
    
    result = np.full_like(new_customers_arr, np.inf, dtype=np.float64)
    
    non_zero_mask = new_customers_arr != 0
    result[non_zero_mask] = total_costs[non_zero_mask] / new_customers_arr[non_zero_mask]
    
    if (np.isscalar(marketing_costs) and 
        np.isscalar(sales_costs) and 
        np.isscalar(new_customers)):
        return float(result[0])
    
    return result

def monthly_recurring_revenue(
    paying_customers: ArrayLike,
    avg_revenue_per_customer: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Monthly Recurring Revenue (MRR).
    
    MRR is the predictable total revenue generated by all the active subscriptions in a month.
    
    Parameters
    ----------
    paying_customers : array-like
        Number of paying customers
    avg_revenue_per_customer : array-like
        Average revenue per customer per month
        
    Returns
    -------
    float or numpy.ndarray
        Monthly Recurring Revenue
        
    Examples
    --------
    >>> monthly_recurring_revenue(100, 50)
    5000.0
    """
    paying_customers_arr = _ensure_array(paying_customers)
    avg_revenue_per_customer_arr = _ensure_array(avg_revenue_per_customer)
    
    result = paying_customers_arr * avg_revenue_per_customer_arr
    
    if (np.isscalar(paying_customers) and 
        np.isscalar(avg_revenue_per_customer)):
        return float(result[0])
    
    return result

def total_monthly_recurring_revenue(
    subscription_revenues: ArrayLike
) -> float:
    """
    Calculate the total Monthly Recurring Revenue (MRR) by summing individual subscriptions.

    Total MRR is the sum of the normalized monthly revenue of every active
    subscription. Use this when you have the monthly revenue of each subscription
    (or customer) and want a single aggregated MRR figure.

    Parameters
    ----------
    subscription_revenues : array-like
        Monthly recurring revenue of each active subscription

    Returns
    -------
    float
        Total Monthly Recurring Revenue

    Examples
    --------
    >>> total_monthly_recurring_revenue([50, 60, 70])
    180.0
    """
    subscription_revenues_arr = _ensure_array(subscription_revenues)

    return float(np.sum(subscription_revenues_arr))

def annual_recurring_revenue(
    paying_customers: ArrayLike,
    avg_revenue_per_customer: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Annual Recurring Revenue (ARR).
    
    ARR is the value of the recurring revenue of a business's term subscriptions normalized for a single calendar year.
    
    Parameters
    ----------
    paying_customers : array-like
        Number of paying customers
    avg_revenue_per_customer : array-like
        Average revenue per customer per month
        
    Returns
    -------
    float or numpy.ndarray
        Annual Recurring Revenue
        
    Examples
    --------
    >>> annual_recurring_revenue(100, 50)
    60000.0
    """
    mrr = monthly_recurring_revenue(paying_customers, avg_revenue_per_customer)
    
    if np.isscalar(mrr):
        return mrr * 12.0
    
    return mrr * 12.0

def net_promoter_score(
    promoters: ArrayLike,
    detractors: ArrayLike,
    total_respondents: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Net Promoter Score (NPS).
    
    NPS measures customer experience and predicts business growth.
    
    Parameters
    ----------
    promoters : array-like
        Number of promoters (customers who rated 9-10)
    detractors : array-like
        Number of detractors (customers who rated 0-6)
    total_respondents : array-like
        Total number of survey respondents
        
    Returns
    -------
    float or numpy.ndarray
        Net Promoter Score (ranges from -100 to 100)
        
    Examples
    --------
    >>> net_promoter_score(70, 10, 100)
    60.0
    """
    promoters_arr = _ensure_array(promoters)
    detractors_arr = _ensure_array(detractors)
    total_respondents_arr = _ensure_array(total_respondents)
    
    result = np.zeros_like(total_respondents_arr, dtype=np.float64)
    
    non_zero_mask = total_respondents_arr != 0
    
    promoters_percent = np.zeros_like(total_respondents_arr, dtype=np.float64)
    detractors_percent = np.zeros_like(total_respondents_arr, dtype=np.float64)
    
    promoters_percent[non_zero_mask] = (promoters_arr[non_zero_mask] / total_respondents_arr[non_zero_mask]) * 100.0
    detractors_percent[non_zero_mask] = (detractors_arr[non_zero_mask] / total_respondents_arr[non_zero_mask]) * 100.0
    
    result = promoters_percent - detractors_percent
    
    if (np.isscalar(promoters) and 
        np.isscalar(detractors) and 
        np.isscalar(total_respondents)):
        return float(result[0])
    
    return result

def revenue_churn_rate(
    revenue_start: ArrayLike,
    revenue_end: ArrayLike,
    new_revenue: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Revenue Churn Rate.
    
    Revenue Churn Rate is the percentage of revenue lost from existing customers in a given period.
    
    Parameters
    ----------
    revenue_start : array-like
        Revenue at the start of the period
    revenue_end : array-like
        Revenue at the end of the period
    new_revenue : array-like
        New revenue acquired during the period
        
    Returns
    -------
    float or numpy.ndarray
        Revenue Churn Rate as a percentage
        
    Examples
    --------
    >>> revenue_churn_rate(10000, 9500, 1000)
    15.0
    """
    revenue_start_arr = _ensure_array(revenue_start)
    revenue_end_arr = _ensure_array(revenue_end)
    new_revenue_arr = _ensure_array(new_revenue)
    
    lost_revenue = revenue_start_arr + new_revenue_arr - revenue_end_arr
    
    result = np.zeros_like(revenue_start_arr, dtype=np.float64)
    
    non_zero_mask = revenue_start_arr != 0
    result[non_zero_mask] = (lost_revenue[non_zero_mask] / revenue_start_arr[non_zero_mask]) * 100.0
    
    if (np.isscalar(revenue_start) and 
        np.isscalar(revenue_end) and 
        np.isscalar(new_revenue)):
        return float(result[0])
    
    return result

def expansion_revenue_rate(
    upsell_revenue: ArrayLike,
    cross_sell_revenue: ArrayLike,
    revenue_start: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Expansion Revenue Rate.
    
    Expansion Revenue Rate is the percentage of additional revenue generated from existing customers.
    
    Parameters
    ----------
    upsell_revenue : array-like
        Revenue from upselling to existing customers
    cross_sell_revenue : array-like
        Revenue from cross-selling to existing customers
    revenue_start : array-like
        Revenue at the start of the period
        
    Returns
    -------
    float or numpy.ndarray
        Expansion Revenue Rate as a percentage
        
    Examples
    --------
    >>> expansion_revenue_rate(1000, 500, 10000)
    15.0
    """
    upsell_revenue_arr = _ensure_array(upsell_revenue)
    cross_sell_revenue_arr = _ensure_array(cross_sell_revenue)
    revenue_start_arr = _ensure_array(revenue_start)
    
    expansion_revenue = upsell_revenue_arr + cross_sell_revenue_arr
    
    result = np.zeros_like(revenue_start_arr, dtype=np.float64)
    
    non_zero_mask = revenue_start_arr != 0
    result[non_zero_mask] = (expansion_revenue[non_zero_mask] / revenue_start_arr[non_zero_mask]) * 100.0
    
    if (np.isscalar(upsell_revenue) and 
        np.isscalar(cross_sell_revenue) and 
        np.isscalar(revenue_start)):
        return float(result[0])
    
    return result

def ltv_cac_ratio(
    ltv: ArrayLike,
    cac: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate LTV:CAC Ratio.
    
    LTV:CAC Ratio is a metric that compares the lifetime value of a customer to the cost of acquiring that customer.
    
    Parameters
    ----------
    ltv : array-like
        Customer Lifetime Value
    cac : array-like
        Customer Acquisition Cost
        
    Returns
    -------
    float or numpy.ndarray
        LTV:CAC Ratio
        
    Examples
    --------
    >>> ltv_cac_ratio(1000, 200)
    5.0
    """
    ltv_arr = _ensure_array(ltv)
    cac_arr = _ensure_array(cac)
    
    result = np.full_like(cac_arr, np.inf, dtype=np.float64)
    
    non_zero_mask = cac_arr != 0
    result[non_zero_mask] = ltv_arr[non_zero_mask] / cac_arr[non_zero_mask]
    
    if (np.isscalar(ltv) and np.isscalar(cac)):
        return float(result[0])
    
    return result

def payback_period(
    cac: ArrayLike,
    avg_monthly_revenue: ArrayLike,
    gross_margin: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate CAC Payback Period in months.
    
    CAC Payback Period is the number of months it takes to recover the cost of acquiring a customer.
    
    Parameters
    ----------
    cac : array-like
        Customer Acquisition Cost
    avg_monthly_revenue : array-like
        Average monthly revenue per customer
    gross_margin : array-like
        Gross margin percentage (0-100)
        
    Returns
    -------
    float or numpy.ndarray
        CAC Payback Period in months
        
    Examples
    --------
    >>> payback_period(1000, 100, 70)
    14.29
    """
    cac_arr = _ensure_array(cac)
    avg_monthly_revenue_arr = _ensure_array(avg_monthly_revenue)
    gross_margin_arr = _ensure_array(gross_margin)
    
    gross_margin_decimal = gross_margin_arr / 100.0
    
    monthly_gross_profit = avg_monthly_revenue_arr * gross_margin_decimal
    
    result = np.full_like(monthly_gross_profit, np.inf, dtype=np.float64)
    
    non_zero_mask = monthly_gross_profit != 0
    result[non_zero_mask] = cac_arr[non_zero_mask] / monthly_gross_profit[non_zero_mask]
    
    if (np.isscalar(cac) and 
        np.isscalar(avg_monthly_revenue) and 
        np.isscalar(gross_margin)):
        return float(result[0])
    
    return result

def customer_satisfaction_score(
    satisfaction_ratings: ArrayLike,
    max_rating: ArrayLike = 5
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Customer Satisfaction Score (CSAT).
    
    CSAT measures how satisfied customers are with a product, service, or interaction.
    
    Parameters
    ----------
    satisfaction_ratings : array-like
        Array of customer satisfaction ratings
    max_rating : array-like, default 5
        Maximum possible rating value
        
    Returns
    -------
    float or numpy.ndarray
        Customer Satisfaction Score as a percentage
        
    Examples
    --------
    >>> customer_satisfaction_score([4, 5, 3, 5, 4])
    84.0
    """
    ratings = np.array(satisfaction_ratings, dtype=np.float64)
    max_rating_arr = _ensure_array(max_rating)
    
    if len(ratings) == 0:
        return 0.0
    
    avg_rating = np.mean(ratings)
    result = np.array([(avg_rating / max_rating_arr[0]) * 100.0], dtype=np.float64)
    
    if np.isscalar(max_rating):
        return float(result[0])
    
    return result

def customer_effort_score(
    effort_ratings: ArrayLike,
    max_rating: ArrayLike = 7
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Customer Effort Score (CES).
    
    CES measures how much effort a customer has to exert to use a product or service.
    Lower scores are better.
    
    Parameters
    ----------
    effort_ratings : array-like
        Array of customer effort ratings
    max_rating : array-like, default 7
        Maximum possible rating value
        
    Returns
    -------
    float or numpy.ndarray
        Customer Effort Score (average)
        
    Examples
    --------
    >>> customer_effort_score([2, 3, 1, 2, 4])
    2.4
    """
    ratings = np.array(effort_ratings, dtype=np.float64)
    
    if len(ratings) == 0:
        return 0.0
    
    result = np.array([np.mean(ratings)], dtype=np.float64)
    
    if np.isscalar(effort_ratings):
        return float(result[0])
    
    return result

def average_revenue_per_user(
    total_revenue: ArrayLike,
    total_users: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Average Revenue Per User (ARPU).
    
    ARPU measures the average revenue generated per user or customer.
    
    Parameters
    ----------
    total_revenue : array-like
        Total revenue for the period
    total_users : array-like
        Total number of users or customers
        
    Returns
    -------
    float or numpy.ndarray
        Average Revenue Per User
        
    Examples
    --------
    >>> average_revenue_per_user(10000, 500)
    20.0
    """
    total_revenue_arr = _ensure_array(total_revenue)
    total_users_arr = _ensure_array(total_users)
    
    result = np.zeros_like(total_users_arr, dtype=np.float64)
    
    non_zero_mask = total_users_arr != 0
    result[non_zero_mask] = total_revenue_arr[non_zero_mask] / total_users_arr[non_zero_mask]
    
    if (np.isscalar(total_revenue) and np.isscalar(total_users)):
        return float(result[0])
    
    return result

def average_revenue_per_paying_user(
    total_revenue: ArrayLike,
    paying_users: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Average Revenue Per Paying User (ARPPU).
    
    ARPPU measures the average revenue generated per paying user or customer.
    
    Parameters
    ----------
    total_revenue : array-like
        Total revenue for the period
    paying_users : array-like
        Number of paying users or customers
        
    Returns
    -------
    float or numpy.ndarray
        Average Revenue Per Paying User
        
    Examples
    --------
    >>> average_revenue_per_paying_user(10000, 200)
    50.0
    """
    total_revenue_arr = _ensure_array(total_revenue)
    paying_users_arr = _ensure_array(paying_users)
    
    result = np.zeros_like(paying_users_arr, dtype=np.float64)
    
    non_zero_mask = paying_users_arr != 0
    result[non_zero_mask] = total_revenue_arr[non_zero_mask] / paying_users_arr[non_zero_mask]
    
    if (np.isscalar(total_revenue) and np.isscalar(paying_users)):
        return float(result[0])
    
    return result

def conversion_rate(
    conversions: ArrayLike,
    total_visitors: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Conversion Rate.
    
    Conversion Rate is the percentage of visitors who take a desired action.
    
    Parameters
    ----------
    conversions : array-like
        Number of conversions (desired actions taken)
    total_visitors : array-like
        Total number of visitors or users
        
    Returns
    -------
    float or numpy.ndarray
        Conversion Rate as a percentage
        
    Examples
    --------
    >>> conversion_rate(50, 1000)
    5.0
    """
    conversions_arr = _ensure_array(conversions)
    total_visitors_arr = _ensure_array(total_visitors)
    
    result = np.zeros_like(total_visitors_arr, dtype=np.float64)
    
    non_zero_mask = total_visitors_arr != 0
    result[non_zero_mask] = (conversions_arr[non_zero_mask] / total_visitors_arr[non_zero_mask]) * 100.0
    
    if (np.isscalar(conversions) and np.isscalar(total_visitors)):
        return float(result[0])
    
    return result

def customer_engagement_score(
    active_days: ArrayLike,
    total_days: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Customer Engagement Score.
    
    Customer Engagement Score measures how actively customers are using a product or service.
    
    Parameters
    ----------
    active_days : array-like
        Number of days the customer was active
    total_days : array-like
        Total number of days in the period
        
    Returns
    -------
    float or numpy.ndarray
        Customer Engagement Score as a percentage
        
    Examples
    --------
    >>> customer_engagement_score(15, 30)
    50.0
    """
    active_days_arr = _ensure_array(active_days)
    total_days_arr = _ensure_array(total_days)
    
    result = np.zeros_like(total_days_arr, dtype=np.float64)
    
    non_zero_mask = total_days_arr != 0
    result[non_zero_mask] = (active_days_arr[non_zero_mask] / total_days_arr[non_zero_mask]) * 100.0
    
    if (np.isscalar(active_days) and np.isscalar(total_days)):
        return float(result[0])
    
    return result

def daily_active_users_ratio(
    daily_active_users: ArrayLike,
    total_users: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Daily Active Users (DAU) Ratio.
    
    DAU Ratio measures the percentage of total users who are active on a daily basis.
    
    Parameters
    ----------
    daily_active_users : array-like
        Number of daily active users
    total_users : array-like
        Total number of users
        
    Returns
    -------
    float or numpy.ndarray
        Daily Active Users Ratio as a percentage
        
    Examples
    --------
    >>> daily_active_users_ratio(500, 2000)
    25.0
    """
    daily_active_users_arr = _ensure_array(daily_active_users)
    total_users_arr = _ensure_array(total_users)
    
    result = np.zeros_like(total_users_arr, dtype=np.float64)
    
    non_zero_mask = total_users_arr != 0
    result[non_zero_mask] = (daily_active_users_arr[non_zero_mask] / total_users_arr[non_zero_mask]) * 100.0
    
    if (np.isscalar(daily_active_users) and np.isscalar(total_users)):
        return float(result[0])
    
    return result

def monthly_active_users_ratio(
    monthly_active_users: ArrayLike,
    total_users: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Monthly Active Users (MAU) Ratio.
    
    MAU Ratio measures the percentage of total users who are active on a monthly basis.
    
    Parameters
    ----------
    monthly_active_users : array-like
        Number of monthly active users
    total_users : array-like
        Total number of users
        
    Returns
    -------
    float or numpy.ndarray
        Monthly Active Users Ratio as a percentage
        
    Examples
    --------
    >>> monthly_active_users_ratio(1500, 2000)
    75.0
    """
    monthly_active_users_arr = _ensure_array(monthly_active_users)
    total_users_arr = _ensure_array(total_users)
    
    result = np.zeros_like(total_users_arr, dtype=np.float64)
    
    non_zero_mask = total_users_arr != 0
    result[non_zero_mask] = (monthly_active_users_arr[non_zero_mask] / total_users_arr[non_zero_mask]) * 100.0
    
    if (np.isscalar(monthly_active_users) and np.isscalar(total_users)):
        return float(result[0])
    
    return result

def stickiness_ratio(
    daily_active_users: ArrayLike,
    monthly_active_users: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Stickiness Ratio (DAU/MAU).
    
    Stickiness Ratio measures how frequently active users engage with a product.
    
    Parameters
    ----------
    daily_active_users : array-like
        Number of daily active users
    monthly_active_users : array-like
        Number of monthly active users
        
    Returns
    -------
    float or numpy.ndarray
        Stickiness Ratio as a percentage
        
    Examples
    --------
    >>> stickiness_ratio(500, 1500)
    33.33
    """
    daily_active_users_arr = _ensure_array(daily_active_users)
    monthly_active_users_arr = _ensure_array(monthly_active_users)
    
    result = np.zeros_like(monthly_active_users_arr, dtype=np.float64)
    
    non_zero_mask = monthly_active_users_arr != 0
    result[non_zero_mask] = (daily_active_users_arr[non_zero_mask] / monthly_active_users_arr[non_zero_mask]) * 100.0
    
    if (np.isscalar(daily_active_users) and 
        np.isscalar(monthly_active_users)):
        return float(result[0])
    
    return result

def gross_margin(
    revenue: ArrayLike,
    cost_of_goods_sold: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Gross Margin.
    
    Gross Margin is the percentage of revenue that exceeds the cost of goods sold.
    
    Parameters
    ----------
    revenue : array-like
        Total revenue
    cost_of_goods_sold : array-like
        Cost of goods sold
        
    Returns
    -------
    float or numpy.ndarray
        Gross Margin as a percentage
        
    Examples
    --------
    >>> gross_margin(10000, 3000)
    70.0
    """
    revenue_arr = _ensure_array(revenue)
    cost_of_goods_sold_arr = _ensure_array(cost_of_goods_sold)
    
    gross_profit = revenue_arr - cost_of_goods_sold_arr
    
    result = np.zeros_like(revenue_arr, dtype=np.float64)
    
    non_zero_mask = revenue_arr != 0
    result[non_zero_mask] = (gross_profit[non_zero_mask] / revenue_arr[non_zero_mask]) * 100.0
    
    if (np.isscalar(revenue) and np.isscalar(cost_of_goods_sold)):
        return float(result[0])
    
    return result

def burn_rate(
    starting_capital: ArrayLike,
    ending_capital: ArrayLike,
    months: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Monthly Burn Rate.
    
    Burn Rate is the rate at which a company is losing money.
    
    Parameters
    ----------
    starting_capital : array-like
        Capital at the start of the period
    ending_capital : array-like
        Capital at the end of the period
    months : array-like
        Number of months in the period
        
    Returns
    -------
    float or numpy.ndarray
        Monthly Burn Rate
        
    Examples
    --------
    >>> burn_rate(100000, 70000, 6)
    5000.0
    """
    starting_capital_arr = _ensure_array(starting_capital)
    ending_capital_arr = _ensure_array(ending_capital)
    months_arr = _ensure_array(months)
    
    capital_used = starting_capital_arr - ending_capital_arr
    
    result = np.zeros_like(months_arr, dtype=np.float64)
    
    non_zero_mask = months_arr != 0
    result[non_zero_mask] = capital_used[non_zero_mask] / months_arr[non_zero_mask]
    
    if (np.isscalar(starting_capital) and 
        np.isscalar(ending_capital) and 
        np.isscalar(months)):
        return float(result[0])
    
    return result

def runway(
    current_capital: ArrayLike,
    monthly_burn_rate: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Runway in months.
    
    Runway is the amount of time a company has before it runs out of money.
    
    Parameters
    ----------
    current_capital : array-like
        Current capital
    monthly_burn_rate : array-like
        Monthly burn rate
        
    Returns
    -------
    float or numpy.ndarray
        Runway in months
        
    Examples
    --------
    >>> runway(100000, 5000)
    20.0
    """
    current_capital_arr = _ensure_array(current_capital)
    monthly_burn_rate_arr = _ensure_array(monthly_burn_rate)
    
    result = np.full_like(monthly_burn_rate_arr, np.inf, dtype=np.float64)
    
    non_zero_mask = monthly_burn_rate_arr != 0
    result[non_zero_mask] = current_capital_arr[non_zero_mask] / monthly_burn_rate_arr[non_zero_mask]
    
    if (np.isscalar(current_capital) and np.isscalar(monthly_burn_rate)):
        return float(result[0])
    
    return result

def virality_coefficient(
    new_users: ArrayLike,
    invites_sent: ArrayLike,
    total_users: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Virality Coefficient (K-factor).
    
    Virality Coefficient measures how many new users each existing user brings in.
    
    Parameters
    ----------
    new_users : array-like
        Number of new users from invites
    invites_sent : array-like
        Number of invites sent
    total_users : array-like
        Total number of users
        
    Returns
    -------
    float or numpy.ndarray
        Virality Coefficient
        
    Examples
    --------
    >>> virality_coefficient(100, 500, 1000)
    0.1
    """
    new_users_arr = _ensure_array(new_users)
    invites_sent_arr = _ensure_array(invites_sent)
    total_users_arr = _ensure_array(total_users)
    
    result = np.zeros_like(total_users_arr, dtype=np.float64)
    
    valid_mask = (total_users_arr != 0) & (invites_sent_arr != 0)
    
    if np.any(valid_mask):
        invites_per_user = np.zeros_like(total_users_arr, dtype=np.float64)
        conversion_rate_val = np.zeros_like(invites_sent_arr, dtype=np.float64)
        
        invites_per_user[valid_mask] = invites_sent_arr[valid_mask] / total_users_arr[valid_mask]
        conversion_rate_val[valid_mask] = new_users_arr[valid_mask] / invites_sent_arr[valid_mask]
        
        result[valid_mask] = invites_per_user[valid_mask] * conversion_rate_val[valid_mask]
    
    if (np.isscalar(new_users) and 
        np.isscalar(invites_sent) and 
        np.isscalar(total_users)):
        return float(result[0])
    
    return result

def time_to_value(
    onboarding_time: ArrayLike,
    setup_time: ArrayLike,
    learning_time: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Time to Value (TTV).
    
    Time to Value is the amount of time it takes for a customer to realize value from a product.
    
    Parameters
    ----------
    onboarding_time : array-like
        Time spent on onboarding
    setup_time : array-like
        Time spent on setup
    learning_time : array-like
        Time spent on learning
        
    Returns
    -------
    float or numpy.ndarray
        Time to Value
        
    Examples
    --------
    >>> time_to_value(2, 3, 5)
    10.0
    """
    onboarding_time_arr = _ensure_array(onboarding_time)
    setup_time_arr = _ensure_array(setup_time)
    learning_time_arr = _ensure_array(learning_time)
    
    result = onboarding_time_arr + setup_time_arr + learning_time_arr
    
    if (np.isscalar(onboarding_time) and 
        np.isscalar(setup_time) and 
        np.isscalar(learning_time)):
        return float(result[0])
    
    return result

def feature_adoption_rate(
    users_adopting_feature: ArrayLike,
    total_users: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Feature Adoption Rate.
    
    Feature Adoption Rate measures the percentage of users who adopt a specific feature.
    
    Parameters
    ----------
    users_adopting_feature : array-like
        Number of users who adopted the feature
    total_users : array-like
        Total number of users
        
    Returns
    -------
    float or numpy.ndarray
        Feature Adoption Rate as a percentage
        
    Examples
    --------
    >>> feature_adoption_rate(300, 1000)
    30.0
    """
    users_adopting_feature_arr = _ensure_array(users_adopting_feature)
    total_users_arr = _ensure_array(total_users)
    
    result = np.zeros_like(total_users_arr, dtype=np.float64)
    
    non_zero_mask = total_users_arr != 0
    result[non_zero_mask] = (users_adopting_feature_arr[non_zero_mask] / total_users_arr[non_zero_mask]) * 100.0
    
    if (np.isscalar(users_adopting_feature) and np.isscalar(total_users)):
        return float(result[0])
    
    return result

def roi(
    revenue: ArrayLike,
    costs: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Return on Investment (ROI).
    
    ROI measures the return on an investment relative to its cost.
    
    Parameters
    ----------
    revenue : array-like
        Revenue or return from the investment
    costs : array-like
        Cost of the investment
        
    Returns
    -------
    float or numpy.ndarray
        Return on Investment as a percentage
        
    Examples
    --------
    >>> roi(150, 100)
    50.0
    >>> roi([150, 200, 250], [100, 120, 150])
    array([50., 66.67, 66.67])
    """
    revenue_arr = _ensure_array(revenue)
    costs_arr = _ensure_array(costs)
    
    result = np.zeros_like(costs_arr, dtype=np.float64)
    
    non_zero_mask = costs_arr != 0
    result[non_zero_mask] = ((revenue_arr[non_zero_mask] - costs_arr[non_zero_mask]) / costs_arr[non_zero_mask]) * 100.0
    
    if (np.isscalar(revenue) and np.isscalar(costs)):
        return float(result[0])

    return result

def net_revenue_retention(
    starting_revenue: ArrayLike,
    expansion_revenue: ArrayLike,
    contraction_revenue: ArrayLike,
    churned_revenue: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Net Revenue Retention (NRR).

    NRR (also called Net Dollar Retention) measures the percentage of recurring
    revenue retained from existing customers over a period, including expansion
    and net of contraction and churn. A value above 100% indicates that
    expansion outpaces lost revenue.

    Parameters
    ----------
    starting_revenue : array-like
        Recurring revenue from existing customers at the start of the period
    expansion_revenue : array-like
        Additional revenue from upsells/cross-sells to existing customers
    contraction_revenue : array-like
        Revenue lost from downgrades by existing customers
    churned_revenue : array-like
        Revenue lost from customers who cancelled

    Returns
    -------
    float or numpy.ndarray
        Net Revenue Retention as a percentage

    Examples
    --------
    >>> net_revenue_retention(10000, 2000, 500, 1000)
    105.0
    """
    starting_arr = _ensure_array(starting_revenue)
    expansion_arr = _ensure_array(expansion_revenue)
    contraction_arr = _ensure_array(contraction_revenue)
    churned_arr = _ensure_array(churned_revenue)

    retained_revenue = starting_arr + expansion_arr - contraction_arr - churned_arr

    result = np.zeros_like(starting_arr, dtype=np.float64)

    non_zero_mask = starting_arr != 0
    result[non_zero_mask] = (retained_revenue[non_zero_mask] / starting_arr[non_zero_mask]) * 100.0

    if (np.isscalar(starting_revenue) and
        np.isscalar(expansion_revenue) and
        np.isscalar(contraction_revenue) and
        np.isscalar(churned_revenue)):
        return float(result[0])

    return result

def gross_revenue_retention(
    starting_revenue: ArrayLike,
    contraction_revenue: ArrayLike,
    churned_revenue: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Gross Revenue Retention (GRR).

    GRR measures the percentage of recurring revenue retained from existing
    customers, excluding any expansion. It is capped at 100% and reflects how
    well a business holds on to its baseline revenue.

    Parameters
    ----------
    starting_revenue : array-like
        Recurring revenue from existing customers at the start of the period
    contraction_revenue : array-like
        Revenue lost from downgrades by existing customers
    churned_revenue : array-like
        Revenue lost from customers who cancelled

    Returns
    -------
    float or numpy.ndarray
        Gross Revenue Retention as a percentage

    Examples
    --------
    >>> gross_revenue_retention(10000, 500, 1000)
    85.0
    """
    starting_arr = _ensure_array(starting_revenue)
    contraction_arr = _ensure_array(contraction_revenue)
    churned_arr = _ensure_array(churned_revenue)

    retained_revenue = starting_arr - contraction_arr - churned_arr

    result = np.zeros_like(starting_arr, dtype=np.float64)

    non_zero_mask = starting_arr != 0
    result[non_zero_mask] = (retained_revenue[non_zero_mask] / starting_arr[non_zero_mask]) * 100.0

    if (np.isscalar(starting_revenue) and
        np.isscalar(contraction_revenue) and
        np.isscalar(churned_revenue)):
        return float(result[0])

    return result

def revenue_growth_rate(
    current_revenue: ArrayLike,
    previous_revenue: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Revenue Growth Rate.

    Revenue Growth Rate measures the percentage change in revenue between two
    periods (e.g., month-over-month or year-over-year).

    Parameters
    ----------
    current_revenue : array-like
        Revenue in the current period
    previous_revenue : array-like
        Revenue in the previous period

    Returns
    -------
    float or numpy.ndarray
        Revenue Growth Rate as a percentage

    Examples
    --------
    >>> revenue_growth_rate(12000, 10000)
    20.0
    """
    current_arr = _ensure_array(current_revenue)
    previous_arr = _ensure_array(previous_revenue)

    result = np.zeros_like(previous_arr, dtype=np.float64)

    non_zero_mask = previous_arr != 0
    result[non_zero_mask] = ((current_arr[non_zero_mask] - previous_arr[non_zero_mask]) / previous_arr[non_zero_mask]) * 100.0

    if (np.isscalar(current_revenue) and np.isscalar(previous_revenue)):
        return float(result[0])

    return result

def saas_quick_ratio(
    new_mrr: ArrayLike,
    expansion_mrr: ArrayLike,
    churned_mrr: ArrayLike,
    contraction_mrr: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the SaaS Quick Ratio.

    The SaaS Quick Ratio measures growth efficiency by comparing revenue gained
    (new and expansion MRR) to revenue lost (churned and contraction MRR).
    A ratio above 4 is generally considered healthy.

    Parameters
    ----------
    new_mrr : array-like
        New MRR added from new customers
    expansion_mrr : array-like
        Additional MRR from existing customers (upsell/cross-sell)
    churned_mrr : array-like
        MRR lost from cancelled customers
    contraction_mrr : array-like
        MRR lost from downgrades

    Returns
    -------
    float or numpy.ndarray
        SaaS Quick Ratio (inf when there is no lost MRR)

    Examples
    --------
    >>> saas_quick_ratio(500, 200, 100, 75)
    4.0
    """
    new_arr = _ensure_array(new_mrr)
    expansion_arr = _ensure_array(expansion_mrr)
    churned_arr = _ensure_array(churned_mrr)
    contraction_arr = _ensure_array(contraction_mrr)

    gained = new_arr + expansion_arr
    lost = churned_arr + contraction_arr

    result = np.full_like(lost, np.inf, dtype=np.float64)

    non_zero_mask = lost != 0
    result[non_zero_mask] = gained[non_zero_mask] / lost[non_zero_mask]

    if (np.isscalar(new_mrr) and
        np.isscalar(expansion_mrr) and
        np.isscalar(churned_mrr) and
        np.isscalar(contraction_mrr)):
        return float(result[0])

    return result

def rule_of_40(
    revenue_growth_rate: ArrayLike,
    profit_margin: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Rule of 40 score.

    The Rule of 40 states that a healthy software company's revenue growth rate
    plus its profit margin should be at least 40%. This function returns the
    combined score so it can be compared against that 40 threshold.

    Parameters
    ----------
    revenue_growth_rate : array-like
        Revenue growth rate as a percentage
    profit_margin : array-like
        Profit margin as a percentage (e.g., EBITDA or free cash flow margin)

    Returns
    -------
    float or numpy.ndarray
        Rule of 40 score (growth rate + profit margin)

    Examples
    --------
    >>> rule_of_40(25, 20)
    45.0
    """
    growth_arr = _ensure_array(revenue_growth_rate)
    margin_arr = _ensure_array(profit_margin)

    result = growth_arr + margin_arr

    if (np.isscalar(revenue_growth_rate) and np.isscalar(profit_margin)):
        return float(result[0])

    return result

def magic_number(
    current_quarter_revenue: ArrayLike,
    previous_quarter_revenue: ArrayLike,
    sales_marketing_spend: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the SaaS Magic Number.

    The Magic Number measures sales and marketing efficiency by comparing the
    annualized increase in revenue to the prior period's sales and marketing
    spend. A value above 0.75 generally indicates efficient growth worth
    investing in further.

    Parameters
    ----------
    current_quarter_revenue : array-like
        Revenue in the current quarter
    previous_quarter_revenue : array-like
        Revenue in the previous quarter
    sales_marketing_spend : array-like
        Sales and marketing spend in the previous quarter

    Returns
    -------
    float or numpy.ndarray
        Magic Number (inf when there is no sales and marketing spend)

    Examples
    --------
    >>> magic_number(1300, 1000, 600)
    2.0
    """
    current_arr = _ensure_array(current_quarter_revenue)
    previous_arr = _ensure_array(previous_quarter_revenue)
    spend_arr = _ensure_array(sales_marketing_spend)

    annualized_new_revenue = (current_arr - previous_arr) * 4.0

    result = np.full_like(spend_arr, np.inf, dtype=np.float64)

    non_zero_mask = spend_arr != 0
    result[non_zero_mask] = annualized_new_revenue[non_zero_mask] / spend_arr[non_zero_mask]

    if (np.isscalar(current_quarter_revenue) and
        np.isscalar(previous_quarter_revenue) and
        np.isscalar(sales_marketing_spend)):
        return float(result[0])

    return result

def operating_margin(
    operating_income: ArrayLike,
    revenue: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Operating Margin.

    Operating Margin is the percentage of revenue left after paying for variable
    and operating costs, before interest and taxes.

    Parameters
    ----------
    operating_income : array-like
        Operating income (EBIT)
    revenue : array-like
        Total revenue

    Returns
    -------
    float or numpy.ndarray
        Operating Margin as a percentage

    Examples
    --------
    >>> operating_margin(2000, 10000)
    20.0
    """
    operating_income_arr = _ensure_array(operating_income)
    revenue_arr = _ensure_array(revenue)

    result = np.zeros_like(revenue_arr, dtype=np.float64)

    non_zero_mask = revenue_arr != 0
    result[non_zero_mask] = (operating_income_arr[non_zero_mask] / revenue_arr[non_zero_mask]) * 100.0

    if (np.isscalar(operating_income) and np.isscalar(revenue)):
        return float(result[0])

    return result

def net_profit_margin(
    net_income: ArrayLike,
    revenue: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Net Profit Margin.

    Net Profit Margin is the percentage of revenue that remains as profit after
    all expenses, including interest and taxes, have been deducted.

    Parameters
    ----------
    net_income : array-like
        Net income (bottom-line profit)
    revenue : array-like
        Total revenue

    Returns
    -------
    float or numpy.ndarray
        Net Profit Margin as a percentage

    Examples
    --------
    >>> net_profit_margin(1500, 10000)
    15.0
    """
    net_income_arr = _ensure_array(net_income)
    revenue_arr = _ensure_array(revenue)

    result = np.zeros_like(revenue_arr, dtype=np.float64)

    non_zero_mask = revenue_arr != 0
    result[non_zero_mask] = (net_income_arr[non_zero_mask] / revenue_arr[non_zero_mask]) * 100.0

    if (np.isscalar(net_income) and np.isscalar(revenue)):
        return float(result[0])

    return result

def ebitda_margin(
    ebitda: ArrayLike,
    revenue: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate EBITDA Margin.

    EBITDA Margin is earnings before interest, taxes, depreciation, and
    amortization expressed as a percentage of revenue. It is a measure of
    operating profitability independent of capital structure.

    Parameters
    ----------
    ebitda : array-like
        Earnings before interest, taxes, depreciation, and amortization
    revenue : array-like
        Total revenue

    Returns
    -------
    float or numpy.ndarray
        EBITDA Margin as a percentage

    Examples
    --------
    >>> ebitda_margin(3000, 10000)
    30.0
    """
    ebitda_arr = _ensure_array(ebitda)
    revenue_arr = _ensure_array(revenue)

    result = np.zeros_like(revenue_arr, dtype=np.float64)

    non_zero_mask = revenue_arr != 0
    result[non_zero_mask] = (ebitda_arr[non_zero_mask] / revenue_arr[non_zero_mask]) * 100.0

    if (np.isscalar(ebitda) and np.isscalar(revenue)):
        return float(result[0])

    return result

def average_revenue_per_account(
    total_revenue: ArrayLike,
    total_accounts: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Average Revenue Per Account (ARPA).

    ARPA measures the average revenue generated per account, which is useful in
    B2B contexts where a single account may include multiple users or seats.

    Parameters
    ----------
    total_revenue : array-like
        Total revenue for the period
    total_accounts : array-like
        Total number of accounts

    Returns
    -------
    float or numpy.ndarray
        Average Revenue Per Account

    Examples
    --------
    >>> average_revenue_per_account(50000, 250)
    200.0
    """
    total_revenue_arr = _ensure_array(total_revenue)
    total_accounts_arr = _ensure_array(total_accounts)

    result = np.zeros_like(total_accounts_arr, dtype=np.float64)

    non_zero_mask = total_accounts_arr != 0
    result[non_zero_mask] = total_revenue_arr[non_zero_mask] / total_accounts_arr[non_zero_mask]

    if (np.isscalar(total_revenue) and np.isscalar(total_accounts)):
        return float(result[0])

    return result

def customer_concentration(
    top_customer_revenue: ArrayLike,
    total_revenue: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Customer Concentration Risk.

    Customer Concentration measures the share of total revenue that comes from
    the largest customer or a group of top customers. A high concentration
    indicates greater revenue risk if those customers leave.

    Parameters
    ----------
    top_customer_revenue : array-like
        Revenue from the top customer(s)
    total_revenue : array-like
        Total revenue across all customers

    Returns
    -------
    float or numpy.ndarray
        Customer Concentration as a percentage

    Examples
    --------
    >>> customer_concentration(4000, 10000)
    40.0
    """
    top_customer_revenue_arr = _ensure_array(top_customer_revenue)
    total_revenue_arr = _ensure_array(total_revenue)

    result = np.zeros_like(total_revenue_arr, dtype=np.float64)

    non_zero_mask = total_revenue_arr != 0
    result[non_zero_mask] = (top_customer_revenue_arr[non_zero_mask] / total_revenue_arr[non_zero_mask]) * 100.0

    if (np.isscalar(top_customer_revenue) and np.isscalar(total_revenue)):
        return float(result[0])

    return result

def bounce_rate(
    single_page_sessions: ArrayLike,
    total_sessions: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Bounce Rate.

    Bounce Rate is the percentage of sessions in which a visitor leaves after
    viewing only a single page without further interaction. Lower is better.

    Parameters
    ----------
    single_page_sessions : array-like
        Number of single-page (bounced) sessions
    total_sessions : array-like
        Total number of sessions

    Returns
    -------
    float or numpy.ndarray
        Bounce Rate as a percentage

    Examples
    --------
    >>> bounce_rate(400, 1000)
    40.0
    """
    single_page_sessions_arr = _ensure_array(single_page_sessions)
    total_sessions_arr = _ensure_array(total_sessions)

    result = np.zeros_like(total_sessions_arr, dtype=np.float64)

    non_zero_mask = total_sessions_arr != 0
    result[non_zero_mask] = (single_page_sessions_arr[non_zero_mask] / total_sessions_arr[non_zero_mask]) * 100.0

    if (np.isscalar(single_page_sessions) and np.isscalar(total_sessions)):
        return float(result[0])

    return result

def cart_abandonment_rate(
    completed_purchases: ArrayLike,
    initiated_checkouts: ArrayLike
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Cart Abandonment Rate.

    Cart Abandonment Rate is the percentage of shopping carts that are created
    but not converted into a completed purchase. Lower is better.

    Parameters
    ----------
    completed_purchases : array-like
        Number of completed purchases
    initiated_checkouts : array-like
        Number of initiated checkouts (carts created)

    Returns
    -------
    float or numpy.ndarray
        Cart Abandonment Rate as a percentage

    Examples
    --------
    >>> cart_abandonment_rate(300, 1000)
    70.0
    """
    completed_purchases_arr = _ensure_array(completed_purchases)
    initiated_checkouts_arr = _ensure_array(initiated_checkouts)

    result = np.zeros_like(initiated_checkouts_arr, dtype=np.float64)

    non_zero_mask = initiated_checkouts_arr != 0
    result[non_zero_mask] = (1.0 - (completed_purchases_arr[non_zero_mask] / initiated_checkouts_arr[non_zero_mask])) * 100.0

    if (np.isscalar(completed_purchases) and np.isscalar(initiated_checkouts)):
        return float(result[0])

    return result

def days_sales_outstanding(
    accounts_receivable: ArrayLike,
    total_credit_sales: ArrayLike,
    days: ArrayLike = 365
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate Days Sales Outstanding (DSO).

    DSO measures the average number of days it takes a company to collect
    payment after a sale has been made on credit. A lower DSO indicates faster
    collection of receivables.

    Parameters
    ----------
    accounts_receivable : array-like
        Outstanding accounts receivable
    total_credit_sales : array-like
        Total credit sales for the period
    days : array-like, default 365
        Number of days in the period

    Returns
    -------
    float or numpy.ndarray
        Days Sales Outstanding (inf when there are no credit sales)

    Examples
    --------
    >>> days_sales_outstanding(50000, 500000, 365)
    36.5
    """
    accounts_receivable_arr = _ensure_array(accounts_receivable)
    total_credit_sales_arr = _ensure_array(total_credit_sales)
    days_arr = np.broadcast_to(_ensure_array(days), total_credit_sales_arr.shape)

    result = np.full_like(total_credit_sales_arr, np.inf, dtype=np.float64)

    non_zero_mask = total_credit_sales_arr != 0
    result[non_zero_mask] = (accounts_receivable_arr[non_zero_mask] / total_credit_sales_arr[non_zero_mask]) * days_arr[non_zero_mask]

    if (np.isscalar(accounts_receivable) and
        np.isscalar(total_credit_sales) and
        np.isscalar(days)):
        return float(result[0])

    return result