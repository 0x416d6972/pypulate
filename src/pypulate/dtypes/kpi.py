"""
KPI Module

This module provides a class for calculating various business metrics
commonly used in SaaS and subscription-based businesses.
"""

import numpy as np
from typing import Union, Dict
from numpy.typing import ArrayLike, NDArray
from ..kpi.business_kpi import (
    churn_rate, retention_rate, customer_lifetime_value,
    customer_acquisition_cost, monthly_recurring_revenue,
    total_monthly_recurring_revenue,
    annual_recurring_revenue, net_promoter_score,
    revenue_churn_rate, expansion_revenue_rate,
    ltv_cac_ratio, payback_period, customer_satisfaction_score,
    customer_effort_score, average_revenue_per_user,
    average_revenue_per_paying_user, conversion_rate,
    customer_engagement_score, daily_active_users_ratio,
    monthly_active_users_ratio, stickiness_ratio,
    gross_margin, burn_rate, runway, virality_coefficient,
    time_to_value, feature_adoption_rate, roi,
    net_revenue_retention, gross_revenue_retention, revenue_growth_rate,
    saas_quick_ratio, rule_of_40, magic_number,
    operating_margin, net_profit_margin, ebitda_margin,
    average_revenue_per_account, customer_concentration,
    bounce_rate, cart_abandonment_rate, days_sales_outstanding
)


class KPI:
    """
    A class for calculating various business KPIs.
    
    This class provides methods for calculating common business metrics
    used in SaaS and subscription-based businesses and maintains state
    to assess overall business health.
    
    Examples
    --------
    >>> from pypulate.dtypes import KPI
    >>> kpi = KPI()
    >>> churn = kpi.churn_rate(100, 90, 10)
    >>> retention = kpi.retention_rate(100, 90, 10)
    >>> clv = kpi.customer_lifetime_value(100, 70, 5)
    >>> health = kpi.health
    """
    
    def __init__(self):
        """Initialize the KPI class with empty state."""
        self._state = {
            'churn_rate': None,
            'retention_rate': None,
            'customer_lifetime_value': None,
            'customer_acquisition_cost': None,
            'monthly_recurring_revenue': None,
            'total_monthly_recurring_revenue': None,
            'annual_recurring_revenue': None,
            'net_promoter_score': None,
            'revenue_churn_rate': None,
            'expansion_revenue_rate': None,
            'ltv_cac_ratio': None,
            'payback_period': None,
            'customer_satisfaction_score': None,
            'customer_effort_score': None,
            'average_revenue_per_user': None,
            'average_revenue_per_paying_user': None,
            'conversion_rate': None,
            'customer_engagement_score': None,
            'daily_active_users_ratio': None,
            'monthly_active_users_ratio': None,
            'stickiness_ratio': None,
            'gross_margin': None,
            'burn_rate': None,
            'runway': None,
            'virality_coefficient': None,
            'time_to_value': None,
            'feature_adoption_rate': None,
            'roi': None,
            'net_revenue_retention': None,
            'gross_revenue_retention': None,
            'revenue_growth_rate': None,
            'saas_quick_ratio': None,
            'rule_of_40': None,
            'magic_number': None,
            'operating_margin': None,
            'net_profit_margin': None,
            'ebitda_margin': None,
            'average_revenue_per_account': None,
            'customer_concentration': None,
            'bounce_rate': None,
            'cart_abandonment_rate': None,
            'days_sales_outstanding': None
        }
    
    @property
    def health(self) -> Dict[str, Union[float, str, Dict[str, Dict[str, Union[float, str]]], None]]:
        """
        Calculate and return the overall health of the business based on stored KPIs.
        
        Returns
        -------
        dict
            Dictionary containing health score and individual component scores
        """
        health_score = 0.0
        components = {}
        total_weight = 0.0 
        weight_per_metric = 0.10 
        
        if self._state['churn_rate'] is not None:
            churn_score = max(0, min(100, 100 - self._state['churn_rate']))
            health_score += churn_score * weight_per_metric
            total_weight += weight_per_metric
            components['churn_rate'] = {
                'score': churn_score,
                'status': 'Excellent' if churn_score >= 95 else 'Good' if churn_score >= 85 else 'Fair' if churn_score >= 70 else 'Poor' if churn_score >= 50 else 'Critical'
            }
            
        if self._state['retention_rate'] is not None:
            retention_score = self._state['retention_rate']
            health_score += retention_score * weight_per_metric
            total_weight += weight_per_metric
            components['retention_rate'] = {
                'score': retention_score,
                'status': 'Excellent' if retention_score >= 95 else 'Good' if retention_score >= 85 else 'Fair' if retention_score >= 70 else 'Poor' if retention_score >= 50 else 'Critical'
            }

        if self._state['ltv_cac_ratio'] is not None:
            ltv_cac_score = min(100, self._state['ltv_cac_ratio'] * 20)
            health_score += ltv_cac_score * weight_per_metric
            total_weight += weight_per_metric
            components['ltv_cac_ratio'] = {
                'score': ltv_cac_score,
                'status': 'Excellent' if ltv_cac_score >= 80 else 'Good' if ltv_cac_score >= 60 else 'Fair' if ltv_cac_score >= 40 else 'Poor' if ltv_cac_score >= 20 else 'Critical'
            }

        if self._state['gross_margin'] is not None:
            margin_score = self._state['gross_margin']
            health_score += margin_score * weight_per_metric
            total_weight += weight_per_metric
            components['gross_margin'] = {
                'score': margin_score,
                'status': 'Excellent' if margin_score >= 80 else 'Good' if margin_score >= 70 else 'Fair' if margin_score >= 50 else 'Poor' if margin_score >= 30 else 'Critical'
            }

        if self._state['net_promoter_score'] is not None:
            nps_score = max(0, min(100, self._state['net_promoter_score'] + 50))
            health_score += nps_score * weight_per_metric
            total_weight += weight_per_metric
            components['net_promoter_score'] = {
                'score': nps_score,
                'status': 'Excellent' if nps_score >= 60 else 'Good' if nps_score >= 50 else 'Fair' if nps_score >= 30 else 'Poor' if nps_score >= 20 else 'Critical'
            }

        if self._state['customer_satisfaction_score'] is not None:
            csat_score = self._state['customer_satisfaction_score']
            health_score += csat_score * weight_per_metric
            total_weight += weight_per_metric
            components['customer_satisfaction_score'] = {
                'score': csat_score,
                'status': 'Excellent' if csat_score >= 90 else 'Good' if csat_score >= 85 else 'Fair' if csat_score >= 70 else 'Poor' if csat_score >= 50 else 'Critical'
            }

        if self._state['expansion_revenue_rate'] is not None:
            expansion_score = min(100, max(0, self._state['expansion_revenue_rate'] * 2))
            health_score += expansion_score * weight_per_metric
            total_weight += weight_per_metric
            components['expansion_revenue_rate'] = {
                'score': expansion_score,
                'status': 'Excellent' if expansion_score >= 30 else 'Good' if expansion_score >= 20 else 'Fair' if expansion_score >= 10 else 'Poor' if expansion_score >= 5 else 'Critical'
            }

        if self._state['stickiness_ratio'] is not None:
            stickiness_score = self._state['stickiness_ratio']
            health_score += stickiness_score * weight_per_metric
            total_weight += weight_per_metric
            components['stickiness_ratio'] = {
                'score': stickiness_score,
                'status': 'Excellent' if stickiness_score >= 70 else 'Good' if stickiness_score >= 50 else 'Fair' if stickiness_score >= 30 else 'Poor' if stickiness_score >= 20 else 'Critical'
            }

        if self._state['runway'] is not None:
            runway_score = min(100, max(0, self._state['runway'] * 5))
            health_score += runway_score * weight_per_metric
            total_weight += weight_per_metric
            components['runway'] = {
                'score': runway_score,
                'status': 'Excellent' if runway_score >= 18 else 'Good' if runway_score >= 12 else 'Fair' if runway_score >= 6 else 'Poor' if runway_score >= 3 else 'Critical'
            }

        if self._state['roi'] is not None:
            roi_score = min(100, max(0, self._state['roi']))
            health_score += roi_score * weight_per_metric
            total_weight += weight_per_metric
            components['roi'] = {
                'score': roi_score,
                'status': 'Excellent' if roi_score >= 50 else 'Good' if roi_score >= 30 else 'Fair' if roi_score >= 15 else 'Poor' if roi_score >= 5 else 'Critical'
            }

        if self._state['net_revenue_retention'] is not None:
            nrr_score = min(100, max(0, self._state['net_revenue_retention']))
            health_score += nrr_score * weight_per_metric
            total_weight += weight_per_metric
            components['net_revenue_retention'] = {
                'score': nrr_score,
                'status': 'Excellent' if nrr_score >= 120 else 'Good' if nrr_score >= 100 else 'Fair' if nrr_score >= 90 else 'Poor' if nrr_score >= 80 else 'Critical'
            }

        if self._state['net_profit_margin'] is not None:
            net_margin_score = min(100, max(0, self._state['net_profit_margin']))
            health_score += net_margin_score * weight_per_metric
            total_weight += weight_per_metric
            components['net_profit_margin'] = {
                'score': net_margin_score,
                'status': 'Excellent' if net_margin_score >= 20 else 'Good' if net_margin_score >= 10 else 'Fair' if net_margin_score >= 5 else 'Poor' if net_margin_score >= 0 else 'Critical'
            }

        final_score = (health_score / total_weight) if total_weight > 0 else None

        return {
            'overall_score': final_score,
            'status': ('Excellent' if final_score >= 90 
                      else 'Good' if final_score >= 75 
                      else 'Fair' if final_score >= 60 
                      else 'Poor' if final_score >= 45 
                      else 'Critical') if final_score is not None else 'Not enough data',
            'components': components,
            'metrics_counted': round(total_weight / weight_per_metric)  
        }
    
    def churn_rate(
        self,
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
        result = churn_rate(customers_start, customers_end, new_customers)
        if isinstance(result, (int, float)):
            self._state['churn_rate'] = result
        return result
    
    def retention_rate(
        self,
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
        result = retention_rate(customers_start, customers_end, new_customers)
        if isinstance(result, (int, float)):
            self._state['retention_rate'] = result
        return result
    
    def customer_lifetime_value(
        self,
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
        result = customer_lifetime_value(
            avg_revenue_per_customer, gross_margin, churn_rate_value, discount_rate
        )
        if isinstance(result, (int, float)):
            self._state['customer_lifetime_value'] = result
        return result
    
    def customer_acquisition_cost(
        self,
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
        result = customer_acquisition_cost(marketing_costs, sales_costs, new_customers)
        if isinstance(result, (int, float)):
            self._state['customer_acquisition_cost'] = result
        return result
    
    def monthly_recurring_revenue(
        self,
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
        result = monthly_recurring_revenue(paying_customers, avg_revenue_per_customer)
        if isinstance(result, (int, float)):
            self._state['monthly_recurring_revenue'] = result
        return result
    
    def total_monthly_recurring_revenue(
        self,
        subscription_revenues: ArrayLike
    ) -> float:
        """
        Calculate the total Monthly Recurring Revenue (MRR) by summing individual subscriptions.

        Total MRR is the sum of the normalized monthly revenue of every active
        subscription.

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
        result = total_monthly_recurring_revenue(subscription_revenues)
        self._state['total_monthly_recurring_revenue'] = result
        return result

    def annual_recurring_revenue(
        self,
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
        result = annual_recurring_revenue(paying_customers, avg_revenue_per_customer)
        if isinstance(result, (int, float)):
            self._state['annual_recurring_revenue'] = result
        return result
    
    def net_promoter_score(
        self,
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
        result = net_promoter_score(promoters, detractors, total_respondents)
        if isinstance(result, (int, float)):
            self._state['net_promoter_score'] = result
        return result
    
    def revenue_churn_rate(
        self,
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
        result = revenue_churn_rate(revenue_start, revenue_end, new_revenue)
        if isinstance(result, (int, float)):
            self._state['revenue_churn_rate'] = result
        return result
    
    def expansion_revenue_rate(
        self,
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
        result = expansion_revenue_rate(upsell_revenue, cross_sell_revenue, revenue_start)
        if isinstance(result, (int, float)):
            self._state['expansion_revenue_rate'] = result
        return result
    
    def ltv_cac_ratio(
        self,
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
        result = ltv_cac_ratio(ltv, cac)
        if isinstance(result, (int, float)):
            self._state['ltv_cac_ratio'] = result
        return result
    
    def payback_period(
        self,
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
        result = payback_period(cac, avg_monthly_revenue, gross_margin)
        if isinstance(result, (int, float)):
            self._state['payback_period'] = result
        return result
    
    def customer_satisfaction_score(
        self,
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
        result = customer_satisfaction_score(satisfaction_ratings, max_rating)
        if isinstance(result, (int, float)):
            self._state['customer_satisfaction_score'] = result
        return result
    
    def customer_effort_score(
        self,
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
        result = customer_effort_score(effort_ratings, max_rating)
        if isinstance(result, (int, float)):
            self._state['customer_effort_score'] = result
        return result
    
    def average_revenue_per_user(
        self,
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
        result = average_revenue_per_user(total_revenue, total_users)
        if isinstance(result, (int, float)):
            self._state['average_revenue_per_user'] = result
        return result
    
    def average_revenue_per_paying_user(
        self,
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
        result = average_revenue_per_paying_user(total_revenue, paying_users)
        if isinstance(result, (int, float)):
            self._state['average_revenue_per_paying_user'] = result
        return result
    
    def conversion_rate(
        self,
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
        result = conversion_rate(conversions, total_visitors)
        if isinstance(result, (int, float)):
            self._state['conversion_rate'] = result
        return result
    
    def customer_engagement_score(
        self,
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
        result = customer_engagement_score(active_days, total_days)
        if isinstance(result, (int, float)):
            self._state['customer_engagement_score'] = result
        return result
    
    def daily_active_users_ratio(
        self,
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
        result = daily_active_users_ratio(daily_active_users, total_users)
        if isinstance(result, (int, float)):
            self._state['daily_active_users_ratio'] = result
        return result
    
    def monthly_active_users_ratio(
        self,
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
        result = monthly_active_users_ratio(monthly_active_users, total_users)
        if isinstance(result, (int, float)):
            self._state['monthly_active_users_ratio'] = result
        return result
    
    def stickiness_ratio(
        self,
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
        result = stickiness_ratio(daily_active_users, monthly_active_users)
        if isinstance(result, (int, float)):
            self._state['stickiness_ratio'] = result
        return result
    
    def gross_margin(
        self,
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
        result = gross_margin(revenue, cost_of_goods_sold)
        if isinstance(result, (int, float)):
            self._state['gross_margin'] = result
        return result
    
    def burn_rate(
        self,
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
        result = burn_rate(starting_capital, ending_capital, months)
        if isinstance(result, (int, float)):
            self._state['burn_rate'] = result
        return result
    
    def runway(
        self,
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
        result = runway(current_capital, monthly_burn_rate)
        if isinstance(result, (int, float)):
            self._state['runway'] = result
        return result
    
    def virality_coefficient(
        self,
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
        result = virality_coefficient(new_users, invites_sent, total_users)
        if isinstance(result, (int, float)):
            self._state['virality_coefficient'] = result
        return result
    
    def time_to_value(
        self,
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
        result = time_to_value(onboarding_time, setup_time, learning_time)
        if isinstance(result, (int, float)):
            self._state['time_to_value'] = result
        return result
    
    def feature_adoption_rate(
        self,
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
        result = feature_adoption_rate(users_adopting_feature, total_users)
        if isinstance(result, (int, float)):
            self._state['feature_adoption_rate'] = result
        return result
    
    def roi(
        self,
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
        result = roi(revenue, costs)
        if isinstance(result, (int, float)):
            self._state['roi'] = result
        return result

    def net_revenue_retention(
        self,
        starting_revenue: ArrayLike,
        expansion_revenue: ArrayLike,
        contraction_revenue: ArrayLike,
        churned_revenue: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Net Revenue Retention (NRR).

        NRR measures the percentage of recurring revenue retained from existing
        customers over a period, including expansion and net of contraction and churn.

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
        result = net_revenue_retention(
            starting_revenue, expansion_revenue, contraction_revenue, churned_revenue
        )
        if isinstance(result, (int, float)):
            self._state['net_revenue_retention'] = result
        return result

    def gross_revenue_retention(
        self,
        starting_revenue: ArrayLike,
        contraction_revenue: ArrayLike,
        churned_revenue: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Gross Revenue Retention (GRR).

        GRR measures the percentage of recurring revenue retained from existing
        customers, excluding any expansion.

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
        result = gross_revenue_retention(
            starting_revenue, contraction_revenue, churned_revenue
        )
        if isinstance(result, (int, float)):
            self._state['gross_revenue_retention'] = result
        return result

    def revenue_growth_rate(
        self,
        current_revenue: ArrayLike,
        previous_revenue: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Revenue Growth Rate.

        Revenue Growth Rate measures the percentage change in revenue between two periods.

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
        result = revenue_growth_rate(current_revenue, previous_revenue)
        if isinstance(result, (int, float)):
            self._state['revenue_growth_rate'] = result
        return result

    def saas_quick_ratio(
        self,
        new_mrr: ArrayLike,
        expansion_mrr: ArrayLike,
        churned_mrr: ArrayLike,
        contraction_mrr: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the SaaS Quick Ratio.

        The SaaS Quick Ratio compares revenue gained (new and expansion MRR) to
        revenue lost (churned and contraction MRR) as a measure of growth efficiency.

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
            SaaS Quick Ratio

        Examples
        --------
        >>> saas_quick_ratio(500, 200, 100, 75)
        4.0
        """
        result = saas_quick_ratio(new_mrr, expansion_mrr, churned_mrr, contraction_mrr)
        if isinstance(result, (int, float)):
            self._state['saas_quick_ratio'] = result
        return result

    def rule_of_40(
        self,
        revenue_growth_rate: ArrayLike,
        profit_margin: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the Rule of 40 score.

        The Rule of 40 states that a healthy software company's revenue growth rate
        plus its profit margin should be at least 40%.

        Parameters
        ----------
        revenue_growth_rate : array-like
            Revenue growth rate as a percentage
        profit_margin : array-like
            Profit margin as a percentage

        Returns
        -------
        float or numpy.ndarray
            Rule of 40 score (growth rate + profit margin)

        Examples
        --------
        >>> rule_of_40(25, 20)
        45.0
        """
        result = rule_of_40(revenue_growth_rate, profit_margin)
        if isinstance(result, (int, float)):
            self._state['rule_of_40'] = result
        return result

    def magic_number(
        self,
        current_quarter_revenue: ArrayLike,
        previous_quarter_revenue: ArrayLike,
        sales_marketing_spend: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the SaaS Magic Number.

        The Magic Number compares the annualized increase in revenue to the prior
        period's sales and marketing spend as a measure of sales efficiency.

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
            Magic Number

        Examples
        --------
        >>> magic_number(1300, 1000, 600)
        2.0
        """
        result = magic_number(
            current_quarter_revenue, previous_quarter_revenue, sales_marketing_spend
        )
        if isinstance(result, (int, float)):
            self._state['magic_number'] = result
        return result

    def operating_margin(
        self,
        operating_income: ArrayLike,
        revenue: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Operating Margin.

        Operating Margin is the percentage of revenue left after paying for
        operating costs, before interest and taxes.

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
        result = operating_margin(operating_income, revenue)
        if isinstance(result, (int, float)):
            self._state['operating_margin'] = result
        return result

    def net_profit_margin(
        self,
        net_income: ArrayLike,
        revenue: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Net Profit Margin.

        Net Profit Margin is the percentage of revenue that remains as profit
        after all expenses, including interest and taxes.

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
        result = net_profit_margin(net_income, revenue)
        if isinstance(result, (int, float)):
            self._state['net_profit_margin'] = result
        return result

    def ebitda_margin(
        self,
        ebitda: ArrayLike,
        revenue: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate EBITDA Margin.

        EBITDA Margin is earnings before interest, taxes, depreciation, and
        amortization expressed as a percentage of revenue.

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
        result = ebitda_margin(ebitda, revenue)
        if isinstance(result, (int, float)):
            self._state['ebitda_margin'] = result
        return result

    def average_revenue_per_account(
        self,
        total_revenue: ArrayLike,
        total_accounts: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Average Revenue Per Account (ARPA).

        ARPA measures the average revenue generated per account.

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
        result = average_revenue_per_account(total_revenue, total_accounts)
        if isinstance(result, (int, float)):
            self._state['average_revenue_per_account'] = result
        return result

    def customer_concentration(
        self,
        top_customer_revenue: ArrayLike,
        total_revenue: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Customer Concentration Risk.

        Customer Concentration measures the share of total revenue that comes from
        the largest customer or a group of top customers.

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
        result = customer_concentration(top_customer_revenue, total_revenue)
        if isinstance(result, (int, float)):
            self._state['customer_concentration'] = result
        return result

    def bounce_rate(
        self,
        single_page_sessions: ArrayLike,
        total_sessions: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Bounce Rate.

        Bounce Rate is the percentage of sessions in which a visitor leaves after
        viewing only a single page without further interaction.

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
        result = bounce_rate(single_page_sessions, total_sessions)
        if isinstance(result, (int, float)):
            self._state['bounce_rate'] = result
        return result

    def cart_abandonment_rate(
        self,
        completed_purchases: ArrayLike,
        initiated_checkouts: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Cart Abandonment Rate.

        Cart Abandonment Rate is the percentage of shopping carts that are created
        but not converted into a completed purchase.

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
        result = cart_abandonment_rate(completed_purchases, initiated_checkouts)
        if isinstance(result, (int, float)):
            self._state['cart_abandonment_rate'] = result
        return result

    def days_sales_outstanding(
        self,
        accounts_receivable: ArrayLike,
        total_credit_sales: ArrayLike,
        days: ArrayLike = 365
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Days Sales Outstanding (DSO).

        DSO measures the average number of days it takes a company to collect
        payment after a sale has been made on credit.

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
            Days Sales Outstanding

        Examples
        --------
        >>> days_sales_outstanding(50000, 500000, 365)
        36.5
        """
        result = days_sales_outstanding(accounts_receivable, total_credit_sales, days)
        if isinstance(result, (int, float)):
            self._state['days_sales_outstanding'] = result
        return result 