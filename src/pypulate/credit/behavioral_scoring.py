"""
Behavioral Scoring Model

This module provides functions for assessing credit risk based on account behavior
over time, rather than just application or bureau data. Behavioral scoring is
particularly useful for account management, credit line increases, and retention strategies.

References:
- Siddiqi, N. (2017). Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards
- Anderson, R. (2007). The Credit Scoring Toolkit: Theory and Practice for Retail Credit Risk Management
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from enum import Enum
from datetime import datetime, timedelta

class BehaviorRiskLevel(Enum):
    """Enum representing different behavior risk levels."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

class AccountStatus(Enum):
    """Enum representing different account statuses."""
    CURRENT = "current"
    DELINQUENT_30 = "30_days_past_due"
    DELINQUENT_60 = "60_days_past_due"
    DELINQUENT_90 = "90_days_past_due"
    DELINQUENT_120_PLUS = "120_plus_days_past_due"
    CHARGED_OFF = "charged_off"
    CLOSED = "closed"

def calculate_behavioral_score(
    account_history: List[Dict[str, Any]],
    account_type: str = "credit_card"
) -> Dict[str, Any]:
    """
    Calculate a behavioral score based on account history.
    
    Parameters
    ----------
    account_history : List[Dict[str, Any]]
        List of account snapshots in chronological order, with each item containing:
        - date: date of snapshot
        - balance: current balance
        - limit: credit limit (if applicable)
        - payment_amount: amount paid
        - min_payment_due: minimum payment due
        - days_past_due: days past due (0 if current)
        - transaction_count: number of transactions
        - transaction_volume: total transaction amount
    account_type : str, default "credit_card"
        Type of account ("credit_card", "personal_loan", "mortgage", "auto_loan")
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - score: int, behavioral score (300-850)
        - risk_level: BehaviorRiskLevel
        - risk_indicators: Dict, key risk indicators
        - suggested_actions: List[str], suggested actions
    """
    if not account_history:
        raise ValueError("Account history cannot be empty")
    
    # Sort history by date if not already sorted
    account_history = sorted(account_history, key=lambda x: x.get('date'))
    
    # Initialize scores for different behavior dimensions
    dimension_scores = {
        "payment_behavior": 0,
        "utilization_behavior": 0,
        "stability": 0,
        "spending_pattern": 0,
        "delinquency": 0
    }
    
    # Define dimension weights based on account type
    if account_type == "credit_card":
        weights = {
            "payment_behavior": 0.30,
            "utilization_behavior": 0.25,
            "stability": 0.15,
            "spending_pattern": 0.15,
            "delinquency": 0.15
        }
    elif account_type in ["personal_loan", "auto_loan"]:
        weights = {
            "payment_behavior": 0.35,
            "utilization_behavior": 0.10,
            "stability": 0.20,
            "spending_pattern": 0.05,
            "delinquency": 0.30
        }
    elif account_type == "mortgage":
        weights = {
            "payment_behavior": 0.40,
            "utilization_behavior": 0.05,
            "stability": 0.25,
            "spending_pattern": 0.05,
            "delinquency": 0.25
        }
    else:
        weights = {
            "payment_behavior": 0.30,
            "utilization_behavior": 0.20,
            "stability": 0.20,
            "spending_pattern": 0.10,
            "delinquency": 0.20
        }
    
    # Calculate risk indicators
    risk_indicators = {}
    
    # Payment Behavior Analysis
    payment_ratio_history = []
    for snapshot in account_history:
        min_payment = snapshot.get('min_payment_due', 0)
        if min_payment > 0:
            payment_amount = snapshot.get('payment_amount', 0)
            ratio = min(payment_amount / min_payment if min_payment > 0 else 0, 3)
            payment_ratio_history.append(ratio)
    
    if payment_ratio_history:
        avg_payment_ratio = sum(payment_ratio_history) / len(payment_ratio_history)
        recent_payment_ratio = payment_ratio_history[-1] if payment_ratio_history else 0
        payment_trend = _calculate_trend(payment_ratio_history)
        
        # Score payment behavior (0-100)
        if avg_payment_ratio >= 1.0:
            if recent_payment_ratio >= 1.5:
                dimension_scores["payment_behavior"] = 100
            elif recent_payment_ratio >= 1.0:
                dimension_scores["payment_behavior"] = 90
            else:
                dimension_scores["payment_behavior"] = 80
        elif avg_payment_ratio >= 0.9:
            dimension_scores["payment_behavior"] = 70
        elif avg_payment_ratio >= 0.7:
            dimension_scores["payment_behavior"] = 50
        else:
            dimension_scores["payment_behavior"] = 30
        
        # Adjust for trend
        if payment_trend > 0.1:  # Improving trend
            dimension_scores["payment_behavior"] = min(100, dimension_scores["payment_behavior"] + 10)
        elif payment_trend < -0.1:  # Deteriorating trend
            dimension_scores["payment_behavior"] = max(0, dimension_scores["payment_behavior"] - 10)
        
        risk_indicators["avg_payment_ratio"] = avg_payment_ratio
        risk_indicators["payment_trend"] = payment_trend
    else:
        dimension_scores["payment_behavior"] = 50  # Default if no payment history
    
    # Utilization Behavior Analysis
    utilization_history = []
    for snapshot in account_history:
        limit = snapshot.get('limit', 0)
        if limit > 0:
            balance = snapshot.get('balance', 0)
            ratio = min(balance / limit if limit > 0 else 0, 1.5)
            utilization_history.append(ratio)
    
    if utilization_history:
        avg_utilization = sum(utilization_history) / len(utilization_history)
        recent_utilization = utilization_history[-1] if utilization_history else 0
        utilization_trend = _calculate_trend(utilization_history)
        
        # Score utilization behavior (0-100)
        if account_type == "credit_card":
            if recent_utilization <= 0.3:
                dimension_scores["utilization_behavior"] = 100
            elif recent_utilization <= 0.5:
                dimension_scores["utilization_behavior"] = 80
            elif recent_utilization <= 0.7:
                dimension_scores["utilization_behavior"] = 60
            elif recent_utilization <= 0.9:
                dimension_scores["utilization_behavior"] = 40
            else:
                dimension_scores["utilization_behavior"] = 20
        else:
            if recent_utilization <= 0.5:
                dimension_scores["utilization_behavior"] = 100
            elif recent_utilization <= 0.7:
                dimension_scores["utilization_behavior"] = 80
            elif recent_utilization <= 0.9:
                dimension_scores["utilization_behavior"] = 60
            else:
                dimension_scores["utilization_behavior"] = 40
        
        # Adjust for trend
        if utilization_trend < -0.05:  # Decreasing utilization (good)
            dimension_scores["utilization_behavior"] = min(100, dimension_scores["utilization_behavior"] + 10)
        elif utilization_trend > 0.05:  # Increasing utilization (concerning)
            dimension_scores["utilization_behavior"] = max(0, dimension_scores["utilization_behavior"] - 10)
        
        risk_indicators["avg_utilization"] = avg_utilization
        risk_indicators["recent_utilization"] = recent_utilization
        risk_indicators["utilization_trend"] = utilization_trend
    else:
        dimension_scores["utilization_behavior"] = 50  # Default if no utilization history
    
    # Delinquency Analysis
    delinquency_history = [snapshot.get('days_past_due', 0) for snapshot in account_history]
    max_delinquency = max(delinquency_history) if delinquency_history else 0
    recent_delinquency = delinquency_history[-1] if delinquency_history else 0
    delinquency_frequency = sum(1 for dpd in delinquency_history if dpd > 0)
    
    # Score delinquency (0-100)
    if max_delinquency == 0:
        dimension_scores["delinquency"] = 100
    elif max_delinquency <= 30:
        dimension_scores["delinquency"] = 70
    elif max_delinquency <= 60:
        dimension_scores["delinquency"] = 50
    elif max_delinquency <= 90:
        dimension_scores["delinquency"] = 30
    else:
        dimension_scores["delinquency"] = 10
    
    # Adjust for frequency and recency
    if delinquency_frequency > 1:
        dimension_scores["delinquency"] = max(0, dimension_scores["delinquency"] - 10 * delinquency_frequency)
    
    if recent_delinquency > 0:
        dimension_scores["delinquency"] = max(0, dimension_scores["delinquency"] - 20)
    
    risk_indicators["max_delinquency"] = max_delinquency
    risk_indicators["delinquency_frequency"] = delinquency_frequency
    
    # Stability Analysis
    if len(account_history) >= 2:
        balance_volatility = _calculate_volatility([snapshot.get('balance', 0) for snapshot in account_history])
        payment_volatility = _calculate_volatility([snapshot.get('payment_amount', 0) for snapshot in account_history])
        
        # Normalize volatility (lower is better)
        avg_balance = sum(snapshot.get('balance', 0) for snapshot in account_history) / len(account_history)
        avg_payment = sum(snapshot.get('payment_amount', 0) for snapshot in account_history) / len(account_history)
        
        normalized_balance_volatility = balance_volatility / max(1, avg_balance)
        normalized_payment_volatility = payment_volatility / max(1, avg_payment)
        
        # Score stability (0-100)
        if normalized_balance_volatility <= 0.1 and normalized_payment_volatility <= 0.1:
            dimension_scores["stability"] = 100
        elif normalized_balance_volatility <= 0.2 and normalized_payment_volatility <= 0.2:
            dimension_scores["stability"] = 80
        elif normalized_balance_volatility <= 0.3 and normalized_payment_volatility <= 0.3:
            dimension_scores["stability"] = 60
        elif normalized_balance_volatility <= 0.5 and normalized_payment_volatility <= 0.5:
            dimension_scores["stability"] = 40
        else:
            dimension_scores["stability"] = 20
        
        risk_indicators["balance_volatility"] = normalized_balance_volatility
        risk_indicators["payment_volatility"] = normalized_payment_volatility
    else:
        dimension_scores["stability"] = 50  # Default if insufficient history
    
    # Spending Pattern Analysis (primarily for credit cards)
    if account_type == "credit_card" and len(account_history) >= 3:
        transaction_counts = [snapshot.get('transaction_count', 0) for snapshot in account_history]
        transaction_volumes = [snapshot.get('transaction_volume', 0) for snapshot in account_history]
        
        # Calculate trend and volatility
        count_trend = _calculate_trend(transaction_counts)
        volume_trend = _calculate_trend(transaction_volumes)
        
        count_volatility = _calculate_volatility(transaction_counts)
        volume_volatility = _calculate_volatility(transaction_volumes)
        
        # Calculate average per-transaction amount
        avg_transaction_sizes = []
        for i in range(len(account_history)):
            if transaction_counts[i] > 0:
                avg_size = transaction_volumes[i] / transaction_counts[i]
                avg_transaction_sizes.append(avg_size)
        
        avg_transaction_trend = _calculate_trend(avg_transaction_sizes) if avg_transaction_sizes else 0
        
        # Score spending pattern (0-100)
        # Stable count, increasing volume is generally good
        if count_volatility <= 0.2 and volume_trend >= 0:
            dimension_scores["spending_pattern"] = 90
        elif count_volatility <= 0.3 and abs(volume_trend) <= 0.1:
            dimension_scores["spending_pattern"] = 70
        elif count_volatility <= 0.5 and volume_volatility <= 0.5:
            dimension_scores["spending_pattern"] = 50
        else:
            dimension_scores["spending_pattern"] = 30
        
        # Adjust for sudden changes in transaction size (potentially concerning)
        if abs(avg_transaction_trend) > 0.3:
            dimension_scores["spending_pattern"] = max(0, dimension_scores["spending_pattern"] - 20)
        
        risk_indicators["transaction_count_trend"] = count_trend
        risk_indicators["transaction_volume_trend"] = volume_trend
        risk_indicators["avg_transaction_trend"] = avg_transaction_trend
    else:
        dimension_scores["spending_pattern"] = 50  # Default if not applicable or insufficient history
    
    # Calculate weighted score
    weighted_score = 0
    for dimension, score in dimension_scores.items():
        weighted_score += score * weights[dimension]
    
    # Map to 300-850 scale
    final_score = 300 + int((weighted_score / 100) * 550)
    
    # Determine risk level
    if final_score >= 750:
        risk_level = BehaviorRiskLevel.VERY_LOW
    elif final_score >= 700:
        risk_level = BehaviorRiskLevel.LOW
    elif final_score >= 650:
        risk_level = BehaviorRiskLevel.MEDIUM
    elif final_score >= 600:
        risk_level = BehaviorRiskLevel.HIGH
    elif final_score >= 550:
        risk_level = BehaviorRiskLevel.VERY_HIGH
    else:
        risk_level = BehaviorRiskLevel.CRITICAL
    
    # Generate suggested actions
    suggested_actions = _generate_suggested_actions(
        dimension_scores, 
        risk_indicators, 
        account_type
    )
    
    return {
        "score": final_score,
        "risk_level": risk_level,
        "dimension_scores": dimension_scores,
        "risk_indicators": risk_indicators,
        "suggested_actions": suggested_actions
    }

def calculate_delinquency_probability(
    account_history: List[Dict[str, Any]],
    forecast_months: int = 3,
    account_type: str = "credit_card"
) -> Dict[str, Any]:
    """
    Calculate probability of delinquency in the coming months.
    
    Parameters
    ----------
    account_history : List[Dict[str, Any]]
        List of account snapshots in chronological order
    forecast_months : int, default 3
        Number of months to forecast
    account_type : str, default "credit_card"
        Type of account
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - probability_30_dpd: float, probability of 30+ days delinquency
        - probability_60_dpd: float, probability of 60+ days delinquency
        - probability_90_dpd: float, probability of 90+ days delinquency
        - key_drivers: List[str], key drivers of probability
    """
    behavioral_score = calculate_behavioral_score(account_history, account_type)
    risk_indicators = behavioral_score["risk_indicators"]
    dimension_scores = behavioral_score["dimension_scores"]
    
    # Basic mapping from behavioral score to probabilities
    # These values would normally be calibrated with historical data
    score = behavioral_score["score"]
    base_prob_30 = _score_to_probability(score, 30)
    base_prob_60 = _score_to_probability(score, 60) 
    base_prob_90 = _score_to_probability(score, 90)
    
    # Adjust for prior delinquency (strongest predictor)
    prior_delinquency_adj = 1.0
    if risk_indicators.get("max_delinquency", 0) > 0:
        delinq_freq = risk_indicators.get("delinquency_frequency", 0)
        max_delinq = risk_indicators.get("max_delinquency", 0)
        
        if max_delinq >= 90:
            prior_delinquency_adj = 3.0
        elif max_delinq >= 60:
            prior_delinquency_adj = 2.5
        elif max_delinq >= 30:
            prior_delinquency_adj = 2.0
        
        # Multiple delinquencies indicate higher risk
        if delinq_freq > 1:
            prior_delinquency_adj += (delinq_freq - 1) * 0.5
    
    # Adjust for payment behavior
    payment_adj = 1.0
    payment_score = dimension_scores.get("payment_behavior", 50)
    if payment_score < 40:
        payment_adj = 1.5
    elif payment_score < 60:
        payment_adj = 1.2
    elif payment_score >= 80:
        payment_adj = 0.8
    
    # Adjust for utilization and trend
    utilization_adj = 1.0
    recent_util = risk_indicators.get("recent_utilization", 0)
    util_trend = risk_indicators.get("utilization_trend", 0)
    
    if recent_util > 0.9:
        utilization_adj = 1.4
    elif recent_util > 0.7:
        utilization_adj = 1.2
    
    if util_trend > 0.1:  # Increasing utilization
        utilization_adj *= 1.2
    
    # Adjust for forecast months
    time_adj = 1.0 + (forecast_months - 3) * 0.1  # Longer forecast = higher risk
    time_adj = max(0.8, min(1.5, time_adj))  # Cap adjustment
    
    # Apply adjustments
    prob_30_dpd = min(0.99, base_prob_30 * prior_delinquency_adj * payment_adj * utilization_adj * time_adj)
    prob_60_dpd = min(0.99, base_prob_60 * prior_delinquency_adj * payment_adj * utilization_adj * time_adj)
    prob_90_dpd = min(0.99, base_prob_90 * prior_delinquency_adj * payment_adj * utilization_adj * time_adj)
    
    # Determine key drivers
    key_drivers = []
    if prior_delinquency_adj > 1.0:
        key_drivers.append("Prior delinquency")
    if payment_adj > 1.0:
        key_drivers.append("Inconsistent payment behavior")
    if utilization_adj > 1.0:
        key_drivers.append("High utilization")
    if util_trend > 0.1:
        key_drivers.append("Increasing balance trend")
    
    if dimension_scores.get("stability", 50) < 50:
        key_drivers.append("Inconsistent account behavior")
    if account_type == "credit_card" and dimension_scores.get("spending_pattern", 50) < 40:
        key_drivers.append("Erratic spending pattern")
    
    if not key_drivers:
        key_drivers.append("Low overall behavioral score")
    
    return {
        "probability_30_dpd": prob_30_dpd,
        "probability_60_dpd": prob_60_dpd,
        "probability_90_dpd": prob_90_dpd,
        "key_drivers": key_drivers
    }

def calculate_account_status(
    days_past_due: int
) -> AccountStatus:
    """
    Calculate account status based on days past due.
    
    Parameters
    ----------
    days_past_due : int
        Days past due
        
    Returns
    -------
    AccountStatus
        Account status
    """
    if days_past_due == 0:
        return AccountStatus.CURRENT
    elif days_past_due <= 30:
        return AccountStatus.DELINQUENT_30
    elif days_past_due <= 60:
        return AccountStatus.DELINQUENT_60
    elif days_past_due <= 90:
        return AccountStatus.DELINQUENT_90
    else:
        return AccountStatus.DELINQUENT_120_PLUS

def evaluate_credit_line_increase(
    account_history: List[Dict[str, Any]],
    requested_increase: float,
    current_limit: float,
    account_type: str = "credit_card"
) -> Dict[str, Any]:
    """
    Evaluate a credit line increase request based on account behavior.
    
    Parameters
    ----------
    account_history : List[Dict[str, Any]]
        List of account snapshots in chronological order
    requested_increase : float
        Requested increase amount
    current_limit : float
        Current credit limit
    account_type : str, default "credit_card"
        Type of account
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - decision: str, "approve", "counter_offer", or "decline"
        - approved_amount: float, approved increase amount (0 if declined)
        - reason_codes: List[str], reason codes for decision
        - risk_indicators: Dict, key risk indicators
    """
    behavioral_score = calculate_behavioral_score(account_history, account_type)
    delinquency_probs = calculate_delinquency_probability(account_history, 6, account_type)
    
    score = behavioral_score["score"]
    risk_level = behavioral_score["risk_level"]
    risk_indicators = behavioral_score["risk_indicators"]
    
    # Calculate requested increase as percentage
    increase_percent = requested_increase / current_limit if current_limit > 0 else float('inf')
    
    # Calculate months since opening (approximate)
    account_age_months = len(account_history)
    
    # Determine eligible increase percentage based on risk level
    if risk_level == BehaviorRiskLevel.VERY_LOW:
        eligible_percent = 0.30
    elif risk_level == BehaviorRiskLevel.LOW:
        eligible_percent = 0.20
    elif risk_level == BehaviorRiskLevel.MEDIUM:
        eligible_percent = 0.10
    elif risk_level == BehaviorRiskLevel.HIGH:
        eligible_percent = 0.05
    else:
        eligible_percent = 0.0
    
    # Adjust for account age
    if account_age_months < 6:
        eligible_percent = 0.0
    elif account_age_months < 12:
        eligible_percent = min(eligible_percent, 0.10)
    
    # Adjust for payment behavior
    payment_score = behavioral_score["dimension_scores"].get("payment_behavior", 0)
    if payment_score < 50:
        eligible_percent *= 0.5
    elif payment_score > 80:
        eligible_percent *= 1.2
    
    # Adjust for high utilization
    recent_utilization = risk_indicators.get("recent_utilization", 0)
    if recent_utilization > 0.8:
        eligible_percent *= 0.5
    
    # Adjust for delinquency probability
    pd_90 = delinquency_probs.get("probability_90_dpd", 0)
    if pd_90 > 0.1:
        eligible_percent = 0.0
    elif pd_90 > 0.05:
        eligible_percent *= 0.5
    
    # Calculate maximum eligible increase
    max_eligible_increase = current_limit * eligible_percent
    
    # Make decision
    reason_codes = []
    
    if max_eligible_increase == 0:
        decision = "decline"
        approved_amount = 0
        
        if account_age_months < 6:
            reason_codes.append("Account too new")
        if pd_90 > 0.1:
            reason_codes.append("High risk of delinquency")
        if risk_level in [BehaviorRiskLevel.VERY_HIGH, BehaviorRiskLevel.CRITICAL]:
            reason_codes.append("High risk behavior profile")
        if risk_indicators.get("max_delinquency", 0) > 30:
            reason_codes.append("Prior delinquency on account")
    
    elif requested_increase <= max_eligible_increase:
        decision = "approve"
        approved_amount = requested_increase
        
        if payment_score > 80:
            reason_codes.append("Excellent payment history")
        if recent_utilization < 0.5:
            reason_codes.append("Responsible utilization")
        if score > 700:
            reason_codes.append("Strong behavioral score")
    
    else:
        decision = "counter_offer"
        approved_amount = max_eligible_increase
        
        if increase_percent > 0.3:
            reason_codes.append("Requested increase too high")
        if recent_utilization > 0.7:
            reason_codes.append("High recent utilization")
        if pd_90 > 0.03:
            reason_codes.append("Elevated delinquency risk")
    
    return {
        "decision": decision,
        "approved_amount": approved_amount,
        "reason_codes": reason_codes,
        "behavioral_score": score,
        "risk_level": risk_level,
        "delinquency_probability_90d": pd_90
    }

def _calculate_trend(values: List[float]) -> float:
    """Calculate linear trend in a series of values."""
    if not values or len(values) < 2:
        return 0
    
    n = len(values)
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(values) / n
    
    numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if abs(denominator) < 1e-10:
        return 0
    
    slope = numerator / denominator
    
    # Normalize by average value to get relative trend
    avg_value = max(0.1, abs(y_mean))
    return slope / avg_value

def _calculate_volatility(values: List[float]) -> float:
    """Calculate volatility (coefficient of variation) in a series of values."""
    if not values or len(values) < 2:
        return 0
    
    mean = sum(values) / len(values)
    if abs(mean) < 1e-10:
        return 0
    
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std_dev = variance ** 0.5
    
    return std_dev / max(0.1, abs(mean))

def _score_to_probability(score: int, delinquency_days: int) -> float:
    """Convert a behavioral score to a probability of delinquency."""
    # These curves are approximations and would normally be calibrated
    # with actual performance data
    if delinquency_days == 30:
        # 30+ DPD probability curve
        if score >= 750:
            return 0.01
        elif score >= 700:
            return 0.03
        elif score >= 650:
            return 0.08
        elif score >= 600:
            return 0.15
        elif score >= 550:
            return 0.25
        else:
            return 0.40
    
    elif delinquency_days == 60:
        # 60+ DPD probability curve (lower probabilities)
        if score >= 750:
            return 0.005
        elif score >= 700:
            return 0.015
        elif score >= 650:
            return 0.04
        elif score >= 600:
            return 0.08
        elif score >= 550:
            return 0.15
        else:
            return 0.25
    
    elif delinquency_days == 90:
        # 90+ DPD probability curve (even lower probabilities)
        if score >= 750:
            return 0.002
        elif score >= 700:
            return 0.008
        elif score >= 650:
            return 0.02
        elif score >= 600:
            return 0.05
        elif score >= 550:
            return 0.10
        else:
            return 0.18
    
    return 0.0

def _generate_suggested_actions(
    dimension_scores: Dict[str, float],
    risk_indicators: Dict[str, Any],
    account_type: str
) -> List[str]:
    """Generate suggested actions based on behavioral analysis."""
    actions = []
    
    # Payment Behavior
    payment_score = dimension_scores.get("payment_behavior", 50)
    if payment_score < 60:
        actions.append("Ensure at least minimum payment is made on time each month")
        if payment_score < 40:
            actions.append("Set up automatic payments to avoid missed payments")
    
    # Utilization
    utilization_score = dimension_scores.get("utilization_behavior", 50)
    recent_util = risk_indicators.get("recent_utilization", 0)
    if account_type == "credit_card" and utilization_score < 60:
        if recent_util > 0.7:
            actions.append("Reduce credit utilization below 30% of limit")
        if recent_util > 0.9:
            actions.append("Make additional payments to reduce balance")
    
    # Delinquency
    delinquency_score = dimension_scores.get("delinquency", 50)
    if delinquency_score < 70:
        actions.append("Bring account current and maintain on-time payments")
        if delinquency_score < 40:
            actions.append("Contact creditor to discuss hardship options if needed")
    
    # Stability
    stability_score = dimension_scores.get("stability", 50)
    if stability_score < 50:
        actions.append("Maintain consistent payment amounts")
        if stability_score < 30:
            actions.append("Avoid frequent balance fluctuations")
    
    # Spending Pattern
    if account_type == "credit_card":
        spending_score = dimension_scores.get("spending_pattern", 50)
        if spending_score < 50:
            actions.append("Maintain consistent spending patterns")
            if spending_score < 30:
                actions.append("Avoid sudden large purchases or cash advances")
    
    return actions 