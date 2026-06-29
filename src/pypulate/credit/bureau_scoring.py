"""
Bureau Scoring Models

This module provides interfaces to work with consumer credit bureau data
and simulate major scoring models like FICO and VantageScore.

References:
- FICO: https://www.myfico.com/credit-education/whats-in-your-credit-score
- VantageScore: https://vantagescore.com/understanding-credit-scores/
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any
import numpy as np

class BureauType(Enum):
    """Enum representing different credit bureaus."""
    EQUIFAX = "equifax"
    EXPERIAN = "experian"
    TRANSUNION = "transunion"
    GENERIC = "generic"

class ScoreType(Enum):
    """Enum representing different scoring models."""
    FICO_8 = "fico_8"
    FICO_9 = "fico_9"
    FICO_10 = "fico_10"
    FICO_10T = "fico_10t"
    VANTAGE_3 = "vantage_3"
    VANTAGE_4 = "vantage_4"

class RiskCategory(Enum):
    """Enum representing credit risk categories."""
    EXCEPTIONAL = "exceptional"
    VERY_GOOD = "very_good"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"

def get_fico_risk_category(score: int) -> RiskCategory:
    """
    Determine FICO score risk category.
    
    Parameters
    ----------
    score : int
        FICO credit score (300-850)
        
    Returns
    -------
    RiskCategory
        Risk category of the score
    """
    if score >= 800:
        return RiskCategory.EXCEPTIONAL
    elif score >= 740:
        return RiskCategory.VERY_GOOD
    elif score >= 670:
        return RiskCategory.GOOD
    elif score >= 580:
        return RiskCategory.FAIR
    elif score >= 300:
        return RiskCategory.POOR
    else:
        raise ValueError("FICO score must be between 300 and 850")

def get_vantage_risk_category(score: int) -> RiskCategory:
    """
    Determine VantageScore risk category.
    
    Parameters
    ----------
    score : int
        VantageScore credit score (300-850)
        
    Returns
    -------
    RiskCategory
        Risk category of the score
    """
    if score >= 781:
        return RiskCategory.EXCEPTIONAL
    elif score >= 661:
        return RiskCategory.GOOD
    elif score >= 601:
        return RiskCategory.FAIR
    elif score >= 500:
        return RiskCategory.POOR
    elif score >= 300:
        return RiskCategory.VERY_POOR
    else:
        raise ValueError("VantageScore must be between 300 and 850")

def calculate_bureau_score(
    bureau_data: Dict[str, Any],
    score_type: Union[str, ScoreType] = ScoreType.FICO_8,
    bureau: Union[str, BureauType] = BureauType.GENERIC
) -> Dict[str, Any]:
    """
    Calculate a credit score based on bureau data.
    
    Parameters
    ----------
    bureau_data : Dict[str, Any]
        Dictionary containing bureau data with the following keys:
        - payment_history: Dict with 'late_payments', 'days_past_due', 'public_records'
        - credit_utilization: Dict with 'total_balances', 'total_limits'
        - credit_age: Dict with 'oldest_account_age', 'average_account_age'
        - account_mix: Dict with counts of different account types
        - new_credit: Dict with 'inquiries_last_12m', 'new_accounts_last_12m'
    score_type : Union[str, ScoreType], default ScoreType.FICO_8
        Type of score to calculate
    bureau : Union[str, BureauType], default BureauType.GENERIC
        Bureau to use for calculation
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - score: int, calculated score
        - risk_category: RiskCategory, risk category
        - factor_scores: Dict, scores for each factor
        - score_type: ScoreType, type of score calculated
        - bureau: BureauType, bureau used
    """
    # Convert string inputs to enum types
    if isinstance(score_type, str):
        score_type = ScoreType(score_type)
    if isinstance(bureau, str):
        bureau = BureauType(bureau)
    
    # Initialize factor scores
    factor_scores = {}
    
    # Default weights based on FICO 8
    weights = {
        "payment_history": 0.35,
        "credit_utilization": 0.30,
        "credit_age": 0.15,
        "account_mix": 0.10,
        "new_credit": 0.10
    }
    
    # Adjust weights based on score type
    if score_type == ScoreType.FICO_10T:
        weights["payment_history"] = 0.40
        weights["credit_utilization"] = 0.25
        weights["account_mix"] = 0.10
    elif score_type in [ScoreType.VANTAGE_3, ScoreType.VANTAGE_4]:
        weights = {
            
            "payment_history": 0.40,
            "credit_utilization": 0.20,
            "credit_age": 0.21,
            "account_mix": 0.11,
            "new_credit": 0.08
        }
    
    # Calculate payment history score (base 0-100)
    ph_data = bureau_data.get("payment_history", {})
    late_payments = ph_data.get("late_payments", 0)
    days_past_due = ph_data.get("days_past_due", 0)
    public_records = ph_data.get("public_records", 0)
    
    # Calculate payment history score (higher is better)
    payment_history_score = 100 - (5 * late_payments) - (0.1 * days_past_due) - (20 * public_records)
    payment_history_score = max(0, min(100, payment_history_score))
    factor_scores["payment_history"] = payment_history_score
    
    # Calculate credit utilization score
    cu_data = bureau_data.get("credit_utilization", {})
    total_balances = cu_data.get("total_balances", 0)
    total_limits = cu_data.get("total_limits", 1)  # Avoid division by zero
    
    utilization_ratio = min(1.0, total_balances / total_limits)
    if utilization_ratio <= 0.10:
        utilization_score = 100
    elif utilization_ratio <= 0.30:
        utilization_score = 90 - ((utilization_ratio - 0.10) * 100)
    else:
        utilization_score = 70 - ((utilization_ratio - 0.30) * 100)
    
    utilization_score = max(0, min(100, utilization_score))
    factor_scores["credit_utilization"] = utilization_score
    
    # Calculate credit age score
    ca_data = bureau_data.get("credit_age", {})
    oldest_account_age = ca_data.get("oldest_account_age", 0)  # in months
    average_account_age = ca_data.get("average_account_age", 0)  # in months
    
    if oldest_account_age >= 120:  # 10+ years
        age_score = 100
    elif oldest_account_age >= 60:  # 5+ years
        age_score = 80 + (oldest_account_age - 60) / 3
    else:
        age_score = min(80, oldest_account_age)
    
    # Adjust for average age
    age_ratio = average_account_age / max(1, oldest_account_age)
    age_score = age_score * (0.8 + (0.2 * age_ratio))
    
    age_score = max(0, min(100, age_score))
    factor_scores["credit_age"] = age_score
    
    # Calculate account mix score
    am_data = bureau_data.get("account_mix", {})
    installment_accounts = am_data.get("installment", 0)
    revolving_accounts = am_data.get("revolving", 0)
    mortgage_accounts = am_data.get("mortgage", 0)
    total_accounts = installment_accounts + revolving_accounts + mortgage_accounts
    
    # Ideal mix: some of each type
    if total_accounts == 0:
        mix_score = 40  # No credit history
    else:
        has_installment = installment_accounts > 0
        has_revolving = revolving_accounts > 0
        has_mortgage = mortgage_accounts > 0
        
        if has_installment and has_revolving and has_mortgage:
            mix_score = 100
        elif has_installment and has_revolving:
            mix_score = 80
        elif has_revolving:
            mix_score = 70
        elif has_installment:
            mix_score = 60
        else:
            mix_score = 50
    
    factor_scores["account_mix"] = mix_score
    
    # Calculate new credit score
    nc_data = bureau_data.get("new_credit", {})
    inquiries_last_12m = nc_data.get("inquiries_last_12m", 0)
    new_accounts_last_12m = nc_data.get("new_accounts_last_12m", 0)
    
    inquiry_score = 100 - (inquiries_last_12m * 5)
    new_account_score = 100 - (new_accounts_last_12m * 10)
    new_credit_score = (inquiry_score + new_account_score) / 2
    new_credit_score = max(0, min(100, new_credit_score))
    factor_scores["new_credit"] = new_credit_score
    
    # Calculate weighted score
    weighted_score = 0
    for factor, score in factor_scores.items():
        weighted_score += score * weights[factor]
    
    # Map to 300-850 scale
    final_score = 300 + int((weighted_score / 100) * 550)
    
    # Get appropriate risk category
    if score_type in [ScoreType.FICO_8, ScoreType.FICO_9, ScoreType.FICO_10, ScoreType.FICO_10T]:
        risk_category = get_fico_risk_category(final_score)
    else:
        risk_category = get_vantage_risk_category(final_score)
    
    return {
        "score": final_score,
        "risk_category": risk_category,
        "factor_scores": factor_scores,
        "score_type": score_type,
        "bureau": bureau
    }

def estimate_approval_odds(
    score: int, 
    product_type: str, 
    score_type: Union[str, ScoreType] = ScoreType.FICO_8
) -> Dict[str, Any]:
    """
    Estimate approval odds for different credit products based on score.
    
    Parameters
    ----------
    score : int
        Credit score
    product_type : str
        Type of credit product ("credit_card", "mortgage", "auto_loan", "personal_loan")
    score_type : Union[str, ScoreType], default ScoreType.FICO_8
        Type of score
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - approval_odds: float, probability of approval (0-1)
        - rate_tier: str, likely rate tier (premium, standard, subprime)
        - suggested_actions: List[str], suggested actions to improve odds
    """
    if isinstance(score_type, str):
        score_type = ScoreType(score_type)
    
    product_thresholds = {
        "credit_card": {
            "premium": 720,
            "standard": 670,
            "subprime": 580
        },
        "mortgage": {
            "premium": 740,
            "standard": 700,
            "subprime": 620
        },
        "auto_loan": {
            "premium": 700,
            "standard": 660,
            "subprime": 600
        },
        "personal_loan": {
            "premium": 720,
            "standard": 680,
            "subprime": 600
        }
    }
    
    if product_type not in product_thresholds:
        raise ValueError(f"Invalid product_type: {product_type}")
    
    thresholds = product_thresholds[product_type]
    
    # Determine tier
    if score >= thresholds["premium"]:
        rate_tier = "premium"
        odds = 0.90 + ((score - thresholds["premium"]) / 1000)
    elif score >= thresholds["standard"]:
        rate_tier = "standard"
        odds = 0.70 + ((score - thresholds["standard"]) / (thresholds["premium"] - thresholds["standard"]) * 0.20)
    elif score >= thresholds["subprime"]:
        rate_tier = "subprime"
        odds = 0.40 + ((score - thresholds["subprime"]) / (thresholds["standard"] - thresholds["subprime"]) * 0.30)
    else:
        rate_tier = "high_risk"
        odds = max(0.05, (score - 300) / (thresholds["subprime"] - 300) * 0.35)
    
    # Suggest actions
    suggested_actions = []
    if rate_tier != "premium":
        suggested_actions.append("Reduce credit utilization below 30%")
        suggested_actions.append("Ensure all payments are made on time")
        
        if rate_tier in ["subprime", "high_risk"]:
            suggested_actions.append("Dispute any errors on credit report")
            suggested_actions.append("Consider becoming an authorized user on a well-established account")
            suggested_actions.append("Avoid applying for new credit in the next 6 months")
    
    return {
        "approval_odds": min(0.99, odds),
        "rate_tier": rate_tier,
        "suggested_actions": suggested_actions
    }

def calculate_score_impact(
    bureau_data: Dict[str, Any],
    action: str,
    parameters: Dict[str, Any],
    score_type: Union[str, ScoreType] = ScoreType.FICO_8
) -> Dict[str, Any]:
    """
    Calculate the impact of a specific action on a credit score.
    
    Parameters
    ----------
    bureau_data : Dict[str, Any]
        Dictionary containing bureau data
    action : str
        Action to simulate (e.g., "pay_down_balance", "add_account")
    parameters : Dict[str, Any]
        Parameters for the action
    score_type : Union[str, ScoreType], default ScoreType.FICO_8
        Type of score to calculate
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - original_score: int, original credit score
        - new_score: int, projected new score
        - score_change: int, point change
        - impact_timeframe: str, estimated time for change to take effect
    """
    # Calculate original score
    original_result = calculate_bureau_score(bureau_data, score_type)
    original_score = original_result["score"]
    
    # Create a copy of bureau data
    modified_data = bureau_data.copy()
    
    # Apply the action
    impact_timeframe = "1-2 months"
    
    if action == "pay_down_balance":
        amount = parameters.get("amount", 0)
        utilization = modified_data.setdefault("credit_utilization", {})
        utilization["total_balances"] = max(0, utilization.get("total_balances", 0) - amount)
    
    elif action == "add_account":
        account_type = parameters.get("account_type", "revolving")
        account_mix = modified_data.setdefault("account_mix", {})
        account_mix[account_type] = account_mix.get(account_type, 0) + 1
        
        new_credit = modified_data.setdefault("new_credit", {})
        new_credit["inquiries_last_12m"] = new_credit.get("inquiries_last_12m", 0) + 1
        new_credit["new_accounts_last_12m"] = new_credit.get("new_accounts_last_12m", 0) + 1
        
        impact_timeframe = "3-6 months"
    
    elif action == "remove_late_payment":
        count = parameters.get("count", 1)
        payment_history = modified_data.setdefault("payment_history", {})
        payment_history["late_payments"] = max(0, payment_history.get("late_payments", 0) - count)
        
        impact_timeframe = "1-2 billing cycles"
    
    elif action == "lower_credit_utilization":
        percentage = parameters.get("percentage", 0.3)
        utilization = modified_data.setdefault("credit_utilization", {})
        total_limits = utilization.get("total_limits", 1)
        utilization["total_balances"] = total_limits * percentage
    
    # Calculate new score
    new_result = calculate_bureau_score(modified_data, score_type)
    new_score = new_result["score"]
    
    return {
        "original_score": original_score,
        "new_score": new_score,
        "score_change": new_score - original_score,
        "impact_timeframe": impact_timeframe
    } 