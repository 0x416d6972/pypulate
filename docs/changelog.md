# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0]

### Added
- New business KPIs in `pypulate.kpi` and the `KPI` class:
  - `net_revenue_retention` (NRR) and `gross_revenue_retention` (GRR)
  - `revenue_growth_rate`, `saas_quick_ratio`, `rule_of_40`, `magic_number`
  - `operating_margin`, `net_profit_margin`, `ebitda_margin`
  - `average_revenue_per_account` (ARPA) and `customer_concentration`
  - `bounce_rate`, `cart_abandonment_rate`, `days_sales_outstanding` (DSO)
- `KPI.health` now incorporates Net Revenue Retention and Net Profit Margin.
- Tests and documentation for all new KPIs.

## [0.3.0]
- Added feature rich preprocessing methods for Parray.
- Added feature rich statistics methods for Parray
- Added complete test of methods.
- Improved docs.

## [0.2.2]
- Various credit and loan models added.


## [0.2.1]
- Added more pricing models.
- Minor improvements in code and docs.

## [0.2.0]

### Added
- New Class for Allocation portfolio
- mean_variance_optimization,
- minimum_variance_portfolio,
- maximum_sharpe_ratio,
- risk_parity_portfolio,
- maximum_diversification_portfolio,
- equal_weight_portfolio,
- market_cap_weight_portfolio,
- hierarchical_risk_parity,
- black_litterman,
- kelly_criterion_optimization

## [0.1.0] - 2025-03-04

### Added
- Initial release 