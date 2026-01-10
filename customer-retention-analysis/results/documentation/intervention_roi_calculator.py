#!/usr/bin/env python3
"""
Standalone ROI Calculator for Retention Campaigns

Calculate break-even and ROI for different campaign scenarios.
"""

def calculate_roi(target_customers, avg_clv, discount_rate, 
                 avg_order_value, expected_orders, campaign_cost):
    """
    Calculate ROI for retention campaign.
    
    Parameters
    ----------
    target_customers : int
        Number of customers to target
    avg_clv : float
        Average CLV of target segment
    discount_rate : float
        Discount percentage (0-1)
    avg_order_value : float
        Average order value
    expected_orders : int
        Expected orders if retained
    campaign_cost : float
        Campaign execution cost per customer
    
    Returns
    -------
    dict
        ROI metrics
    """
    # Costs
    discount_cost = avg_order_value * expected_orders * discount_rate
    cost_per_customer = discount_cost + campaign_cost
    total_budget = cost_per_customer * target_customers
    
    # Value
    value_per_saved = avg_clv - discount_cost
    
    # Break-even
    breakeven_rate = cost_per_customer / value_per_saved
    breakeven_customers = target_customers * breakeven_rate
    
    return {
        'cost_per_customer': cost_per_customer,
        'value_per_saved': value_per_saved,
        'total_budget': total_budget,
        'breakeven_rate': breakeven_rate,
        'breakeven_customers': breakeven_customers
    }


def print_roi_scenarios(results, target_customers, retention_rates=[0.10, 0.20, 0.30, 0.40]):
    """Print ROI table for different retention rates."""
    print("\nROI Scenarios:")
    print("="*80)
    print(f"{'Retention':<12} {'Saved':<10} {'Revenue':<15} {'Cost':<15} {'Net ROI':<15} {'Status'}")
    print("-"*80)
    
    for rate in retention_rates:
        customers_saved = target_customers * rate
        revenue_saved = customers_saved * results['value_per_saved']
        net_roi = revenue_saved - results['total_budget']
        status = "✅ Profit" if net_roi > 0 else "❌ Loss"
        
        print(f"{rate*100:>5.0f}%{'':<6} "
              f"{customers_saved:>5.0f}{'':<5} "
              f"£{revenue_saved:>10,.0f}{'':<3} "
              f"£{results['total_budget']:>10,.0f}{'':<3} "
              f"£{net_roi:>10,.0f}{'':<3} "
              f"{status}")


if __name__ == "__main__":
    # Example: Save At All Costs segment
    results = calculate_roi(
        target_customers=178,
        avg_clv=6540,
        discount_rate=0.20,
        avg_order_value=95,
        expected_orders=3,
        campaign_cost=5
    )
    
    print("Campaign Economics:")
    print("="*80)
    print(f"Target Customers: {178}")
    print(f"Cost per Customer: £{results['cost_per_customer']:.2f}")
    print(f"Value per Saved: £{results['value_per_saved']:.2f}")
    print(f"Total Budget: £{results['total_budget']:,.2f}")
    print(f"\nBreak-Even Rate: {results['breakeven_rate']:.2%}")
    print(f"Customers Needed: {int(results['breakeven_customers'])}")
    
    print_roi_scenarios(results, 178)
