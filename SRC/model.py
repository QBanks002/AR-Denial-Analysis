
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
from matplotlib.patches import Patch

def run_analysis(filepath='data/ar_denials_cleaned.csv'):

    df = pd.read_csv(filepath, parse_dates=['DOS', 'Submitted Date', 'Worked Date', 'Follow up date'])

    # ── Revenue loss summary ──────────────────────────────────────────
    total_billed   = df['Billed Amount'].sum()
    total_balance  = df['Balance Amount'].sum()
    denied_df      = df[df['is_denied']]
    denied_billed  = denied_df['Billed Amount'].sum()
    denied_balance = denied_df['Balance Amount'].sum()
    denial_rate    = len(denied_df) / len(df)

    print(f"{'='*45}")
    print(f"  REVENUE LOSS SUMMARY")
    print(f"{'='*45}")
    print(f"  Total AR (billed):        ${total_billed:>10,.2f}")
    print(f"  Total AR (balance):       ${total_balance:>10,.2f}")
    print(f"  Total denied (billed):    ${denied_billed:>10,.2f}")
    print(f"  Total denied (balance):   ${denied_balance:>10,.2f}")
    print(f"  Denial rate:              {denial_rate:>10.1%}")
    print(f"  % of balance at risk:     {denied_balance/total_balance:>10.1%}")
    print(f"{'='*45}")

    # ── Denial by root cause ──────────────────────────────────────────
    fixable_codes = [
        'Incorrect Submission', 'Missing Modifiers', 'Invalid Procedure Code',
        'Provider Info Missing', 'Dx inconsistent with CPT', 'Duplicate Claim',
        'Claim not on file', 'Claim Error'
    ]
    denial_by_cause = (
        denied_df.groupby('Status Code')
        .agg(claim_count=('Balance Amount', 'count'),
             total_balance=('Balance Amount', 'sum'),
             avg_balance=('Balance Amount', 'mean'))
        .sort_values('total_balance', ascending=False)
        .round(2)
    )
    denial_by_cause['fixable'] = denial_by_cause.index.isin(fixable_codes)
    denial_by_cause['pct_of_denied'] = (denial_by_cause['total_balance'] / denied_balance * 100).round(1)

    # ── Denial by payer ───────────────────────────────────────────────
    payer_summary = (
        df.groupby('Insurance Name')
        .agg(total_claims=('Balance Amount', 'count'),
             total_balance=('Balance Amount', 'sum'),
             denied_claims=('is_denied', 'sum'),
             denied_balance=('Balance Amount', lambda x: x[df.loc[x.index, 'is_denied']].sum()))
        .round(2)
    )
    payer_summary['denial_rate']        = (payer_summary['denied_claims'] / payer_summary['total_claims'] * 100).round(1)
    payer_summary['pct_balance_denied'] = (payer_summary['denied_balance'] / payer_summary['total_balance'] * 100).round(1)
    payer_summary = payer_summary.sort_values('denied_balance', ascending=False)

    # ── Recovery scenarios ────────────────────────────────────────────
    fixable_balance = df[df['is_fixable']]['Balance Amount'].sum()
    scenarios = pd.DataFrame({
        'Scenario': ['Conservative (25% reduction)', 'Moderate (50% reduction)',
                     'Aggressive (75% reduction)', 'Full fixable recovery (100%)'],
        'Denial Reduction': [0.25, 0.50, 0.75, 1.0],
        'Revenue Recovered': [denied_balance * 0.25, denied_balance * 0.50,
                              denied_balance * 0.75, fixable_balance]
    })
    scenarios['% of Total AR Recovered'] = (scenarios['Revenue Recovered'] / total_balance * 100).round(2)
    scenarios['Revenue Recovered']        = scenarios['Revenue Recovered'].round(2)

    # ── Fixable claims worklist ───────────────────────────────────────
    fixable_claims = (
        df[df['is_fixable']]
        [['VisitID#', 'Patient Name', 'Insurance Name', 'Status Code',
          'Action Code', 'Billed Amount', 'Balance Amount', 'Aging Days', 'Follow up date']]
        .sort_values('Balance Amount', ascending=False)
        .reset_index(drop=True)
    )

    # ── Visualizations ────────────────────────────────────────────────
    os.makedirs('outputs', exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Claim Denial Revenue Impact Analysis', fontsize=15, y=1.01)

    cause_plot = denial_by_cause.sort_values('total_balance', ascending=True)
    colors = ['#e07b54' if f else '#6baed6' for f in cause_plot['fixable']]
    cause_plot['total_balance'].plot(kind='barh', ax=axes[0,0], color=colors)
    axes[0,0].set_title('Denied balance by root cause')
    axes[0,0].set_xlabel('Balance ($)')
    axes[0,0].xaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    axes[0,0].legend(handles=[Patch(color='#e07b54', label='Fixable'),
                               Patch(color='#6baed6', label='Non-fixable')])

    payer_plot = payer_summary.sort_values('denied_balance', ascending=True)
    payer_plot['denied_balance'].plot(kind='barh', ax=axes[0,1], color='steelblue')
    axes[0,1].set_title('Denied balance by payer')
    axes[0,1].set_xlabel('Balance ($)')
    axes[0,1].xaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))

    axes[1,0].bar(['25%', '50%', '75%', '100%\n(fixable only)'],
                  scenarios['Revenue Recovered'],
                  color=['#74c476', '#41ab5d', '#238b45', '#005a32'])
    axes[1,0].set_title('Projected revenue recovery by scenario')
    axes[1,0].set_ylabel('Revenue recovered ($)')
    axes[1,0].yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    for i, v in enumerate(scenarios['Revenue Recovered']):
        axes[1,0].text(i, v + 200, f'${v:,.0f}', ha='center', fontsize=9)

    status_totals = df.groupby('Status')['Balance Amount'].sum().sort_values(ascending=False)
    status_totals.plot(kind='bar', ax=axes[1,1], color='coral', edgecolor='white')
    axes[1,1].set_title('Total balance by claim status')
    axes[1,1].set_ylabel('Balance ($)')
    axes[1,1].yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    axes[1,1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig('outputs/denial_revenue_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── Export reports ────────────────────────────────────────────────
    denial_by_cause.to_csv('outputs/denial_by_cause.csv')
    scenarios.to_csv('outputs/recovery_scenarios.csv', index=False)
    fixable_claims.to_csv('outputs/fixable_claims_worklist.csv', index=False)
    payer_summary.to_csv('outputs/payer_denial_summary.csv')
    print("All reports saved to outputs/")

    return denial_by_cause, payer_summary, scenarios, fixable_claims

if __name__ == '__main__':
    run_analysis()
