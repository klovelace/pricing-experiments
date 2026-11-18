# Decision memo: how deep should the discount be?

**To:** Product, Finance and Growth leads
**From:** Kraemer Lovelace
**Re:** Results of a four-arm randomized price test (50,083 users) and the next test to run

## Recommendation

1. **Ship the smallest discount that works: the 1:1 match.** Any discount lifted response by about 23%. The 2:1 and 3:1 discounts cost two and three times as much and bought no additional response (joint test p = 0.51).
2. **Target the discount instead of giving it to everyone.** The effect sits in identifiable segments. Under a cross-fitted targeting rule, offering the match to 70% of the list raised 22% more incremental revenue per user and cut discount spend by 35%.
3. **Run the next test below 1:1, not above it.** The demand curve is flat between $0.50 and $0.25, so the open question is where the floor is. Test 0.5:1 and 0.25:1 against 1:1, and keep a long-run holdout. Plan on about 160,000 users per arm.

## What we tested

Prior donors were randomized into a control letter or a matching-grant letter. The match lowers the donor's effective price of delivering $1 to the charity: $1.00 in control, then $0.50, $0.33 and $0.25 at 1:1, 2:1 and 3:1. Two presentation details were cross-randomized in the same mailing: the stated size of the match budget and the suggested-gift anchor. *Data: Karlan & List (2007), public domain. This memo is an independent reanalysis framed as a pricing decision.*

The test is clean. Arm sizes match the design (sample-ratio-mismatch test p = 1.00), and none of eight pre-period covariates differ between arms at p < 0.05.

## Results

| Metric | Control | Any match | Difference |
|---|---|---|---|
| Response rate | 1.79% | 2.20% | +0.42 pts (+23%), 95% CI [+0.16, +0.67], p = 0.001 |
| Revenue per user | $0.81 | $0.97 | +$0.15 (+19%), 95% CI [−$0.00, +$0.31], p = 0.055 |

- **Price elasticity of participation is −0.22** (bootstrap 95% CI −0.36 to −0.08). Demand responds to the first price cut and then flattens.
- **Deeper discounts are wasted.** Incremental revenue per matching dollar is $0.13 at 1:1, $0.10 at 2:1 and $0.04 at 3:1.
- **Who responds:** red-state users gained +$0.38 per user (Holm-adjusted p = 0.01), against +$0.00 in blue states (interaction p = 0.02). Small prior donors gained +$0.22 (p = 0.001).
- **The anchor didn't matter.** Suggesting 1.25× or 1.5× the prior gift changed neither response nor gift size.
- **Revenue is heavy-tailed.** The revenue result holds under 99th and 99.5th percentile winsorization and under a bootstrap. It sits at the edge of significance, so conversion is the metric to power on.

## Risks and what I'd watch

- **Short-run only.** These outcomes come from a single mailing. A discount can pull revenue forward or train users to wait for the next offer. Keep a holdout and track 6- and 12-month revenue.
- **Segment effects were tested on six pre-specified cuts with Holm correction,** but the targeting rule should be validated in the next test before it's rolled out.
- **Pre-period covariates cut variance by less than 2% here,** because past giving barely predicts whether someone gives this time. In a product with stable usage (trading volume, for example), CUPED on the pre-period metric typically does much better. Budget sample size without counting on it.

## Next test

| | |
|---|---|
| Arms | 1:1 (current best) · 0.5:1 · 0.25:1 · no discount · 10% long-run holdout |
| Primary metric | Conversion, sized at about 160K users per arm for a 10% revenue lift (≈ 40K per arm for 20%) |
| Guardrails | 6-month repeat rate, discount spend per incremental dollar |
| Decision rule | Pick the cheapest arm whose conversion lift is within 10% of 1:1 at the 90% upper bound |
