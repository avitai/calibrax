# calibrax.ci

CI pipeline tools for regression gating and automated bisection.

## CI Guard

`CIGuard` compares the latest run against the stored baseline and returns a
`GuardResult` indicating pass/fail. Use with the CLI `calibrax check` command
for CI exit-code signaling.

::: calibrax.ci.guard

## Bisection Engine

`BisectionEngine` binary-searches git history to find the commit that introduced
a regression. Restores the original HEAD after completion.

::: calibrax.ci.bisection
