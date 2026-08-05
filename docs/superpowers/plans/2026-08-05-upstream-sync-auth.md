# Upstream Sync Pull Request Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create upstream-sync pull requests with the repository's existing pull-request token.

**Architecture:** Keep the scheduled sync, branch update, merge, and pull-request API calls unchanged. Set `GH_TOKEN` only in the Create Pull Request to Main step to `secrets.SWIFTLM_PR_TOKEN`, so both lookup and creation authenticate with that token.

**Tech Stack:** GitHub Actions YAML and GitHub CLI.

## Global Constraints

- Change only the Create Pull Request to Main step in `.github/workflows/upstream-sync.yml`.
- Never expose the value of `SWIFTLM_PR_TOKEN`.
- Preserve branch creation, merge, schedule, and review requirements.

---

### Task 1: Authenticate pull request operations

**Files:**
- Create: `docs/superpowers/plans/2026-08-05-upstream-sync-auth.md`
- Modify: `.github/workflows/upstream-sync.yml`

**Interfaces:**
- Consumes: Repository secret `SWIFTLM_PR_TOKEN`.
- Produces: `GH_TOKEN` for the existing `gh api` pull-request lookup and creation calls.

- [ ] Set `GH_TOKEN` in Create Pull Request to Main to `${{ secrets.SWIFTLM_PR_TOKEN }}`.
- [ ] Keep the step condition, API endpoints, request fields, and all other workflow steps unchanged.
- [ ] Run `actionlint .github/workflows/upstream-sync.yml` and require exit status 0.
- [ ] After merge, dispatch the workflow and require a successful run that opens or finds `sync/upstream-latest` targeting `main`.
- [ ] Commit with `Use SWIFTLM_PR_TOKEN for upstream sync pull requests`.
