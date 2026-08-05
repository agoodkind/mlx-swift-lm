# Repair upstream sync pull request authentication

## Current behavior

The daily upstream sync fetches, merges, and pushes `sync/upstream-latest`.
Pull request creation then fails because it uses `github.token` while repository
settings prohibit that token from creating pull requests.

## Design

Use the existing `SWIFTLM_PR_TOKEN` secret for pull request lookup and creation.
Keep branch creation, merge behavior, scheduling, and review requirements unchanged.

## Verification

The current failed scheduled run is the baseline reproduction. After merging this
change, dispatch the workflow and require a successful run plus an open pull request
from `sync/upstream-latest` into `main`.
