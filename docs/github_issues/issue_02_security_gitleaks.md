TITLE: [security] Add gitleaks secret-scan to CI (fixed)

**Finding**
- No secret scanning in CI. A Kaggle API token was pasted in chat during setup
  (since rotated by the human — confirmed 2026-10-08); `~/.kaggle/` lives
  outside the repo.

**Fix applied**
- `gitleaks/gitleaks-action@v2` step added to `.github/workflows/ci.yml`.
- Evidence: CI run #30 job step "Gitleaks scan (secrets)" = success (2026-10-08) [V].

**Residual**
- Keep the step pinned; treat any future chat-pasted credential as rotate-first.
