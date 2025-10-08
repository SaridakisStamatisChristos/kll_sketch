# Why visible CI evidence matters

Our repository already runs GitHub Actions and uploads coverage to Codecov. Those badges are useful because they point to the
latest pipeline results, but they do not replace in-repo evidence of how the project is validated. Having local documentation of
our automation brings several benefits:

1. **Discoverability for contributors.** Newcomers who clone the repository often work offline or from mirrors where external
   dashboards are unavailable. A local reference helps them understand how to reproduce the checks without chasing links or
   needing network access.
2. **Reproducible workflows.** When the CI definition files live only in external systems, it is impossible to guarantee that the
   codebase can be validated independently. Keeping the workflow files, scripts, and configuration versioned in the repo makes
   the project self-contained and allows reviewers to audit the exact commands that gate merges.
3. **Traceability over time.** Badges surface only the most recent status. If the external pipelines change behaviour—or the
   external service goes away—we lose the historical context. Documenting the CI entry points (e.g. workflow YAML, coverage
   commands) in the repository ensures the validation story survives platform migrations.
4. **Security and trust.** Reviewers and users can inspect in-repo CI definitions to verify that the advertised checks are the
   ones actually executed. This transparency is a core expectation for open-source projects; otherwise, the project depends on a
   third-party configuration that might drift without review.
5. **Onboarding automation.** A repository-local description of the CI makes it trivial to run the same checks locally (for
   example via `make` targets or scripts). This keeps feedback fast and reduces breakages that only surface after pushing to the
   remote pipelines.

In short: keeping CI evidence—workflow YAML, scripts, and documentation—inside the repository is not redundant with using
GitHub Actions or Codecov. It is the foundation that makes those services trustworthy, reproducible, and accessible to every
contributor.
