# Security Policy

## Supported Versions

| Version | Supported |
|----------|------------|
| 0.1.x | ✅ Active |
| < 0.1 | ❌ No liability |

## Reporting a Vulnerability

Let's make AI Chess Battle secure for everyone.  
If you discover a security vulnerability, please let me know as soon as possible!

**Please do NOT open a public Issue** presenting the vulnerability if you need it fixed before disclosure.

**Email**: [Provide your security contact email here]  
**PGP**: [Optionally link to your public PGP key]

### Response Timeline

- **Acknowledgment**: Within 48 hours of report
- **Initial Assessment**: Within 5 business days
- **Public Disclosure**: Coordinated with reporter after fix deployment

## Security Design Principles

- **API keys are never persisted server-side:** Keys entered in the Streamlit sidebar remain in the user's session state only.
- **All provider API calls are user-facing:** The app proxies requests through the user's session — keys never touch a shared server.
- **Self-hosted deployments:** No telemetry, no data exfiltration. Run completely offline with local models (Ollama).

## Scope

- The Streamlit web application
- The headless benchmark runner
- Docker deployment configuration
- Development tooling (CI/CD pipelines, pre-commit hooks)

## Outside Scope

- Third-party provider APIs (OpenRouter, NVIDIA NIM, etc.) — report those to those providers
- Personal information stored by external providers (these apps are just gateways)