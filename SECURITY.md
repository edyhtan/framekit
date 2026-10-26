# Security policy

## Reporting a vulnerability

Please report vulnerabilities **privately**, not in a public issue or pull request:

1. Open the repository's **Security** tab on GitHub.
2. Click **Report a vulnerability** (GitHub private vulnerability reporting).
3. Describe the problem, the affected version or commit, and how to reproduce it.

Only the maintainer and you can see the report. If the button isn't shown, open a public issue that says only that you have a security report (no details), and the maintainer will set up a private channel.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |
| older | no |

Fixes go into the latest release on the supported line, published as a new patch tag (`v0.1.N`).

## Scope

In scope: the library code in this repository (`framekit.contracts`, `framekit.models`, `framekit.frames`, `framekit.extraction` and the `framekit` command), its packaging, and its CI workflows and dependency pins. Examples:

- a way for a provider key to leak, for instance into logs, error messages or response-cache files;
- unsafe handling of model responses or cache records (code execution, path traversal, deserialization);
- a way for an input file (a lens spec, search space, corpus, plan or gold set) to make framekit run a command, read a file it shouldn't, or send one to a model provider. Spec files are read with YAML's safe loader, a lens's prompt files must stay inside its folder, and ffmpeg runs with argument lists, never through a shell;
- a workflow that could run untrusted code with write access, or a dependency or action pin that can be swapped.

Context that limits what framekit can expose:

- **No footage:** the repository contains no video, frames, thumbnails, ripping tools or downloaders, and test videos are generated test patterns. Extraction reads the video files you point it at, writes analysis copies only to its work folder, and sends frames only to the models your plan names (for a `byo_media` corpus, only once the plan says it may: `policy.governance.model_inputs`). The 0.1.x line (contracts and model access) doesn't read video files at all.
- **Your own code runs only when you ask:** `--sources MODULE:NAME` and `--scorer MODULE:NAME` import Python code you name, by design.
- **The only secrets it reads are provider API keys** (and optional base-URL overrides), from the environment, by the names in the provider configs: the built-in catalog's (`HIVE_API_KEY`, `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`), or the variable a provider registered by an application names. It has no other environment-bound settings.
- **Known gaps**, to be closed later ([docs/adapters.md](docs/adapters.md), "Security"):
  - a provider's key is sent to that provider's base URL, and the official SDKs also send headers taken from their own environment variables (`OPENAI_CUSTOM_HEADERS`, `ANTHROPIC_CUSTOM_HEADERS`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`) to whatever host a provider points at, including custom ones. framekit doesn't strip them yet;
  - nothing binds a key to a host yet: a provider config may name any key variable, including the built-in providers', and its key then goes to that config's base URL. Review provider configs you didn't write;
  - framekit doesn't check the URL scheme: use `https` for every host that isn't on the same machine, since an `http://` base URL sends the key in clear text.

  A report showing another way for a key to reach a host it wasn't configured for is in scope.

Out of scope: vulnerabilities in the model providers' services or in their official SDKs (report those upstream), and in applications that use framekit (report those to their projects).

## What to expect

framekit is a personal, non-commercial project maintained in spare time, so there's no guaranteed timeline. The aim is to acknowledge a report within about a week, agree on a fix and a disclosure date with you, and credit you in the advisory if you'd like.

How the project protects its own supply chain (pinned actions, release cooldown, dependency audit, CodeQL, zizmor) is described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §2.2.
