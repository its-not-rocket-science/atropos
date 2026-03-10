#!/usr/bin/env python3
"""Crawl HuggingFace Hub and GitHub for LLM models and pruning implementations.

This script discovers:
1. Available LLMs on HuggingFace Hub by parameter range
2. Pruning/optimization implementations on GitHub
3. Generates compatibility reports for Atropos testing

Usage:
    python model-discovery-crawler.py --source huggingface --max-params 3B
    python model-discovery-crawler.py --source github --query "llm pruning"
    python model-discovery-crawler.py --full-discovery --output models.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


@dataclass
class DiscoveredModel:
    """Information about a discovered model."""

    name: str
    source: str  # "huggingface" or "github"
    params_b: float | None = None
    description: str = ""
    url: str = ""
    tags: list[str] = field(default_factory=list)
    downloads: int | None = None
    likes: int | None = None
    last_updated: str = ""
    license: str = ""
    task: str = ""
    atropos_compatible: bool = False
    estimated_memory_gb: float | None = None
    pruning_applicable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PruningImplementation:
    """Information about a pruning implementation."""

    name: str
    source: str  # "github" or "pypi"
    url: str
    description: str = ""
    stars: int | None = None
    language: str = ""
    last_updated: str = ""
    # integrated, planned, not_integrated
    atropos_integration_status: str = "not_integrated"


def estimate_params_from_name(model_id: str) -> float | None:
    """Estimate parameter count from model name."""
    name_lower = model_id.lower()

    # Try to extract from name patterns like "125m", "1.1b", "7b"
    patterns = [
        (r"(\d+\.?\d*)b", 1e9),  # 1.1b -> 1.1B
        (r"(\d+\.?\d*)m", 1e6),  # 125m -> 125M
    ]

    for pattern, multiplier in patterns:
        match = re.search(pattern, name_lower)
        if match:
            try:
                return float(match.group(1)) * multiplier / 1e9
            except ValueError:
                continue

    # Known model families
    known = {
        "gpt2": 0.124,
        "gpt2-medium": 0.355,
        "gpt2-large": 0.774,
        "gpt2-xl": 1.5,
        "gpt-2": 0.124,
        "gpt-3": 175.0,
        "gpt-4": 1000.0,  # Estimated
        "llama-7b": 7.0,
        "llama-13b": 13.0,
        "llama-30b": 30.0,
        "llama-65b": 65.0,
        "llama-2-7b": 7.0,
        "llama-2-13b": 13.0,
        "llama-2-70b": 70.0,
        "llama-3-8b": 8.0,
        "llama-3-70b": 70.0,
        "pythia-70m": 0.07,
        "pythia-160m": 0.16,
        "pythia-410m": 0.41,
        "pythia-1b": 1.0,
        "pythia-1.4b": 1.4,
        "pythia-2.8b": 2.8,
        "pythia-6.9b": 6.9,
        "pythia-12b": 12.0,
        "opt-125m": 0.125,
        "opt-350m": 0.35,
        "opt-1.3b": 1.3,
        "opt-2.7b": 2.7,
        "opt-6.7b": 6.7,
        "opt-13b": 13.0,
        "bloom-560m": 0.56,
        "bloom-1b7": 1.7,
        "bloom-3b": 3.0,
        "bloom-7b1": 7.1,
        "bloom": 176.0,
        "mistral-7b": 7.0,
        "mixtral-8x7b": 47.0,  # MoE
        "mixtral-8x22b": 141.0,  # MoE
        "falcon-7b": 7.0,
        "falcon-40b": 40.0,
        "falcon-180b": 180.0,
        "phi-2": 2.7,
        "phi-3": 3.8,
    }

    for key, value in known.items():
        if key in name_lower:
            return value

    return None


def estimate_memory_gb(params_b: float | None) -> float | None:
    """Estimate memory usage from parameter count."""
    if params_b is None:
        return None
    # Rough estimate: 2 bytes per parameter (FP16) + overhead
    return params_b * 2 * 1.2


def search_huggingface(
    max_params: float | None = None,
    min_params: float | None = None,
    task: str = "text-generation",
    library: str = "transformers",
    limit: int = 50,
) -> list[DiscoveredModel]:
    """Search HuggingFace Hub for models.

    Args:
        max_params: Maximum parameter count in billions
        min_params: Minimum parameter count in billions
        task: Model task (e.g., text-generation)
        library: ML library (e.g., transformers)
        limit: Maximum number of results

    Returns:
        List of discovered models
    """

    print(f"Searching HuggingFace Hub for {task} models...")

    api = HfApi()
    models = api.list_models(
        task=task,
        library=library,
        sort="downloads",
        direction=-1,
        limit=limit * 2,  # Fetch more to filter
    )

    results = []
    for model in models:
        model_id = model.modelId

        # Skip if private or gated
        if model.private:
            continue

        # Estimate parameters
        params_b = estimate_params_from_name(model_id)

        # Filter by size
        if max_params and params_b and params_b > max_params:
            continue
        if min_params and params_b and params_b < min_params:
            continue

        # Check tags for more info
        tags = model.tags or []

        # Check if it's a generative model suitable for pruning
        applicable = any(
            t in tags
            for t in ["text-generation", "causal-lm", "llm", "transformers"]
        )

        # Estimate memory
        memory_gb = estimate_memory_gb(params_b)

        # Check compatibility (small enough to test reasonably)
        compatible = params_b is not None and params_b <= 10

        results.append(
            DiscoveredModel(
                name=model_id,
                source="huggingface",
                params_b=params_b,
                description=model.cardData.get(
                    "description", "") if model.cardData else "",
                url=f"https://huggingface.co/{model_id}",
                tags=list(tags),
                downloads=model.downloads,
                likes=model.likes,
                last_updated=model.lastModified.strftime(
                    "%Y-%m-%d") if model.lastModified else "",
                license=model.cardData.get(
                    "license", "") if model.cardData else "",
                task=task,
                atropos_compatible=compatible,
                estimated_memory_gb=memory_gb,
                pruning_applicable=applicable,
            )
        )

        if len(results) >= limit:
            break

    print(f"Found {len(results)} models")
    return results


def search_github(
    query: str,
    language: str = "python",
    min_stars: int = 10,
    limit: int = 30,
) -> list[PruningImplementation]:
    """Search GitHub for pruning implementations.

    Args:
        query: Search query
        language: Programming language filter
        min_stars: Minimum star count
        limit: Maximum results

    Returns:
        List of pruning implementations
    """

    print(f"Searching GitHub for: {query}...")

    # Build search query
    search_terms = f"{query} language:{language} stars:>{min_stars}"
    encoded_query = urllib.parse.quote(search_terms)

    url = f"https://api.github.com/search/repositories?q={encoded_query}&sort=stars&order=desc&per_page={limit}"

    try:
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/vnd.github.v3+json")
        req.add_header("User-Agent", "atropos-model-discovery")

        # Add GitHub token if available
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token:
            req.add_header("Authorization", f"token {github_token}")

        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())

    except Exception as e:
        print(f"Error searching GitHub: {e}")
        return []

    results = []
    for repo in data.get("items", []):
        # Check if already integrated
        integrated_repos = [
            "llm-pruner",
            "wanda",
            "sparsegpt",
        ]
        status = "integrated" if any(
            ir in repo["full_name"].lower() for ir in integrated_repos
        ) else "not_integrated"

        results.append(
            PruningImplementation(
                name=repo["full_name"],
                source="github",
                url=repo["html_url"],
                description=repo.get("description", ""),
                stars=repo.get("stargazers_count"),
                language=repo.get("language", ""),
                last_updated=repo.get("updated_at", "").split("T")[0],
                atropos_integration_status=status,
            )
        )

    print(f"Found {len(results)} repositories")
    return results


def search_pypi_pruning_tools() -> list[PruningImplementation]:
    """Search PyPI for pruning-related packages."""

    print("Searching PyPI for pruning packages...")

    # queries = ["pruning", "sparse", "quantization", "model-optimization"]

    # for query in queries:
    #     url = f"https://pypi.org/search/?q={query}"
    #     # Note: PyPI doesn't have a simple JSON API for search
    #     # This is a placeholder for actual implementation

    # Known packages
    known_packages = [
        PruningImplementation(
            name="torch-pruning",
            source="pypi",
            url="https://pypi.org/project/torch-pruning/",
            description="Structural pruning for neural networks",
            atropos_integration_status="not_integrated",
        ),
        PruningImplementation(
            name="neural-compressor",
            source="pypi",
            url="https://pypi.org/project/neural-compressor/",
            description="Intel's model compression toolkit",
            atropos_integration_status="not_integrated",
        ),
    ]

    return known_packages


def generate_report(
    models: list[DiscoveredModel],
    implementations: list[PruningImplementation],
    output_path: Path | None = None,
) -> str:
    """Generate a markdown report."""

    lines = [
        "# Atropos Model Discovery Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        f"- Models discovered: {len(models)}",
        f"- Pruning implementations: {len(implementations)}",
        f"- Atropos-compatible models: {sum(1 for m in models if m.atropos_compatible)}",
        "",
        "## Recommended Models for Atropos Testing",
        "",
    ]

    # Filter compatible models
    compatible = [m for m in models if m.atropos_compatible]

    if compatible:
        lines.append("| Model | Params | Memory | Downloads | Source |")
        lines.append("|-------|--------|--------|-----------|--------|")

        for m in sorted(compatible, key=lambda x: x.params_b or 0):
            params_str = f"{m.params_b:.2f}B" if m.params_b else "?"
            mem_str = f"{m.estimated_memory_gb:.1f}GB" if m.estimated_memory_gb else "?"
            dl_str = f"{m.downloads:,}" if m.downloads else "?"
            lines.append(
                f"| [{m.name}]({m.url}) | {params_str} | {mem_str} | {dl_str} | HF |")

    lines.extend(["", "## All Discovered Models", ""])

    for size_range, label in [
        ("<= 1", "Small (≤1B)"),
        ("1-7", "Medium (1-7B)"),
        (">7", "Large (>7B)"),
    ]:
        lines.append(f"### {label}")
        lines.append("")

        if size_range == "<= 1":
            filtered = [m for m in models if m.params_b and m.params_b <= 1]
        elif size_range == "1-7":
            filtered = [
                m for m in models if m.params_b and 1 < m.params_b <= 7]
        else:
            filtered = [m for m in models if m.params_b and m.params_b > 7]

        if filtered:
            lines.append("| Model | Params | Task | Pruning |")
            lines.append("|-------|--------|------|---------|")
            for m in sorted(filtered, key=lambda x: x.downloads or 0, reverse=True)[:10]:
                params = f"{m.params_b:.2f}B" if m.params_b else "?"
                pruning = "✅" if m.pruning_applicable else "❌"
                lines.append(
                    f"| [{m.name}]({m.url}) | {params} | {m.task} | {pruning} |")
        else:
            lines.append("_No models found in this range_")

        lines.append("")

    lines.extend(["", "## Pruning Implementations", ""])

    if implementations:
        lines.append("| Name | Stars | Language | Integration |")
        lines.append("|------|-------|----------|-------------|")

        for impl in sorted(implementations, key=lambda x: x.stars or 0, reverse=True):
            stars = f"⭐ {impl.stars:,}" if impl.stars else "?"
            status = {
                "integrated": "✅",
                "planned": "📝",
                "not_integrated": "⬜",
            }.get(impl.atropos_integration_status, "?")
            lines.append(
                f"| [{impl.name}]({impl.url}) | {stars} | {impl.language} | {status} |"
            )

    lines.extend(["", "## Atropos Scenarios", ""])
    lines.append("You can use these models with Atropos:")
    lines.append("")
    lines.append("```bash")

    for m in compatible[:5]:
        if m.params_b:
            lines.append(f"# {m.name} ({m.params_b:.2f}B params)")
            lines.append(f"atropos validate edge-coder --model {m.name}")
            lines.append("")

    lines.append("```")

    content = "\n".join(lines)

    if output_path:
        output_path.write_text(content, encoding="utf-8")
        print(f"\nReport saved to: {output_path}")

    return content


def export_to_yaml(models: list[DiscoveredModel], output_path: Path) -> None:
    """Export discovered models to YAML for Atropos."""
    try:
        import yaml
    except ImportError:
        print("Error: pyyaml not installed. Run: pip install pyyaml")
        sys.exit(1)

    data = {
        "models": [
            {
                "name": m.name,
                "params_b": m.params_b,
                "memory_gb": m.estimated_memory_gb,
                "source": m.source,
                "url": m.url,
                "atropos_compatible": m.atropos_compatible,
            }
            for m in models
            if m.atropos_compatible
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"Exported {len(data['models'])} models to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Discover LLM models and pruning implementations"
    )
    parser.add_argument(
        "--source",
        choices=["huggingface", "github", "all"],
        default="all",
        help="Source to search",
    )
    parser.add_argument(
        "--max-params",
        type=float,
        default=10.0,
        help="Maximum parameter count in billions (default: 10)",
    )
    parser.add_argument(
        "--min-params",
        type=float,
        default=None,
        help="Minimum parameter count in billions",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of results (default: 50)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="llm pruning",
        help="GitHub search query",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for report",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "yaml"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--full-discovery",
        action="store_true",
        help="Run comprehensive discovery (slow)",
    )

    args = parser.parse_args()

    if args.full_discovery:
        args.source = "all"
        args.limit = 200

    models: list[DiscoveredModel] = []
    implementations: list[PruningImplementation] = []

    # Search HuggingFace
    if args.source in ("huggingface", "all"):
        hf_models = search_huggingface(
            max_params=args.max_params,
            min_params=args.min_params,
            limit=args.limit,
        )
        models.extend(hf_models)

    # Search GitHub
    if args.source in ("github", "all"):
        gh_repos = search_github(
            query=args.query,
            limit=args.limit,
        )
        implementations.extend(gh_repos)

        # Also search PyPI
        pypi_packages = search_pypi_pruning_tools()
        implementations.extend(pypi_packages)

    # Generate output
    if args.format == "json":
        output = {
            "models": [m.to_dict() for m in models],
            "implementations": [asdict(i) for i in implementations],
        }
        content = json.dumps(output, indent=2)
        if args.output:
            args.output.write_text(content)
        else:
            print(content)

    elif args.format == "yaml":
        if args.output:
            export_to_yaml(models, args.output)
        else:
            print("YAML format requires --output")
            sys.exit(1)

    else:  # markdown
        content = generate_report(models, implementations, args.output)
        if not args.output:
            print(content)


if __name__ == "__main__":
    main()
