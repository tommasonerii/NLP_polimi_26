"""Lightweight tool-based answer strategies for PoliMillionaire.

The functions in this module are intentionally simple and fast. They are not a
complete math solver: they implement deterministic tools for question families
that are cheap to recognize, then fall back to another strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm, isqrt
import re
from typing import Callable, Iterable, Optional


@dataclass
class ToolDecision:
    option_id: int
    strategy: str
    confidence: float
    explanation: str


def _option_items(question) -> list[tuple[int, str]]:
    return [(opt.id, str(opt.text)) for opt in question.options]


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _parse_number(text: str) -> Optional[Fraction]:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    return Fraction(match.group(0))


def _find_option_by_number(question, value: Fraction, tolerance: float = 1e-9) -> Optional[int]:
    for option_id, option_text in _option_items(question):
        parsed = _parse_number(option_text)
        if parsed is None:
            continue
        if abs(float(parsed - value)) <= tolerance:
            return option_id
    return None


def _find_option_containing(question, patterns: Iterable[str]) -> Optional[int]:
    lowered_patterns = [_normalize(pattern) for pattern in patterns]
    for option_id, option_text in _option_items(question):
        normalized = _normalize(option_text)
        if any(pattern in normalized for pattern in lowered_patterns):
            return option_id
    return None


def tool_lcm_gcd_options(question) -> Optional[ToolDecision]:
    """Solve option-checkable questions involving lcm(a, b) / gcd(a, b)."""
    text = str(question.text)
    low = _normalize(text)
    if "least common multiple" not in low or "greatest common divisor" not in low:
        return None

    target_match = re.search(r"result\s+is\s+(\d+)", low)
    known_match = re.search(r"one integer is\s+(\d+)", low)
    if not target_match or not known_match:
        return None

    target = int(target_match.group(1))
    known = int(known_match.group(1))

    valid = []
    for option_id, option_text in _option_items(question):
        candidate = _parse_number(option_text)
        if candidate is None or candidate.denominator != 1 or candidate <= 0:
            continue
        other = int(candidate)
        ratio = lcm(known, other) // gcd(known, other)
        if ratio == target:
            valid.append((other, option_id))

    if not valid:
        return None

    other, option_id = min(valid)
    return ToolDecision(
        option_id=option_id,
        strategy="tool_lcm_gcd_options",
        confidence=0.95,
        explanation=f"Checked answer options: lcm({known}, x) / gcd({known}, x) = {target}; smallest valid x is {other}.",
    )


def tool_correlation(question) -> Optional[ToolDecision]:
    """Handle basic correlation invariance and r-squared interpretation."""
    text = str(question.text)
    low = _normalize(text)
    options_low = _normalize(" ".join(option_text for _, option_text in _option_items(question)))
    all_text_low = f"{low} {options_low}"
    if "correlation" not in low:
        return None

    r_match = re.search(r"r\s*=\s*(-?\d+(?:\.\d+)?)", low)
    if not r_match:
        r_match = re.search(r"correlation\s+(?:of|is|between)?\s*(-?\d+(?:\.\d+)?)", low)
    if r_match:
        r = Fraction(r_match.group(1))
    else:
        candidates = [Fraction(x) for x in re.findall(r"-?\d+\.\d+", low)]
        correlations = [x for x in candidates if -1 <= x <= 1]
        if not correlations:
            return None
        r = correlations[0]

    if "converted" in low or "centimeters" in low or "cm" in low:
        option_id = _find_option_by_number(question, r)
        if option_id is not None:
            return ToolDecision(
                option_id=option_id,
                strategy="tool_correlation_invariance",
                confidence=0.95,
                explanation="Correlation is unchanged by multiplying one variable by a positive constant.",
            )

    if "variation" in all_text_low or "explained" in all_text_low:
        percent = float(r * r * 100)
        option_id = _find_option_containing(question, [f"{percent:g}%", f"{percent:.0f}%"])
        if option_id is not None:
            return ToolDecision(
                option_id=option_id,
                strategy="tool_correlation_r_squared",
                confidence=0.9,
                explanation=f"Explained variation is r^2 = {float(r):.3g}^2 = {percent:g}%.",
            )

    return None


def _is_squarefree(n: int) -> bool:
    if n <= 1:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % (d * d) == 0:
            return False
    return True


def tool_simple_field_extension(question) -> Optional[ToolDecision]:
    """Handle simple Q(sqrt(a)+sqrt(b)) over Q examples."""
    text = str(question.text)
    low = _normalize(text)
    if "field extension" not in low or "sqrt" not in low:
        return None

    radicands = [int(x) for x in re.findall(r"sqrt\((\d+)\)", low)]
    unique = sorted(set(radicands))
    if len(unique) < 2:
        return None

    if all(_is_squarefree(n) for n in unique[:2]):
        degree = Fraction(4)
        option_id = _find_option_by_number(question, degree)
        if option_id is not None:
            return ToolDecision(
                option_id=option_id,
                strategy="tool_simple_field_extension",
                confidence=0.75,
                explanation=f"For distinct squarefree radicands {unique[:2]}, Q(sqrt(a)+sqrt(b)) typically has degree 4.",
            )

    return None


def tool_direct_product_quotient_order(question) -> Optional[ToolDecision]:
    """Handle order of (Z_m x Z_n)/(<1, 1>) when visible in the text."""
    text = str(question.text)
    low = _normalize(text)
    if "factor group" not in low and "quotient" not in low:
        return None

    match = re.search(r"z[_\s]*(\d+)\s*x\s*z[_\s]*(\d+).+<\s*1\s*,\s*1\s*>", low)
    if not match:
        return None

    m, n = int(match.group(1)), int(match.group(2))
    group_order = m * n
    subgroup_order = lcm(m, n)
    quotient_order = Fraction(group_order, subgroup_order)
    option_id = _find_option_by_number(question, quotient_order)
    if option_id is None:
        return None

    return ToolDecision(
        option_id=option_id,
        strategy="tool_direct_product_quotient_order",
        confidence=0.8,
        explanation=f"|Z_{m} x Z_{n}|={group_order}; order of <(1,1)> is lcm({m},{n})={subgroup_order}; quotient order={quotient_order}.",
    )


def first_option_fallback(question) -> ToolDecision:
    option_id = question.options[0].id
    return ToolDecision(
        option_id=option_id,
        strategy="fallback_first_option",
        confidence=0.2,
        explanation="No deterministic tool matched; using first-option fallback.",
    )


DEFAULT_TOOLS: tuple[Callable, ...] = (
    tool_lcm_gcd_options,
    tool_correlation,
    tool_simple_field_extension,
    tool_direct_product_quotient_order,
)


def choose_with_agentic_tools(question, fallback: Callable = first_option_fallback) -> ToolDecision:
    """Try deterministic tools in order, then use fallback."""
    for tool in DEFAULT_TOOLS:
        decision = tool(question)
        if decision is not None:
            return decision
    return fallback(question)
