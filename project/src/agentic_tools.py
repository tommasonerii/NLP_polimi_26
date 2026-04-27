"""SymPy-based tool strategies for PoliMillionaire.

This module implements a small agentic layer: each tool recognizes a cheap,
specific question family, computes/checks the answer with SymPy, and returns an
option id plus an explanation. It is deliberately conservative; unknown
questions should fall back to retrieval or an LLM rather than forcing a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable, Optional

import sympy as sp


@dataclass
class ToolDecision:
    option_id: int
    strategy: str
    confidence: float
    explanation: str


def _option_items(question) -> list[tuple[int, str]]:
    return [(opt.id, str(opt.text)) for opt in question.options]


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def _parse_number(text: str) -> Optional[sp.Rational | sp.Float | sp.Integer]:
    """Parse the first numeric value in option text into a SymPy number."""
    match = re.search(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    if not match:
        return None
    return sp.Rational(match.group(0))


def _numeric_equal(a, b, tolerance: float = 1e-9) -> bool:
    return abs(float(sp.N(a - b))) <= tolerance


def _find_option_by_number(question, value, tolerance: float = 1e-9) -> Optional[int]:
    value = sp.N(value) if not getattr(value, "is_Rational", False) else value
    for option_id, option_text in _option_items(question):
        parsed = _parse_number(option_text)
        if parsed is None:
            continue
        if _numeric_equal(parsed, value, tolerance=tolerance):
            return option_id
    return None


def _find_option_containing(question, patterns: Iterable[str]) -> Optional[int]:
    lowered_patterns = [_normalize(pattern) for pattern in patterns]
    for option_id, option_text in _option_items(question):
        normalized = _normalize(option_text)
        if any(pattern in normalized for pattern in lowered_patterns):
            return option_id
    return None


def _integer_options(question) -> list[tuple[int, int]]:
    values = []
    for option_id, option_text in _option_items(question):
        parsed = _parse_number(option_text)
        if parsed is not None and parsed.is_integer:
            values.append((option_id, int(parsed)))
    return values


def _is_squarefree(n: int) -> bool:
    if n <= 1:
        return False
    return all(exp == 1 for exp in sp.factorint(n).values())


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

    target = sp.Integer(target_match.group(1))
    known = sp.Integer(known_match.group(1))

    valid: list[tuple[int, int]] = []
    for option_id, other_int in _integer_options(question):
        if other_int <= 0:
            continue
        other = sp.Integer(other_int)
        ratio = sp.ilcm(known, other) / sp.igcd(known, other)
        if sp.simplify(ratio - target) == 0:
            valid.append((other_int, option_id))

    if not valid:
        return None

    other, option_id = min(valid)
    return ToolDecision(
        option_id=option_id,
        strategy="tool_lcm_gcd_options",
        confidence=0.95,
        explanation=f"Checked options with SymPy: lcm({known}, x) / gcd({known}, x) = {target}; smallest valid x is {other}.",
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
        r = sp.Rational(r_match.group(1))
    else:
        candidates = [sp.Rational(x) for x in re.findall(r"-?\d+\.\d+", low)]
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
        percent = sp.simplify(r**2 * 100)
        percent_float = float(sp.N(percent))
        option_id = _find_option_containing(question, [f"{percent_float:g}%", f"{percent_float:.0f}%"])
        if option_id is not None:
            return ToolDecision(
                option_id=option_id,
                strategy="tool_correlation_r_squared",
                confidence=0.9,
                explanation=f"Explained variation is r^2 = {float(sp.N(r)):.3g}^2 = {percent_float:g}%.",
            )

    return None


def tool_simple_field_extension(question) -> Optional[ToolDecision]:
    """Handle simple Q(sqrt(a)+sqrt(b)) over Q examples."""
    text = str(question.text)
    low = _normalize(text)
    if "field extension" not in low or "sqrt" not in low:
        return None

    radicands = sorted({int(x) for x in re.findall(r"sqrt\((\d+)\)", low)})
    if len(radicands) < 2:
        return None

    a, b = radicands[:2]
    if not _is_squarefree(a) or not _is_squarefree(b):
        return None

    # For distinct squarefree a,b, sqrt(a)+sqrt(b) generates a biquadratic
    # extension in the typical quiz case.
    degree = sp.Integer(4)
    option_id = _find_option_by_number(question, degree)
    if option_id is None:
        return None

    alpha = sp.sqrt(a) + sp.sqrt(b)
    minpoly = sp.minpoly(alpha)
    return ToolDecision(
        option_id=option_id,
        strategy="tool_simple_field_extension",
        confidence=0.85,
        explanation=f"SymPy minpoly({alpha}) = {minpoly}, degree {sp.degree(minpoly)}.",
    )


def tool_direct_product_quotient_order(question) -> Optional[ToolDecision]:
    """Handle order of (Z_m x Z_n)/(<1, 1>) when visible in the text."""
    text = str(question.text)
    low = _normalize(text)
    if "factor group" not in low and "quotient" not in low:
        return None

    match = re.search(r"z[_\s]*(\d+)\s*x\s*z[_\s]*(\d+).+<\s*1\s*,\s*1\s*>", low)
    if not match:
        return None

    m, n = sp.Integer(match.group(1)), sp.Integer(match.group(2))
    group_order = m * n
    subgroup_order = sp.ilcm(m, n)
    quotient_order = sp.simplify(group_order / subgroup_order)
    option_id = _find_option_by_number(question, quotient_order)
    if option_id is None:
        return None

    return ToolDecision(
        option_id=option_id,
        strategy="tool_direct_product_quotient_order",
        confidence=0.8,
        explanation=f"|Z_{m} x Z_{n}|={group_order}; order of <(1,1)> is lcm({m},{n})={subgroup_order}; quotient order={quotient_order}.",
    )


def tool_basic_equation_options(question) -> Optional[ToolDecision]:
    """Solve very explicit one-variable equations when the answer is numeric."""
    low = _normalize(question.text)
    if "=" not in low or not any(token in low for token in ["solve", "find x", "value of x"]):
        return None

    # Conservative parser: only use math characters around a single equation.
    eq_match = re.search(r"([0-9xX+\-*/().\s]+)=([0-9xX+\-*/().\s]+)", str(question.text))
    if not eq_match:
        return None

    x = sp.symbols("x")
    try:
        left = sp.sympify(eq_match.group(1).replace("X", "x"))
        right = sp.sympify(eq_match.group(2).replace("X", "x"))
        solutions = sp.solve(sp.Eq(left, right), x)
    except (sp.SympifyError, TypeError, ValueError):
        return None

    if len(solutions) != 1:
        return None

    option_id = _find_option_by_number(question, sp.N(solutions[0]))
    if option_id is None:
        return None

    return ToolDecision(
        option_id=option_id,
        strategy="tool_basic_equation_options",
        confidence=0.85,
        explanation=f"SymPy solved {left} = {right}; x = {solutions[0]}.",
    )


def first_option_fallback(question) -> ToolDecision:
    option_id = question.options[0].id
    return ToolDecision(
        option_id=option_id,
        strategy="fallback_first_option",
        confidence=0.2,
        explanation="No deterministic SymPy tool matched; using first-option fallback.",
    )


DEFAULT_TOOLS: tuple[Callable, ...] = (
    tool_lcm_gcd_options,
    tool_correlation,
    tool_simple_field_extension,
    tool_direct_product_quotient_order,
    tool_basic_equation_options,
)


def choose_with_agentic_tools(question, fallback: Callable = first_option_fallback) -> ToolDecision:
    """Try deterministic tools in order, then use fallback."""
    for tool in DEFAULT_TOOLS:
        decision = tool(question)
        if decision is not None:
            return decision
    return fallback(question)
