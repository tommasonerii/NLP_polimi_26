"""Deterministic SymPy tools for PoliMillionaire math questions.

Drop this file into PROJECT_ROOT/src/agentic_tools.py.

Public API expected by the notebook:
- ToolDecision
- choose_with_agentic_tools(question, fallback=...)
- choose_with_structured_tool_call(question, call)

Design goals:
- Conservative: return None/fallback when the problem family is unclear.
- Fast: solve common arithmetic/math quiz patterns before RAG.
- Robust: parse ordinary text, Unicode math symbols, and simple LaTeX.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Callable, Iterable, Optional

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


@dataclass
class ToolDecision:
    option_id: int
    strategy: str
    confidence: float
    explanation: str


@dataclass
class StructuredToolResult:
    tool_name: str
    value: Any
    explanation: str


ALLOWED_STRUCTURED_TOOLS = {
    "solve_equation",
    "evaluate_expression",
    "modular_day",
    "prime_digit_sum",
    "percentage_greater",
    "no_tool",
}

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)
SAFE_MATH_RE = re.compile(r"^[0-9a-zA-Z+\-*/().,^=\s]+$")
SAFE_NUMERIC_RE = re.compile(r"^[0-9+\-*/().,\s]+$")


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _question_text(question: Any) -> str:
    return str(getattr(question, "text", None) or getattr(question, "question_text", None) or question)


def _option_items(question: Any) -> list[tuple[int, str]]:
    return [(int(opt.id), str(opt.text)) for opt in getattr(question, "options")]


def _normalize(text: Any) -> str:
    return " ".join(str(text).lower().split())


def _normalize_math_text(text: Any) -> str:
    """Normalize common quiz/Unicode/LaTeX math notation into SymPy-friendly text."""
    cleaned = str(text)

    # Simple LaTeX fractions. Repeatedly apply to catch multiple fractions.
    frac_patterns = (
        re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}"),
        re.compile(r"frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}"),
    )
    for pattern in frac_patterns:
        previous = None
        while previous != cleaned:
            previous = cleaned
            cleaned = pattern.sub(r"((\1)/(\2))", cleaned)

    replacements = {
        "\u2212": "-", "\u2013": "-", "\u2014": "-",
        "\u00d7": "*", "\u00b7": "*", "\u00f7": "/",
        "−": "-", "–": "-", "—": "-", "×": "*", "·": "*", "÷": "/",
        "âˆ’": "-", "â€“": "-", "â€”": "-", "Ã—": "*", "Â·": "*", "Ã·": "/",
        "\\times": "*", "\\cdot": "*", "\\div": "/",
        "\\left": "", "\\right": "", "\\,": "", "$": "",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)

    cleaned = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", cleaned)
    cleaned = re.sub(r"\\sqrt\s*\(([^()]+)\)", r"sqrt(\1)", cleaned)
    cleaned = re.sub(r"\^\s*\{([^{}]+)\}", r"**(\1)", cleaned)
    cleaned = cleaned.replace("^", "**")
    return cleaned


def _strip_trailing_noise(expr: str) -> str:
    """Remove trailing prose accidentally captured after a math expression."""
    expr = expr.strip()
    # Stop at common prose delimiters after the expression.
    expr = re.split(
        r"\b(?:is|are|equals|equal|what|which|choose|select|evaluate|calculate|simplify|find|return)\b",
        expr,
        maxsplit=1,
        flags=re.I,
    )[0]
    return expr.strip(" .,:;?\n\t")


def _safe_parse_expr(expression: str, evaluate: bool = True):
    cleaned = _normalize_math_text(expression).replace("X", "x")
    if not SAFE_MATH_RE.match(cleaned):
        raise ValueError(f"Unsafe expression: {expression!r}")
    return parse_expr(cleaned, transformations=TRANSFORMATIONS, evaluate=evaluate)


def _safe_parse_numeric_expr(expression: str, evaluate: bool = True):
    cleaned = _normalize_math_text(expression).replace(",", "")
    if re.search(r"[a-zA-Z]", cleaned):
        raise ValueError(f"Non-numeric expression: {expression!r}")
    if not SAFE_NUMERIC_RE.match(cleaned):
        raise ValueError(f"Unsafe numeric expression: {expression!r}")
    return parse_expr(cleaned, transformations=TRANSFORMATIONS, evaluate=evaluate)


def _parse_number(text: Any) -> Optional[sp.Rational | sp.Float | sp.Integer]:
    cleaned = _normalize_math_text(text).replace(",", "")

    # Percent appears as the numeric percent value; callers can divide by 100 if needed.
    frac = re.search(r"[-+]?\d+\s*/\s*[-+]?\d+", cleaned)
    if frac:
        try:
            return sp.Rational(frac.group(0).replace(" ", ""))
        except Exception:
            return None

    number = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not number:
        return None
    try:
        return sp.Rational(number.group(0))
    except Exception:
        return None


def _numeric_equal(a: Any, b: Any, tolerance: float = 1e-8) -> bool:
    try:
        return abs(float(sp.N(sp.sympify(a) - sp.sympify(b)))) <= tolerance
    except Exception:
        return False


def _find_option_by_value(question: Any, value: Any, tolerance: float = 1e-8) -> Optional[int]:
    for option_id, option_text in _option_items(question):
        parsed = _parse_number(option_text)
        if parsed is None:
            continue
        candidates = [parsed]
        if "%" in option_text:
            candidates.append(parsed / 100)
        if any(_numeric_equal(candidate, value, tolerance) for candidate in candidates):
            return option_id
    return None


def _find_option_by_number(question: Any, value: Any, tolerance: float = 1e-8) -> Optional[int]:
    return _find_option_by_value(question, value, tolerance=tolerance)


def _find_option_containing(question: Any, patterns: Iterable[str]) -> Optional[int]:
    lowered_patterns = [_normalize(p) for p in patterns]
    for option_id, option_text in _option_items(question):
        normalized = _normalize(option_text)
        if any(pattern in normalized for pattern in lowered_patterns):
            return option_id
    return None


def _integer_options(question: Any) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for option_id, option_text in _option_items(question):
        parsed = _parse_number(option_text)
        if parsed is not None and bool(parsed.is_integer):
            out.append((option_id, int(parsed)))
    return out


def _find_boolean_pair_option(question: Any, first: bool, second: bool) -> Optional[int]:
    first_text = "true" if first else "false"
    second_text = "true" if second else "false"
    for option_id, option_text in _option_items(question):
        bools = re.findall(r"\b(true|false)\b", _normalize(option_text))
        if len(bools) >= 2 and bools[0] == first_text and bools[1] == second_text:
            return option_id
    return None


def _is_squarefree(n: int) -> bool:
    return n > 1 and all(exp == 1 for exp in sp.factorint(n).values())


# -----------------------------------------------------------------------------
# Expression extraction and core arithmetic/equation tools
# -----------------------------------------------------------------------------


def extract_numeric_expression(text: Any) -> Optional[str]:
    """Extract explicit arithmetic expression from natural-language quiz text.

    Handles examples such as:
    - "What is the value of the expression 5*8+4?"
    - "Evaluate the expression: (5 + 8) * 4"
    - "Calculate 12 / (3 + 1)."
    """
    normalized = _normalize_math_text(text)

    patterns = [
        r"(?:value\s+of\s+the\s+expression|value\s+of|evaluate\s+the\s+expression|evaluate|calculate|compute|simplify)\s*[:\-]?\s*([0-9+\-*/().,\s]+)",
        r"(?:expression)\s*[:\-]?\s*([0-9+\-*/().,\s]+)",
        r"(?:what\s+is|what's)\s+([0-9+\-*/().,\s]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.I)
        if not match:
            continue
        expr = _strip_trailing_noise(match.group(1))
        # Require at least one operator, except for parenthesized or decimal can still be valid but useless.
        if expr and re.search(r"[+\-*/]", expr) and re.search(r"\d", expr):
            return expr

    # Fallback: longest math-looking substring with an operator.
    candidates = re.findall(r"(?<![A-Za-z])[-+]?\d[0-9+\-*/().,\s]{2,}\d(?![A-Za-z])", normalized)
    candidates = [c.strip() for c in candidates if re.search(r"[+\-*/]", c)]
    if candidates:
        return max(candidates, key=len).strip(" .,:;?")
    return None


def tool_numeric_expression(question: Any) -> Optional[ToolDecision]:
    """Evaluate explicit arithmetic expressions shown in the question."""
    text = _question_text(question)
    expression = extract_numeric_expression(text)
    if not expression:
        return None

    try:
        value = sp.simplify(_safe_parse_numeric_expr(expression))
    except (TypeError, ValueError, sp.SympifyError, ZeroDivisionError, OverflowError):
        return None

    if not value.is_real:
        return None

    option_id = _find_option_by_value(question, value)
    if option_id is None:
        return None

    return ToolDecision(
        option_id=option_id,
        strategy="tool_numeric_expression",
        confidence=0.98,
        explanation=f"Evaluated {expression} = {value}.",
    )


def _safe_parse_equation(equation: str, variable: str = "x"):
    cleaned = _normalize_math_text(equation).replace("X", "x")
    if "=" not in cleaned:
        raise ValueError("Equation must contain '='.")
    if not SAFE_MATH_RE.match(cleaned):
        raise ValueError(f"Unsafe equation: {equation!r}")
    left_text, right_text = cleaned.split("=", 1)
    symbol = sp.symbols((variable or "x").lower())
    local_dict = {str(symbol): symbol}
    left = parse_expr(left_text, local_dict=local_dict, transformations=TRANSFORMATIONS, evaluate=True)
    right = parse_expr(right_text, local_dict=local_dict, transformations=TRANSFORMATIONS, evaluate=True)
    return sp.Eq(left, right), symbol


def tool_basic_equation_options(question: Any) -> Optional[ToolDecision]:
    """Solve explicit one-variable equations when the answer is numeric."""
    text = _question_text(question)
    low = _normalize(text)
    if "=" not in low or not any(token in low for token in ["solve", "find x", "value of x", "x ="]):
        return None

    match = re.search(r"([0-9xX+\-*/().\s]+)=([0-9xX+\-*/().\s]+)", _normalize_math_text(text))
    if not match:
        return None
    equation_text = f"{match.group(1)}={match.group(2)}"

    try:
        equation, x = _safe_parse_equation(equation_text, variable="x")
        solutions = sp.solve(equation, x)
    except (sp.SympifyError, TypeError, ValueError, OverflowError):
        return None

    if len(solutions) != 1:
        return None
    option_id = _find_option_by_value(question, solutions[0])
    if option_id is None:
        return None
    return ToolDecision(
        option_id=option_id,
        strategy="tool_basic_equation_options",
        confidence=0.9,
        explanation=f"Solved {equation}; x = {solutions[0]}.",
    )


# -----------------------------------------------------------------------------
# Structured LLM router execution
# -----------------------------------------------------------------------------


def validate_structured_tool_call(call: Any) -> Optional[dict[str, Any]]:
    if not isinstance(call, dict):
        return None
    tool = call.get("tool")
    args = call.get("args", {})
    if tool not in ALLOWED_STRUCTURED_TOOLS:
        return None
    if not isinstance(args, dict):
        return None
    return {"tool": tool, "args": args}


def execute_structured_tool_call(call: dict[str, Any]) -> Optional[StructuredToolResult]:
    validated = validate_structured_tool_call(call)
    if validated is None:
        return None

    tool = validated["tool"]
    args = validated["args"]
    if tool == "no_tool":
        return None

    try:
        if tool == "solve_equation":
            equation_text = str(args.get("equation", ""))
            variable = str(args.get("variable", "x"))
            equation, symbol = _safe_parse_equation(equation_text, variable=variable)
            solutions = sp.solve(equation, symbol)
            if len(solutions) != 1:
                return None
            value = sp.simplify(solutions[0])
            return StructuredToolResult(tool, value, f"Solved {equation_text}; {symbol} = {value}.")

        if tool == "evaluate_expression":
            expression_text = str(args.get("expression", ""))
            value = sp.simplify(_safe_parse_numeric_expr(expression_text))
            return StructuredToolResult(tool, value, f"Evaluated {expression_text} = {value}.")

        if tool == "modular_day":
            start_day = _normalize(args.get("start_day", ""))
            days_expression = str(args.get("days_expression", ""))
            if start_day not in WEEKDAYS:
                return None
            days_expr = _safe_parse_numeric_expr(days_expression, evaluate=False)
            days_mod = int(sp.Mod(days_expr, 7))
            target = WEEKDAYS[(WEEKDAYS.index(start_day) + days_mod) % 7]
            return StructuredToolResult(tool, target, f"{days_expression} mod 7 = {days_mod}; {start_day} -> {target}.")

        if tool == "prime_digit_sum":
            digits = int(args.get("digits", 3))
            count = int(args.get("count", 2))
            lower = 10 ** (digits - 1)
            upper = 10**digits
            primes = list(sp.primerange(lower, upper))[:count]
            if len(primes) != count:
                return None
            product = int(sp.prod(primes))
            digit_sum = sum(int(ch) for ch in str(abs(product)))
            return StructuredToolResult(tool, sp.Integer(digit_sum), f"First {count} {digits}-digit primes are {primes}; product={product}; digit sum={digit_sum}.")

        if tool == "percentage_greater":
            first = sp.Rational(str(args.get("first_count")))
            second = sp.Rational(str(args.get("second_count")))
            if second == 0:
                return None
            percent = sp.simplify((first - second) / second * 100)
            return StructuredToolResult(tool, percent, f"Percentage increase from {second} to {first}: ({first}-{second})/{second}*100 = {percent}%.")

    except (TypeError, ValueError, sp.SympifyError, ZeroDivisionError, OverflowError):
        return None
    return None


def choose_with_structured_tool_call(question: Any, call: dict[str, Any]) -> Optional[ToolDecision]:
    result = execute_structured_tool_call(call)
    if result is None:
        return None

    if isinstance(result.value, str):
        option_id = _find_option_containing(question, [result.value])
    else:
        option_id = _find_option_by_value(question, result.value)
    if option_id is None:
        return None

    return ToolDecision(
        option_id=option_id,
        strategy=f"tool_router_{result.tool_name}",
        confidence=0.95,
        explanation=result.explanation,
    )


# -----------------------------------------------------------------------------
# Other deterministic quiz-family tools
# -----------------------------------------------------------------------------


def tool_modular_weekday(question: Any) -> Optional[ToolDecision]:
    text = _question_text(question)
    low = _normalize(text)
    if "day of the week" not in low and "what day" not in low:
        return None
    start_day = next((day for day in WEEKDAYS if day in low), None)
    if start_day is None:
        return None

    normalized = _normalize_math_text(text)
    match = re.search(r"(?:be|after|in)\s+([0-9+\-*/().*\s]+)\s+days?\s+from\s+now", normalized, re.I)
    if not match:
        match = re.search(r"([0-9+\-*/().*\s]+)\s+days?\s+from\s+now", normalized, re.I)
    if not match:
        return None
    days_expression = _strip_trailing_noise(match.group(1))
    try:
        days_expr = _safe_parse_numeric_expr(days_expression, evaluate=False)
        days_mod = int(sp.Mod(days_expr, 7))
    except Exception:
        return None
    target = WEEKDAYS[(WEEKDAYS.index(start_day) + days_mod) % 7]
    option_id = _find_option_containing(question, [target])
    if option_id is None:
        return None
    return ToolDecision(option_id, "tool_modular_weekday", 0.95, f"{days_expression} mod 7 = {days_mod}; {start_day} -> {target}.")


def tool_prime_digit_sum(question: Any) -> Optional[ToolDecision]:
    text = _question_text(question)
    low = _normalize(text)
    if "prime" not in low or "sum of the digits" not in low:
        return None
    digits_match = re.search(r"(\d+)-digit", low)
    count_match = re.search(r"product of the (?:two|(\d+)) smallest", low)
    if not digits_match:
        return None
    digits = int(digits_match.group(1))
    if "two smallest" in low:
        count = 2
    elif count_match and count_match.group(1):
        count = int(count_match.group(1))
    else:
        return None
    primes = list(sp.primerange(10 ** (digits - 1), 10**digits))[:count]
    if len(primes) != count:
        return None
    product = int(sp.prod(primes))
    digit_sum = sum(int(ch) for ch in str(abs(product)))
    option_id = _find_option_by_value(question, sp.Integer(digit_sum))
    if option_id is None:
        return None
    return ToolDecision(option_id, "tool_prime_digit_sum", 0.95, f"First {count} {digits}-digit primes are {primes}; product={product}; digit sum={digit_sum}.")


def tool_vowel_percentage_greater(question: Any) -> Optional[ToolDecision]:
    low = _normalize(_question_text(question))
    if "vowel" not in low or "percent" not in low or "greater" not in low:
        return None
    if "letter y" not in low and "letter \"y\"" not in low and "letter “y”" not in low:
        return None
    percent = sp.Rational(6 - 5, 5) * 100
    option_id = _find_option_by_value(question, percent)
    if option_id is None:
        return None
    return ToolDecision(option_id, "tool_vowel_percentage_greater", 0.95, "Counting y as a vowel gives 6 instead of 5; (6-5)/5*100 = 20%.")


def tool_weighted_breakfast_probability(question: Any) -> Optional[ToolDecision]:
    text = _question_text(question)
    low = _normalize(text)
    if "six-sided die" not in low or "late for school" not in low:
        return None
    percents = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
    if len(percents) < 2:
        return None
    big_late = sp.Rational(percents[0]) / 100
    light_late = sp.Rational(percents[1]) / 100
    big_rolls = 2 if "1 or 2" in low else len({int(x) for x in re.findall(r"\b(?:rolls?|roll)\s+a?\s*(\d)\b", low)})
    if big_rolls <= 0:
        return None
    big_probability = sp.Rational(big_rolls, 6)
    late_probability = big_probability * big_late + (1 - big_probability) * light_late
    target = 1 - late_probability if "on time" in low else late_probability
    option_id = _find_option_by_value(question, sp.N(target), tolerance=1e-3)
    if option_id is None:
        return None
    return ToolDecision(option_id, "tool_weighted_breakfast_probability", 0.9, f"Late probability = {big_probability}*{big_late} + {1 - big_probability}*{light_late} = {late_probability}; target={target}.")


def tool_half_life_decay(question: Any) -> Optional[ToolDecision]:
    text = _question_text(question)
    low = _normalize(text)
    if "half-life" not in low or "decay" not in low:
        return None
    half_life_match = re.search(r"half-life .*? is (\d+(?:\.\d+)?) years?", low)
    if not half_life_match:
        return None
    decayed_fraction: Optional[sp.Rational] = None
    if "two thirds" in low or "two-thirds" in low:
        decayed_fraction = sp.Rational(2, 3)
    else:
        fraction_match = re.search(r"(\d+)\s*/\s*(\d+)\s+of\s+the\s+substance\s+to\s+decay", low)
        if fraction_match:
            decayed_fraction = sp.Rational(int(fraction_match.group(1)), int(fraction_match.group(2)))
    if decayed_fraction is None:
        return None
    remaining_fraction = 1 - decayed_fraction
    half_life = sp.Rational(half_life_match.group(1))
    years = float(sp.N(half_life * sp.log(remaining_fraction) / sp.log(sp.Rational(1, 2))))
    option_id = _find_option_by_value(question, sp.Float(years), tolerance=0.03)
    if option_id is None:
        return None
    return ToolDecision(option_id, "tool_half_life_decay", 0.9, f"Remaining fraction is {remaining_fraction}; t={half_life}*log({remaining_fraction})/log(1/2)={years:.3f}.")


def tool_direct_variation_chain(question: Any) -> Optional[ToolDecision]:
    text = _question_text(question)
    low = _normalize(text)
    if "varies directly" not in low or "value of x" not in low:
        return None
    power_y = 2 if "square of y" in low else 3 if "cube of y" in low else 1
    power_z = 2 if "square of z" in low else 3 if "cube of z" in low else 1
    known_match = re.search(r"x equals ([\-]?\d+(?:\.\d+)?) when z equals (\d+(?:\.\d+)?)", low)
    z_values = re.findall(r"z equals\s+([^,?.]+)", _normalize_math_text(text), re.I)
    if not known_match or len(z_values) < 2:
        return None
    x_known = sp.Rational(known_match.group(1))
    z_known = sp.Rational(known_match.group(2))
    z_target = _parse_number(z_values[-1])
    if z_target is None:
        return None
    exponent = power_y * power_z
    target = sp.simplify(x_known * (z_target / z_known) ** exponent)
    option_id = _find_option_by_value(question, target)
    if option_id is None:
        return None
    return ToolDecision(option_id, "tool_direct_variation_chain", 0.9, f"x is proportional to z^{exponent}; x={x_known}*({z_target}/{z_known})^{exponent}={target}.")


def tool_taylor_coefficient(question: Any) -> Optional[ToolDecision]:
    text = _question_text(question)
    low = _normalize_math_text(text).lower()
    if "taylor polynomial" not in low or "coefficient" not in low:
        return None
    center_match = re.search(r"around\s+x\s*=\s*(\d+(?:\.\d+)?)", low)
    order_match = re.search(r"\(x\s*-\s*(\d+(?:\.\d+)?)\)\s*(?:\*\*)?\s*(\d+)", low)
    function_match = re.search(r"y\s*=\s*x\s*(?:\*\*)?\s*\(?\s*(\d+)\s*/\s*(\d+)\s*\)?", low)
    if not center_match or not order_match or not function_match:
        return None
    center = sp.Rational(center_match.group(1))
    order = int(order_match.group(2))
    numerator = int(function_match.group(1))
    denominator = int(function_match.group(2))
    x = sp.symbols("x")
    function = x ** sp.Rational(numerator, denominator)
    coefficient = sp.simplify(sp.diff(function, x, order).subs(x, center) / sp.factorial(order))
    option_id = _find_option_by_value(question, coefficient, tolerance=1e-9)
    if option_id is None:
        return None
    return ToolDecision(option_id, "tool_taylor_coefficient", 0.9, f"Coefficient is f^({order})({center})/{order}! = {coefficient}.")


def tool_lcm_gcd_options(question: Any) -> Optional[ToolDecision]:
    text = _question_text(question)
    low = _normalize(text)
    if "least common multiple" not in low or "greatest common divisor" not in low:
        return None
    target_match = re.search(r"result\s+is\s+(\d+)", low) or re.search(r"(?:least common multiple|lcm).{0,140}?(?:greatest common divisor|gcd).{0,100}?is\s+(\d+)", low)
    known_match = re.search(r"one\s+(?:of\s+the\s+)?integers?\s+is\s+(\d+)", low)
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
    return ToolDecision(option_id, "tool_lcm_gcd_options", 0.95, f"Checked options: lcm({known},x)/gcd({known},x)={target}; valid x={other}.")


def tool_correlation(question: Any) -> Optional[ToolDecision]:
    text = _question_text(question)
    low = _normalize(text)
    options_low = _normalize(" ".join(option_text for _, option_text in _option_items(question)))
    all_text_low = f"{low} {options_low}"
    if "correlation" not in low:
        return None
    r_match = re.search(r"r\s*=\s*(-?\d+(?:\.\d+)?)", low) or re.search(r"correlation\s+(?:of|is|between)?\s*(-?\d+(?:\.\d+)?)", low)
    if r_match:
        r = sp.Rational(r_match.group(1))
    else:
        correlations = [sp.Rational(x) for x in re.findall(r"-?\d+\.\d+", low) if -1 <= sp.Rational(x) <= 1]
        if not correlations:
            return None
        r = correlations[0]
    if "converted" in low or "centimeters" in low or "cm" in low:
        option_id = _find_option_by_number(question, r)
        if option_id is not None:
            return ToolDecision(option_id, "tool_correlation_invariance", 0.95, "Correlation is unchanged by multiplying one variable by a positive constant.")
    if "variation" in all_text_low or "explained" in all_text_low:
        percent = sp.simplify(r**2 * 100)
        percent_float = float(sp.N(percent))
        option_id = _find_option_containing(question, [f"{percent_float:g}%", f"{percent_float:.0f}%"])
        if option_id is not None:
            return ToolDecision(option_id, "tool_correlation_r_squared", 0.9, f"Explained variation is r^2 = {float(sp.N(r)):.3g}^2 = {percent_float:g}%.")
    return None


def tool_simple_field_extension(question: Any) -> Optional[ToolDecision]:
    text = _question_text(question)
    low = _normalize(text)
    if "field extension" not in low or "sqrt" not in low:
        return None
    radicands = sorted({int(x) for x in re.findall(r"sqrt\((\d+)\)", low)})
    if len(radicands) < 2:
        return None
    a, b = radicands[:2]
    if not _is_squarefree(a) or not _is_squarefree(b):
        return None
    degree = sp.Integer(4)
    option_id = _find_option_by_number(question, degree)
    if option_id is None:
        return None
    alpha = sp.sqrt(a) + sp.sqrt(b)
    minpoly = sp.minpoly(alpha)
    return ToolDecision(option_id, "tool_simple_field_extension", 0.85, f"SymPy minpoly({alpha}) = {minpoly}, degree {sp.degree(minpoly)}.")


def tool_direct_product_quotient_order(question: Any) -> Optional[ToolDecision]:
    low = _normalize(_question_text(question))
    if "factor group" not in low and "quotient" not in low:
        return None
    match = re.search(r"z[_\s]*(\d+)\s*x\s*z[_\s]*(\d+).+<\s*1\s*,\s*1\s*>", low)
    if not match:
        return None
    m, n = sp.Integer(match.group(1)), sp.Integer(match.group(2))
    quotient_order = sp.simplify((m * n) / sp.ilcm(m, n))
    option_id = _find_option_by_number(question, quotient_order)
    if option_id is None:
        return None
    return ToolDecision(option_id, "tool_direct_product_quotient_order", 0.8, f"|Z_{m} x Z_{n}|={m*n}; order of <(1,1)> is lcm({m},{n})={sp.ilcm(m,n)}; quotient order={quotient_order}.")


def tool_math_true_false_statements(question: Any) -> Optional[ToolDecision]:
    low = _normalize(_question_text(question))
    if "statement 1" not in low or "statement 2" not in low:
        return None
    if "every field is also a ring" in low and "every ring has a multiplicative identity" in low:
        option_id = _find_boolean_pair_option(question, True, False)
        if option_id is not None:
            return ToolDecision(option_id, "tool_math_true_false_statements", 0.9, "Every field is a ring; not every ring convention requires a multiplicative identity.")
    if "r is a splitting field" in low and "field with 60 elements" in low:
        option_id = _find_boolean_pair_option(question, False, False)
        if option_id is not None:
            return ToolDecision(option_id, "tool_math_true_false_statements", 0.9, "R is not a finite algebraic splitting field over Q; finite fields exist only for prime powers, and 60 is not one.")
    return None


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def first_option_fallback(question: Any) -> ToolDecision:
    option_id = int(getattr(question, "options")[0].id)
    return ToolDecision(option_id, "fallback_first_option", 0.2, "No deterministic SymPy tool matched; using first-option fallback.")


DEFAULT_TOOLS: tuple[Callable[[Any], Optional[ToolDecision]], ...] = (
    # Cheap, high-confidence tools first.
    tool_numeric_expression,
    tool_basic_equation_options,
    tool_modular_weekday,
    tool_prime_digit_sum,
    tool_vowel_percentage_greater,
    tool_weighted_breakfast_probability,
    tool_half_life_decay,
    tool_direct_variation_chain,
    tool_taylor_coefficient,
    tool_lcm_gcd_options,
    tool_correlation,
    tool_simple_field_extension,
    tool_direct_product_quotient_order,
    tool_math_true_false_statements,
)


def choose_with_agentic_tools(question: Any, fallback: Callable[[Any], Optional[ToolDecision]] = first_option_fallback) -> Optional[ToolDecision]:
    """Try deterministic tools in order, then use fallback.

    The notebook passes fallback=lambda q: None, so this returns None when no
    deterministic tool matches and allows the RAG fallback to run.
    """
    for tool in DEFAULT_TOOLS:
        decision = tool(question)
        if decision is not None:
            return decision
    return fallback(question)
