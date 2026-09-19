# backend/app/agent/tools/calculator_tool.py
"""
计算器工具
安全的数学表达式计算，使用 AST 解析防止注入攻击
"""

import ast
import json
import operator
import re
from typing import Any

from backend.app.agent.tools.base import BaseTool, ToolResult
from backend.app.common.logger import logger

# ====== 安全的 AST 运算映射 ======
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,  # 负号
    ast.UAdd: operator.pos,  # 正号
}


class CalculatorTool(BaseTool):
    """
    数学计算工具

    功能：
    - 支持基础四则运算、幂运算、取余
    - 使用 AST 解析，禁止执行任意代码
    - 只允许数字和运算符，防止注入攻击
    """

    name = "calculator"
    description = (
        "数值计算工具：对已知数值做四则、括号、幂(**)、max/min/sum 运算。\n"
        "仅当需要从给定数据推导新数值时调用（增长率/百分比/求和/均值/金额等），禁止模型心算或估算。\n"
        "不要用于：纯数字大小对比、排序、枚举排名（无需计算）；以及题干已直接给出所需数值时（直接引用，勿重算）。"
    )
    permission = "safe"
    parameters_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "单个/多行数学表达式字符串；与expressions二选一。",
            },
            "expressions": {
                "type": "array",
                "description": "批量独立表达式数组，优先使用；与expression二选一。",
                "items": {
                    "type": "string",
                    "description": "表达式字符串，可附加业务标签。",
                },
            },
            "precision": {
                "type": "integer",
                "description": "结果保留小数位数，默认2",
                "default": 2,
            },
            "steps": {
                "type": "array",
                "description": "串行多步依赖计算，顺序执行，后续表达式可引用前面定义的变量。",
                "items": {
                    "type": "object",
                    "properties": {
                        "var": {
                            "type": "string",
                            "description": "可选：当前结果绑定变量名，供后续步骤使用",
                        },
                        "expr": {
                            "type": "string",
                            "description": "当前步骤数学表达式，支持引用已定义变量",
                        },
                    },
                    "required": ["expr"],
                },
            },
        },
        "required": [],
    }

    @staticmethod
    def _coerce_int(value) -> "int | None":
        """
        兜底：把模型可能传入的非严格整数收敛为 int。
        覆盖模型偶尔"犯蠢"的情况：
          "2.0" (字符串) / 2.0 (float) / 2 (int) -> 2
        无法解析（如 "abc"）时返回 None，交由默认值或校验处理。
        """
        try:
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(round(value))
            if isinstance(value, str):
                s = value.strip()
                if not s:
                    return None
                return int(round(float(s)))
        except (ValueError, TypeError):
            return None
        return None

    def validate_args(self, **kwargs):
        """
        父类强校验：在 Pydantic 校验前先把 precision 收敛为 int，
        避免 7B 模型传入 "2.0" / 2.0 时在校验阶段被拦、导致无谓重试。
        """
        if "precision" in kwargs:
            coerced = self._coerce_int(kwargs["precision"])
            if coerced is not None:
                kwargs["precision"] = coerced
        return super().validate_args(**kwargs)

    async def run(
        self,
        expression: str = None,
        expressions: Any = None,
        steps: Any = None,
        precision: int = 2,
        **kwargs,
    ) -> ToolResult:
        """
        执行安全的数学计算，支持单表达式、多表达式或多步依赖计算。

        Args:
            expression: 单个表达式字符串，或多行文本（每行一个表达式）。
            expressions: 多个表达式数组（优先）。
            steps: 依赖式多步计算步骤。
            precision: 结果精度（小数位数）。

        Returns:
            ToolResult: 计算结果
        """
        # 兜底：大模型可能把 precision 传成 "2.0" / 2.0 等非严格整数，
        # 统一收敛为 int 后再参与 round()，避免 TypeError。
        coerced = self._coerce_int(precision)
        precision = coerced if coerced is not None else 2

        try:
            # 方案A：多步/依赖式计算优先
            if steps:
                return await self._run_steps(steps, precision)

            expr_list = self._collect_expressions(expression, expressions)
            if not expr_list:
                return ToolResult(
                    success=False,
                    data=None,
                    error="expression 或 expressions 必须提供一个非空表达式",
                )

            # 单个表达式：保持原有返回格式，兼容旧逻辑
            if len(expr_list) == 1:
                return await self._eval_expression(expr_list[0], precision)

            # 多个表达式：逐条计算并汇总
            return await self._eval_multiple(expr_list, precision)

        except SyntaxError as e:
            return ToolResult(success=False, data=None, error=f"表达式语法错误: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, data=None, error=f"计算失败: {str(e)}")

    def _collect_expressions(self, expression: Any, expressions: Any) -> list[str]:
        """
        从 expression / expressions 中收集所有待计算的表达式。
        支持：数组、JSON 字符串数组、多行文本、单行文本。
        """
        expr_list: list[str] = []

        # 1. 优先处理 expressions 参数
        if expressions is not None:
            parsed = expressions
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except (json.JSONDecodeError, TypeError):
                    # 若解析失败，按换行/分号拆成数组
                    parsed = [line.strip() for line in re.split(r"[\n;]+", parsed) if line.strip()]
            if isinstance(parsed, list):
                expr_list = [str(item).strip() for item in parsed if str(item).strip()]
            elif isinstance(parsed, (int, float)):
                expr_list = [str(parsed)]

        # 2. 兼容 expression 参数
        if not expr_list and expression is not None:
            if isinstance(expression, (list, tuple)):
                expr_list = [str(item).strip() for item in expression if str(item).strip()]
            else:
                text = str(expression).strip()
                # 单行且不含换号/分号：直接作为一个表达式
                if "\n" not in text and ";" not in text:
                    # 兼容弱模型把多个表达式用逗号拼接成一个 expression 的情况
                    # 例如："3680 * 0.042, 2950 * 0.058, ..."
                    parts = [p.strip() for p in text.split(",") if p.strip()]
                    if len(parts) > 1 and all(self._looks_like_expression(p) for p in parts):
                        expr_list = parts
                    else:
                        expr_list = [text]
                else:
                    expr_list = [line.strip() for line in re.split(r"[\n;]+", text) if line.strip()]

        return expr_list

    def _looks_like_expression(self, s: str) -> bool:
        """
        判断字符串是否像一个独立的数学表达式。
        用于把逗号分隔的多表达式拆分时，避免误拆函数参数（如 max(1,2,3)）。
        """
        s = s.strip()
        if not s:
            return False
        # 至少包含一个数字
        if not re.search(r"\d", s):
            return False
        # 只允许数字、运算符、括号、小数点、百分号、空格
        if not re.match(r"^[0-9+\-*/().\^%\s]+$", s):
            return False
        # 括号匹配
        if s.count("(") != s.count(")"):
            return False
        return True

    def _extract_label(self, raw_line: str) -> str | None:
        """
        从 '家居家纺: (3680-2450)/2450*100' 中提取类别标签。
        如果没有可识别的标签，返回 None。
        """
        if not raw_line:
            return None
        # 匹配 标签[:：=是]表达式 的形式；标签应为非空文本，且不是纯数字/运算符开头
        m = re.match(r"^\s*([^\n:：=是]+?)\s*[:：=是]\s*(.+)$", raw_line)
        if m:
            label = m.group(1).strip()
            # 避免把表达式本身的一部分误判为标签（如 "10:5"）
            if label and not re.match(r"^[\d\s+\-*/().,]+$", label):
                return label
        return None

    def _extract_expression(self, raw_line: str) -> str | None:
        """
        从可能带标签的行中提取纯数学表达式。
        """
        if not raw_line:
            return None
        # 先尝试去掉标签
        label = self._extract_label(raw_line)
        if label:
            m = re.match(r"^\s*" + re.escape(label) + r"\s*[:：=是]\s*(.+)$", raw_line)
            if m:
                return m.group(1).strip()
        # 去除常见前缀
        cleaned = re.sub(r"^(?:计算|表达式|expression)[：:]?\s*", "", raw_line, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        return cleaned if cleaned else None

    def _try_parse_number_list(self, expression: str, precision: int) -> ToolResult | None:
        """
        兼容弱模型把已算出的数值列表当成表达式传入的情况。
        例如：'570.94, 336.0, 202.5, 927.73' -> 返回列表、最小值、最大值。
        """
        if "," not in expression:
            return None
        # 规范化后应为纯数字、逗号、点、正负号组成的字符串
        if not re.match(r"^[\d.,+\-]+$", expression):
            return None
        parts = [p.strip() for p in expression.split(",") if p.strip()]
        if len(parts) < 2:
            return None
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return None
        nums_str = ", ".join(str(n) for n in nums)
        min_v = round(min(nums), precision)
        max_v = round(max(nums), precision)
        formatted = f"数值列表: {nums_str}\n最小值: {min_v}\n最大值: {max_v}"
        return ToolResult(
            success=True,
            data=formatted,
            metadata={
                "expression": expression,
                "is_number_list": True,
                "numbers": nums,
                "min": min_v,
                "max": max_v,
                "precision": precision,
            },
        )

    async def _eval_expression(self, expression: str, precision: int) -> ToolResult:
        """
        计算单个表达式，返回与旧版 run 单表达式一致的 ToolResult。
        """
        if not expression or not expression.strip():
            return ToolResult(success=False, data=None, error="表达式不能为空")

        if len(expression) > 500:
            return ToolResult(success=False, data=None, error="表达式过长，最大500字符")

        logger.info(f"[计算器] 原始表达式: '{expression}'")
        expression = self._normalize_expression(expression.strip())
        logger.info(f"[计算器] 规范化后表达式: '{expression}'")

        if not expression:
            return ToolResult(success=False, data=None, error="表达式不能为空")

        # 兼容：模型传入逗号分隔的纯数字列表（用于比较大小）
        number_list_res = self._try_parse_number_list(expression, precision)
        if number_list_res:
            return number_list_res

        priority_warning = self._check_operator_priority(expression)
        if priority_warning:
            logger.warning(f"[计算器] 表达式可能存在运算符优先级问题: {priority_warning}")

        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if not self._is_safe_node(node):
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"非法表达式：包含不允许的元素 '{type(node).__name__}'",
                )

        result = self._safe_eval(tree.body)
        if isinstance(result, float):
            result = round(result, precision)

        formatted_result = f"{expression} = {result}"
        return ToolResult(
            success=True,
            data=formatted_result,
            metadata={"expression": expression, "precision": precision, "raw_result": result},
        )

    async def _eval_multiple(self, expressions: list[str], precision: int) -> ToolResult:
        """
        批量计算多个表达式，返回汇总结果。
        """
        results: list[dict[str, Any]] = []
        any_success = False

        for idx, raw_line in enumerate(expressions, start=1):
            label = self._extract_label(raw_line) or f"表达式{idx}"
            expr_text = self._extract_expression(raw_line)
            if not expr_text:
                results.append(
                    {
                        "label": label,
                        "expression": raw_line,
                        "result": None,
                        "raw_result": None,
                        "error": "无法从中提取合法数学表达式",
                    }
                )
                continue

            single_res = await self._eval_expression(expr_text, precision)
            if single_res.success:
                any_success = True
                results.append(
                    {
                        "label": label,
                        "expression": expr_text,
                        "result": single_res.metadata.get("raw_result"),
                        "raw_result": single_res.metadata.get("raw_result"),
                        "error": None,
                    }
                )
            else:
                results.append(
                    {
                        "label": label,
                        "expression": expr_text,
                        "result": None,
                        "raw_result": None,
                        "error": single_res.error,
                    }
                )

        if not any_success:
            first_error = next(
                (r["error"] for r in results if r.get("error")), "所有表达式计算失败"
            )
            return ToolResult(success=False, data=None, error=first_error)

        formatted_lines = []
        for r in results:
            if r["error"]:
                formatted_lines.append(f"{r['label']}: 错误 - {r['error']}")
            else:
                formatted_lines.append(f"{r['label']}: {r['expression']} = {r['result']}")
        formatted = "\n".join(formatted_lines)

        # 取最后一个成功结果作为最终值（方便后续链式引用）
        final_result = None
        for r in reversed(results):
            if r["error"] is None:
                final_result = r["result"]
                break

        return ToolResult(
            success=True,
            data=formatted,
            metadata={
                "multi": True,
                "expressions": results,
                "precision": precision,
                "final_result": final_result,
            },
        )

    async def _run_steps(self, steps: Any, precision: int = 2) -> ToolResult:
        """
        方案A：多步 / 依赖式计算执行。

        steps: list of {"var": str(optional), "expr": str}
        按顺序执行，后续 expr 可通过变量名引用之前步骤的结果（变量传递）。
        """
        import re as _re

        if not isinstance(steps, list) or not steps:
            return ToolResult(success=False, data=None, error="steps 必须是非空列表")
        var_env: dict[str, float] = {}
        lines: list[str] = []
        step_results: list[dict] = []
        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict) or not step.get("expr"):
                return ToolResult(success=False, data=None, error=f"第{idx}步缺少 expr 字段")
            expr_raw = str(step.get("expr", "")).strip()
            var_name = step.get("var")
            if not expr_raw:
                return ToolResult(success=False, data=None, error=f"第{idx}步 expr 为空")
            # 变量替换（单词边界，避免误替如 gross/grossmargin）
            expr_sub = expr_raw
            for name, val in var_env.items():
                expr_sub = _re.sub(r"\b" + _re.escape(name) + r"\b", str(val), expr_sub)
            logger.info(f"[计算器-steps] 第{idx}步 替换后: '{expr_sub}' (来自 '{expr_raw}')")
            try:
                norm = self._normalize_expression(expr_sub)
                tree = ast.parse(norm, mode="eval")
                for node in ast.walk(tree):
                    if not self._is_safe_node(node):
                        return ToolResult(
                            success=False,
                            data=None,
                            error=f"第{idx}步非法表达式：包含不允许的元素 '{type(node).__name__}'",
                        )
                result = self._safe_eval(tree.body)
                if isinstance(result, float):
                    result = round(result, precision)
            except Exception as e:
                return ToolResult(
                    success=False, data=None, error=f"第{idx}步计算失败: {expr_raw} -> {e}"
                )
            if var_name:
                var_env[var_name] = result
            lines.append(f"步骤{idx}: {expr_raw} = {result}")
            step_results.append({"var": var_name, "expr": expr_raw, "result": result})
            logger.info(f"[计算器-steps] 第{idx}步结果: {result}")
        formatted = "\n".join(lines)
        final = step_results[-1]["result"] if step_results else None
        return ToolResult(
            success=True,
            data=formatted,
            metadata={
                "is_steps": True,
                "steps": step_results,
                "final_result": final,
                "precision": precision,
            },
        )

    def _normalize_expression(self, expression: str) -> str:
        """
        规范化表达式：将 LLM 可能生成的各种格式转换为中缀表达式
        """
        import re

        logger.info(f"[计算器-normalize] 输入: '{expression}'")

        # 1. 去除首尾引号和空白
        expression = expression.strip()
        while (expression.startswith('"') and expression.endswith('"')) or (
            expression.startswith("'") and expression.endswith("'")
        ):
            expression = expression[1:-1].strip()

        # 2. 去除问题标记：=、?、！、中文标点等
        # 2.1 幂运算符号归一：'a^b' 中的 '^' 映射为 Python 幂运算符 '**'
        # （部分模型按 XOR/通用记号输出 '^'，而 AST 白名单仅允许 '**'，否则会被判为"非法表达式"）
        expression = expression.replace("^", "**")
        expression = expression.replace("=", "").replace("?", "").replace("!", "")
        expression = expression.replace("？", "").replace("！", "")
        expression = expression.replace("，", ",").replace("。", "")

        # 3. 处理常见的函数调用格式（LangChain Tool 风格）
        func_map = {
            "Add": "+",
            "Sub": "-",
            "Mult": "*",
            "Div": "/",
            "Pow": "**",
            "Mod": "%",
            "add": "+",
            "sub": "-",
            "mult": "*",
            "div": "/",
            "pow": "**",
            "mod": "%",
            "Addition": "+",
            "Subtraction": "-",
            "Multiplication": "*",
            "Division": "/",
            "addition": "+",
            "subtraction": "-",
            "multiplication": "*",
            "division": "/",
            "sum": "+",
            "difference": "-",
            "product": "*",
            "quotient": "/",
            "Sum": "+",
            "Difference": "-",
            "Product": "*",
            "Quotient": "/",
            "plus": "+",
            "minus": "-",
            "times": "*",
            "divided_by": "/",
            "Plus": "+",
            "Minus": "-",
            "Times": "*",
            "DividedBy": "/",
        }

        def split_args(args_str):
            args = []
            current = ""
            depth = 0
            for char in args_str:
                if char == "(":
                    depth += 1
                    current += char
                elif char == ")":
                    depth -= 1
                    current += char
                elif char == "," and depth == 0:
                    args.append(current.strip())
                    current = ""
                else:
                    current += char
            if current:
                args.append(current.strip())
            return args

        # 3.1 辅助函数：找到匹配的右括号
        def find_matching_paren(s: str, start: int) -> int:
            depth = 0
            for i in range(start, len(s)):
                if s[i] == "(":
                    depth += 1
                elif s[i] == ")":
                    depth -= 1
                    if depth == 0:
                        return i
            return -1

        # 3.2 从内到外逐层替换函数调用（支持嵌套括号）
        max_iter = 20
        iteration = 0
        changed = True
        func_names = list(func_map.keys())
        func_pattern = re.compile(r"\b(" + "|".join(re.escape(f) for f in func_names) + r")\s*\(")

        while changed and iteration < max_iter:
            changed = False
            iteration += 1

            matches = list(func_pattern.finditer(expression))
            if not matches:
                break

            replaced = False
            for match in reversed(matches):
                func_name = match.group(1)
                paren_start = match.end() - 1
                paren_end = find_matching_paren(expression, paren_start)
                if paren_end == -1:
                    continue

                args_str = expression[paren_start + 1 : paren_end]
                logger.info(f"[计算器-normalize] 匹配到函数: {func_name}({args_str})")

                op = func_map.get(func_name)
                if op:
                    args_list = split_args(args_str)
                    if len(args_list) == 2:
                        replacement = f"({args_list[0]}{op}{args_list[1]})"
                        expression = (
                            expression[: match.start()] + replacement + expression[paren_end + 1 :]
                        )
                        logger.info(f"[计算器-normalize] 替换后: '{expression}'")
                        changed = True
                        replaced = True
                        break
                    elif len(args_list) == 1:
                        replacement = f"({op}{args_list[0]})"
                        expression = (
                            expression[: match.start()] + replacement + expression[paren_end + 1 :]
                        )
                        logger.info(f"[计算器-normalize] 一元运算替换后: '{expression}'")
                        changed = True
                        replaced = True
                        break

            if not replaced:
                break

        # 5. 处理 Markdown 代码块格式
        expression = expression.replace("```", "").replace("`", "")

        # 6. 处理 "计算：" 或 "expression:" 等前缀
        prefix_patterns = [
            r"^计算[：:]\s*",
            r"^expression[：:]\s*",
            r"^表达式[：:]\s*",
            r"^结果是[：:]\s*",
        ]
        for pattern in prefix_patterns:
            expression = re.sub(pattern, "", expression, flags=re.IGNORECASE)

        # 7. 去除数字中的千分位逗号（如 1,360 -> 1360），但保留函数参数分隔逗号
        expression = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expression)

        # 8. 去除所有空白字符
        expression = re.sub(r"\s+", "", expression)

        logger.info(f"[计算器-normalize] 最终输出: '{expression}'")
        return expression

    def _check_operator_priority(self, expression: str) -> str | None:
        """
        检查表达式是否可能存在运算符优先级错误

        主要检测：LLM错误地将 a+b*c 转换为 (a+b)*c 的情况
        即：括号内只有加减运算，括号外跟乘除运算的模式

        Args:
            expression: 规范化后的表达式

        Returns:
            警告信息，如果没有问题则返回 None
        """
        import re

        # 匹配 (A+B)*C 或 (A-B)*C 模式
        # 其中 A, B, C 是数字
        pattern1 = r"^\(\s*[\d.]+\s*[+\-]\s*[\d.]+\s*\)\s*[*/%]\s*[\d.]+$"
        # 匹配 A*(B+C) 或 A/(B-C) 模式
        pattern2 = r"^[\d.]+\s*[*/%]\s*\(\s*[\d.]+\s*[+\-]\s*[\d.]+\s*\)$"
        # 匹配 ((A+B)*C)*D 等多层嵌套
        pattern3 = r"^\(\(\s*[\d.]+\s*[+\-]\s*[\d.]+\s*\)\s*[*/%]\s*[\d.]+\s*\)"

        if re.match(pattern1, expression) or re.match(pattern3, expression):
            return f"检测到表达式 '{expression}' 可能存在运算符优先级问题。括号内是加减运算，括号外是乘除运算。如果原意是先算乘除后算加减，请去掉括号。例如 '65+9*2' 应该直接传 '65+9*2'，不要传 '(65+9)*2'。"

        if re.match(pattern2, expression):
            return f"检测到表达式 '{expression}' 可能存在运算符优先级问题。括号内是加减运算，括号外是乘除运算。如果原意是先算乘除后算加减，请去掉括号。"

        return None

    def _split_args(self, args_str: str) -> list:
        """
        安全分割函数参数（处理嵌套括号）

        Args:
            args_str: 参数字符串

        Returns:
            参数列表
        """
        args = []
        current = ""
        depth = 0

        for char in args_str:
            if char == "(" and depth == 0:
                depth += 1
                current += char
            elif char == ")" and depth > 0:
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += char

        if current:
            args.append(current.strip())

        return args

    def _is_safe_node(self, node: ast.AST) -> bool:
        """
        判断 AST 节点是否安全

        Args:
            node: AST 节点

        Returns:
            bool: 是否允许
        """
        safe_node_types = (
            ast.Expression,  # 表达式根节点
            ast.Constant,  # 常量（数字）
            ast.BinOp,  # 二元运算
            ast.UnaryOp,  # 一元运算（负号）
            ast.Num,  # Python 3.7 兼容：数字节点
            ast.Call,  # 函数调用（仅允许 max/min）
            ast.Name,  # 函数名（配合 ast.Call 使用）
            ast.Load,  # 名称加载上下文
        )

        if isinstance(node, safe_node_types):
            # 对于二元运算和一元运算，进一步检查运算符是否在允许列表
            if isinstance(node, (ast.BinOp, ast.UnaryOp)):
                return type(node.op) in SAFE_OPERATORS
            return True

        # 运算符节点本身（ast.Add, ast.Mult 等）也需要允许
        # 因为 ast.walk() 会遍历到这些节点
        if type(node) in SAFE_OPERATORS:
            return True

        return False

    def _safe_eval(self, node: ast.AST) -> Any:
        """
        安全地执行 AST 计算

        Args:
            node: AST 节点

        Returns:
            计算结果
        """
        # 常量节点（Python 3.8+）
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("只支持数字常量")

        # 数字节点（Python 3.7 兼容）
        if isinstance(node, ast.Num):
            return node.n

        # 二元运算
        if isinstance(node, ast.BinOp):
            left = self._safe_eval(node.left)
            right = self._safe_eval(node.right)
            op_func = SAFE_OPERATORS[type(node.op)]
            return op_func(left, right)

        # 一元运算（负号）
        if isinstance(node, ast.UnaryOp):
            operand = self._safe_eval(node.operand)
            op_func = SAFE_OPERATORS[type(node.op)]
            return op_func(operand)

        # 函数调用：仅允许 max / min / sum
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(f"不支持函数类型: {type(node.func).__name__}")
            func_name = node.func.id
            if func_name not in ("max", "min", "sum"):
                raise ValueError(f"不允许的函数调用: {func_name}")
            if not node.args:
                raise ValueError(f"{func_name}() 需要至少一个参数")
            if node.keywords:
                raise ValueError(f"{func_name}() 不支持关键字参数")
            values = [self._safe_eval(arg) for arg in node.args]
            if func_name == "max":
                return max(values)
            elif func_name == "min":
                return min(values)
            else:
                return sum(values)

        raise ValueError(f"不支持节点类型: {type(node).__name__}")
