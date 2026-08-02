# backend/app/agent/tools/calculator_tool.py
"""
计算器工具
安全的数学表达式计算，使用 AST 解析防止注入攻击
"""

import ast
import operator
from typing import Dict, Any
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
        "执行数学计算。"
        "当用户需要计算数值、公式推导、金额计算时调用。"
        "输入必须是合法的数学表达式字符串，如'(3+5)*2'或'100/5+20'。"
    )
    permission = "safe"

    parameters_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如'(3+5)*2'、'100*0.8'、'2**10'"
            },
            "precision": {
                "type": "integer",
                "description": "小数精度位数，默认2",
                "default": 2
            }
        },
        "required": ["expression"]
    }

    async def run(
        self,
        expression: str,
        precision: int = 2,
        **kwargs
    ) -> ToolResult:
        """
        执行安全的数学计算

        Args:
            expression: 数学表达式字符串
            precision: 结果精度（小数位数）

        Returns:
            ToolResult: 计算结果
        """
        try:
            logger.info(f"[计算器] 原始表达式: '{expression}'")

            # 校验：表达式不能为空
            if not expression or not expression.strip():
                return ToolResult(
                    success=False,
                    data=None,
                    error="表达式不能为空"
                )

            # 校验：表达式长度限制（防止超长攻击）
            if len(expression) > 100:
                return ToolResult(
                    success=False,
                    data=None,
                    error="表达式过长，最大100字符"
                )

            # 预处理：尝试将函数调用格式转换为中缀表达式
            expression = self._normalize_expression(expression.strip())
            logger.info(f"[计算器] 规范化后表达式: '{expression}'")

            # AST 解析
            tree = ast.parse(expression, mode='eval')

            # 安全校验：只允许特定节点类型
            for node in ast.walk(tree):
                if not self._is_safe_node(node):
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"非法表达式：包含不允许的元素 '{type(node).__name__}'"
                    )

            # 执行计算
            result = self._safe_eval(tree.body)

            # 格式化精度
            if isinstance(result, float):
                result = round(result, precision)

            formatted_result = f"{expression} = {result}"

            return ToolResult(
                success=True,
                data=formatted_result,
                metadata={
                    "expression": expression,
                    "precision": precision,
                    "raw_result": result
                }
            )

        except SyntaxError as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"表达式语法错误: {str(e)}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"计算失败: {str(e)}"
            )

    def _normalize_expression(self, expression: str) -> str:
        """
        规范化表达式：将 LLM 可能生成的各种格式转换为中缀表达式
        """
        import re

        logger.info(f"[计算器-normalize] 输入: '{expression}'")

        # 1. 去除首尾引号和空白
        expression = expression.strip()
        while (expression.startswith('"') and expression.endswith('"')) or \
              (expression.startswith("'") and expression.endswith("'")):
            expression = expression[1:-1].strip()

        # 2. 去除问题标记：=、?、！、中文标点等
        expression = expression.replace('=', '').replace('?', '').replace('!', '')
        expression = expression.replace('？', '').replace('！', '')
        expression = expression.replace('，', ',').replace('。', '')

        # 3. 处理常见的函数调用格式（LangChain Tool 风格）
        func_map = {
            'Add': '+', 'Sub': '-', 'Mult': '*', 'Div': '/',
            'Pow': '**', 'Mod': '%',
            'add': '+', 'sub': '-', 'mult': '*', 'div': '/',
            'pow': '**', 'mod': '%',
            'Addition': '+', 'Subtraction': '-', 'Multiplication': '*', 'Division': '/',
            'addition': '+', 'subtraction': '-', 'multiplication': '*', 'division': '/',
            'sum': '+', 'difference': '-', 'product': '*', 'quotient': '/',
            'Sum': '+', 'Difference': '-', 'Product': '*', 'Quotient': '/',
            'plus': '+', 'minus': '-', 'times': '*', 'divided_by': '/',
            'Plus': '+', 'Minus': '-', 'Times': '*', 'DividedBy': '/',
        }

        def split_args(args_str):
            args = []
            current = ''
            depth = 0
            for char in args_str:
                if char == '(':
                    depth += 1
                    current += char
                elif char == ')':
                    depth -= 1
                    current += char
                elif char == ',' and depth == 0:
                    args.append(current.strip())
                    current = ''
                else:
                    current += char
            if current:
                args.append(current.strip())
            return args

        # 3.1 辅助函数：找到匹配的右括号
        def find_matching_paren(s: str, start: int) -> int:
            depth = 0
            for i in range(start, len(s)):
                if s[i] == '(':
                    depth += 1
                elif s[i] == ')':
                    depth -= 1
                    if depth == 0:
                        return i
            return -1

        # 3.2 从内到外逐层替换函数调用（支持嵌套括号）
        max_iter = 20
        iteration = 0
        changed = True
        func_names = list(func_map.keys())
        func_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(f) for f in func_names) + r')\s*\('
        )

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

                args_str = expression[paren_start + 1:paren_end]
                logger.info(f"[计算器-normalize] 匹配到函数: {func_name}({args_str})")

                op = func_map.get(func_name)
                if op:
                    args_list = split_args(args_str)
                    if len(args_list) == 2:
                        replacement = f"({args_list[0]}{op}{args_list[1]})"
                        expression = expression[:match.start()] + replacement + expression[paren_end + 1:]
                        logger.info(f"[计算器-normalize] 替换后: '{expression}'")
                        changed = True
                        replaced = True
                        break
                    elif len(args_list) == 1:
                        replacement = f"({op}{args_list[0]})"
                        expression = expression[:match.start()] + replacement + expression[paren_end + 1:]
                        logger.info(f"[计算器-normalize] 一元运算替换后: '{expression}'")
                        changed = True
                        replaced = True
                        break

            if not replaced:
                break

        # 4. 处理中文数字和运算符
        cn_num_map = {
            '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
            '十': '10', '百': '100', '千': '1000', '万': '10000',
        }
        cn_op_map = {
            '加': '+', '减': '-', '乘': '*', '除': '/',
            '加上': '+', '减去': '-', '乘以': '*', '除以': '/',
        }

        # 5. 处理 Markdown 代码块格式
        expression = expression.replace('```', '').replace('`', '')

        # 6. 处理 "计算：" 或 "expression:" 等前缀
        prefix_patterns = [
            r'^计算[：:]\s*',
            r'^expression[：:]\s*',
            r'^表达式[：:]\s*',
            r'^结果是[：:]\s*',
        ]
        for pattern in prefix_patterns:
            expression = re.sub(pattern, '', expression, flags=re.IGNORECASE)

        # 7. 去除所有空白字符
        expression = re.sub(r'\s+', '', expression)

        logger.info(f"[计算器-normalize] 最终输出: '{expression}'")
        return expression

    def _split_args(self, args_str: str) -> list:
        """
        安全分割函数参数（处理嵌套括号）

        Args:
            args_str: 参数字符串

        Returns:
            参数列表
        """
        args = []
        current = ''
        depth = 0

        for char in args_str:
            if char == '(' and depth == 0:
                depth += 1
                current += char
            elif char == ')' and depth > 0:
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                args.append(current.strip())
                current = ''
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
            ast.Constant,    # 常量（数字）
            ast.BinOp,       # 二元运算
            ast.UnaryOp,     # 一元运算（负号）
            ast.Num,         # Python 3.7 兼容：数字节点
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

        raise ValueError(f"不支持节点类型: {type(node).__name__}")