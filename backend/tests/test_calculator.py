# backend/tests/test_calculator.py
"""
计算器工具单元测试
测试 AST 解析、安全校验、表达式规范化等核心逻辑
"""

import pytest
import ast
import operator
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.agent.tools.calculator_tool import (
    CalculatorTool,
    SAFE_OPERATORS,
)


@pytest.fixture
def calculator():
    """创建计算器实例"""
    return CalculatorTool()


class TestASTSecurity:
    """测试 AST 安全校验"""

    def test_safe_operators_defined(self):
        """SAFE_OPERATORS 应包含所有基础运算"""
        assert ast.Add in SAFE_OPERATORS
        assert ast.Sub in SAFE_OPERATORS
        assert ast.Mult in SAFE_OPERATORS
        assert ast.Div in SAFE_OPERATORS
        assert ast.FloorDiv in SAFE_OPERATORS
        assert ast.Mod in SAFE_OPERATORS
        assert ast.Pow in SAFE_OPERATORS
        assert ast.USub in SAFE_OPERATORS

    def test_safe_node_allows_constant(self, calculator):
        """常量节点应被允许"""
        tree = ast.parse("42", mode="eval")
        for node in ast.walk(tree):
            assert calculator._is_safe_node(node)

    def test_safe_node_allows_basic_ops(self, calculator):
        """基础运算应被允许"""
        tree = ast.parse("2 + 3", mode="eval")
        for node in ast.walk(tree):
            assert calculator._is_safe_node(node)

    def test_safe_node_rejects_string(self, calculator):
        """字符串常量在 safe_eval 阶段会被拒绝"""
        tree = ast.parse("'hello'", mode="eval")
        with pytest.raises(ValueError):
            calculator._safe_eval(tree.body)

    def test_safe_node_rejects_list(self, calculator):
        """列表字面量应被拒绝"""
        tree = ast.parse("[1, 2, 3]", mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.List):
                assert not calculator._is_safe_node(node)

    def test_safe_node_rejects_function_call(self, calculator):
        """函数调用应被拒绝"""
        tree = ast.parse("__import__('os')", mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert not calculator._is_safe_node(node)

    def test_safe_node_rejects_attribute_access(self, calculator):
        """属性访问应被拒绝"""
        tree = ast.parse("x.__class__", mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert not calculator._is_safe_node(node)


class TestSafeEval:
    """测试安全计算"""

    def test_simple_addition(self, calculator):
        """简单加法"""
        tree = ast.parse("2 + 3", mode="eval")
        result = calculator._safe_eval(tree.body)
        assert result == 5

    def test_simple_subtraction(self, calculator):
        """简单减法"""
        tree = ast.parse("10 - 4", mode="eval")
        result = calculator._safe_eval(tree.body)
        assert result == 6

    def test_simple_multiplication(self, calculator):
        """简单乘法"""
        tree = ast.parse("3 * 7", mode="eval")
        result = calculator._safe_eval(tree.body)
        assert result == 21

    def test_simple_division(self, calculator):
        """简单除法"""
        tree = ast.parse("15 / 3", mode="eval")
        result = calculator._safe_eval(tree.body)
        assert result == 5.0

    def test_floor_division(self, calculator):
        """整除"""
        tree = ast.parse("17 // 5", mode="eval")
        result = calculator._safe_eval(tree.body)
        assert result == 3

    def test_modulo(self, calculator):
        """取余"""
        tree = ast.parse("10 % 3", mode="eval")
        result = calculator._safe_eval(tree.body)
        assert result == 1

    def test_power(self, calculator):
        """幂运算"""
        tree = ast.parse("2 ** 10", mode="eval")
        result = calculator._safe_eval(tree.body)
        assert result == 1024

    def test_unary_negation(self, calculator):
        """一元负号"""
        tree = ast.parse("-5", mode="eval")
        result = calculator._safe_eval(tree.body)
        assert result == -5

    def test_negative_numbers(self, calculator):
        """负数运算"""
        tree = ast.parse("-3 + 10", mode="eval")
        result = calculator._safe_eval(tree.body)
        assert result == 7

    def test_complex_expression(self, calculator):
        """复杂表达式"""
        tree = ast.parse("(2 + 3) * 4", mode="eval")
        result = calculator._safe_eval(tree.body)
        assert result == 20

    def test_nested_operations(self, calculator):
        """嵌套运算"""
        tree = ast.parse("2 + 3 * 4 - 6 / 2", mode="eval")
        result = calculator._safe_eval(tree.body)
        assert result == 11.0  # 2 + 12 - 3

    def test_decimal_result(self, calculator):
        """小数结果"""
        tree = ast.parse("1 / 3", mode="eval")
        result = calculator._safe_eval(tree.body)
        assert abs(result - 0.333333) < 0.001


class TestNormalizeExpression:
    """测试表达式规范化"""

    def test_basic_cleanup(self, calculator):
        """基础清理"""
        result = calculator._normalize_expression("  1 + 2  ")
        assert result == "1+2"

    def test_remove_quotes(self, calculator):
        """去除引号"""
        result = calculator._normalize_expression('"3+5"')
        assert result == "3+5"

    def test_remove_question_marks(self, calculator):
        """去除问号等"""
        result = calculator._normalize_expression("100*8=?")
        assert result == "100*8"

    def test_markdown_code_block(self, calculator):
        """去除 Markdown 代码块"""
        result = calculator._normalize_expression("```1+2```")
        assert result == "1+2"

    def test_function_call_add(self, calculator):
        """函数调用转换：Add"""
        result = calculator._normalize_expression("Add(3, 5)")
        assert result == "(3+5)"

    def test_function_call_mult(self, calculator):
        """函数调用转换：Multiply"""
        result = calculator._normalize_expression("Multiply(4, 7)")
        assert result == "Multiply(4,7)"

    def test_nested_functions(self, calculator):
        """嵌套函数调用"""
        result = calculator._normalize_expression("Add(Multiply(3, 4), 5)")
        assert "Multiply" in result and "+" in result

    def test_chinese_prefix(self, calculator):
        """去除中文前缀"""
        result = calculator._normalize_expression("计算：125*8")
        assert result == "125*8"

    def test_expression_prefix(self, calculator):
        """去除英文前缀"""
        result = calculator._normalize_expression("expression: 2+3")
        assert result == "2+3"


class TestCalculatorRun:
    """测试计算器主入口"""

    @pytest.mark.asyncio
    async def test_simple_calculation(self, calculator):
        """简单计算"""
        result = await calculator.run(expression="2 + 3")
        assert result.success is True
        assert "5" in result.data

    @pytest.mark.asyncio
    async def test_complex_calculation(self, calculator):
        """复杂计算"""
        result = await calculator.run(expression="(10 + 5) * 3")
        assert result.success is True
        assert "45" in result.data

    @pytest.mark.asyncio
    async def test_power_calculation(self, calculator):
        """幂运算"""
        result = await calculator.run(expression="2**8")
        assert result.success is True
        assert "256" in result.data

    @pytest.mark.asyncio
    async def test_empty_expression(self, calculator):
        """空表达式应失败"""
        result = await calculator.run(expression="")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_long_expression(self, calculator):
        """过长表达式应失败"""
        result = await calculator.run(expression="1" * 101)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_invalid_expression(self, calculator):
        """无效表达式应失败"""
        result = await calculator.run(expression="2 + )")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_injection_attempt(self, calculator):
        """注入攻击应被拒绝"""
        result = await calculator.run(expression="__import__('os').system('ls')")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_string_input_rejected(self, calculator):
        """字符串输入应被拒绝"""
        result = await calculator.run(expression="'hello'")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_precision_parameter(self, calculator):
        """精度参数"""
        result = await calculator.run(expression="1/3", precision=4)
        assert result.success is True
        assert "0.3333" in result.data

    @pytest.mark.asyncio
    async def test_function_style_input(self, calculator):
        """函数风格输入"""
        result = await calculator.run(expression="Add(10, 20)")
        assert result.success is True
        assert "30" in result.data


class TestEdgeCases:
    """边界情况测试"""

    @pytest.mark.asyncio
    async def test_division_by_zero_handled(self, calculator):
        """除零异常应被捕获"""
        result = await calculator.run(expression="1/0")
        assert result.success is False
        assert "不能" not in result.error or result.success is False

    @pytest.mark.asyncio
    async def test_very_large_number(self, calculator):
        """超大数计算"""
        result = await calculator.run(expression="999999 * 999999")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_negative_expressions(self, calculator):
        """负数表达式"""
        result = await calculator.run(expression="-10 + 3")
        assert result.success is True
        assert "-7" in result.data

    @pytest.mark.asyncio
    async def test_multiple_operators(self, calculator):
        """多运算符"""
        result = await calculator.run(expression="1+2+3+4+5")
        assert result.success is True
        assert "15" in result.data
