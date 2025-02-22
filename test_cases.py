# test_cases.py

TEST_CASES = {
    "nl_to_sql": [
        {"query": "Get the total sales from the orders table.", "expected": "SELECT SUM(total) FROM orders;"},
        {"query": "Find all employees who joined after 2020.", "expected": "SELECT * FROM employees WHERE join_date > '2020-01-01';"},
        {"query": "Show the average price of all products.", "expected": "SELECT AVG(price) FROM products;"},
        {"query": "Get the total revenue per customer from the orders table, including their names.",
         "expected": "SELECT customers.name, SUM(orders.total) FROM customers JOIN orders ON customers.id = orders.customer_id GROUP BY customers.name;"},
        {"query": "Find all employees who have a salary above the department average salary.",
         "expected": "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees GROUP BY department_id);"}
    ],
    "sql_to_nl": [
        {"query": "SELECT COUNT(*) FROM customers;", "expected": "How many customers are there?"},
        {"query": "SELECT name FROM students WHERE grade = 'A';", "expected": "List the students who got an A grade."},
        {"query": "SELECT product_name FROM products WHERE stock < 10;", "expected": "Which products have less than 10 in stock?"},
        {"query": "SELECT department, COUNT(*) FROM employees WHERE join_date > '2022-01-01' GROUP BY department HAVING COUNT(*) > 5;",
         "expected": "List departments where more than 5 employees joined after January 1, 2022."},
        {"query": "SELECT name, (SELECT AVG(salary) FROM employees) AS avg_salary FROM employees;",
         "expected": "Show each employee's name along with the company's average salary."}
    ],
    "nl_to_sql_jp": [
        {"query": "注文テーブルから総売上を取得してください。", "expected": "SELECT SUM(total) FROM orders;"},
        {"query": "2020年以降に入社した従業員をすべて見つけてください。", "expected": "SELECT * FROM employees WHERE join_date > '2020-01-01';"},
        {"query": "すべての製品の平均価格を表示してください。", "expected": "SELECT AVG(price) FROM products;"},
        {"query": "注文テーブルから顧客ごとの総売上を取得し、名前も含めてください。",
         "expected": "SELECT customers.name, SUM(orders.total) FROM customers JOIN orders ON customers.id = orders.customer_id GROUP BY customers.name;"},
        {"query": "部門の平均給与を超える給与を持つすべての従業員を見つけてください。",
         "expected": "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees GROUP BY department_id);"}
    ],
    "sql_to_nl_jp": [
        {"query": "SELECT COUNT(*) FROM customers;", "expected": "顧客の総数はいくつですか？"},
        {"query": "SELECT name FROM students WHERE grade = 'A';", "expected": "A評価を獲得した学生を一覧表示してください。"},
        {"query": "SELECT product_name FROM products WHERE stock < 10;", "expected": "在庫が10未満の製品はどれですか？"},
        {"query": "SELECT department, COUNT(*) FROM employees WHERE join_date > '2022-01-01' GROUP BY department HAVING COUNT(*) > 5;",
         "expected": "2022年1月1日以降に5人以上が入社した部門を一覧表示してください。"},
        {"query": "SELECT name, (SELECT AVG(salary) FROM employees) AS avg_salary FROM employees;",
         "expected": "各従業員の名前と会社の平均給与を表示してください。"}
    ],
    "general_knowledge": [
        {"query": "What is the capital of France?", "expected": "Paris"},
        {"query": "Who wrote 'To Kill a Mockingbird'?", "expected": "Harper Lee"},
        {"query": "What is the boiling point of water?", "expected": "100°C"}
    ],
    "math": [
        {"query": "Integrate x^2 + 3x + 5", "expected": ""},
        {"query": "Solve for x: 2x + 3 = 11", "expected": "x = 4"},
    ],
    "coding": [
        {"query": "Write a Python function to reverse a string", "expected": "def reverse_string"},
        {"query": "What does the 'map' function do in Python?", "expected": "applies a function to all elements"},
    ]
}

def get_test_cases(category="nl_to_sql"):
    """Returns test cases based on category."""
    return TEST_CASES.get(category, TEST_CASES["nl_to_sql"])
