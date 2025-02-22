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
    ]
}

def get_test_cases(category="nl_to_sql"):
    """Returns test cases based on category."""
    return TEST_CASES.get(category, TEST_CASES["nl_to_sql"])
