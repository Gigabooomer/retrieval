# test_cases.py

TEST_CASES = {
    "nl_to_sql": [
        {"query": "Get the total sales from the orders table.", "expected": "SELECT SUM(total) FROM orders;"},
        {"query": "Find all employees who joined after 2020.", "expected": "SELECT * FROM employees WHERE join_date > '2020-01-01';"},
        {"query": "Show the average price of all products.", "expected": "SELECT AVG(price) FROM products;"},
    ],
    "sql_to_nl": [
        {"query": "SELECT COUNT(*) FROM customers;", "expected": "How many customers are there?"},
        {"query": "SELECT name FROM students WHERE grade = 'A';", "expected": "List the students who got an A grade."},
        {"query": "SELECT product_name FROM products WHERE stock < 10;", "expected": "Which products have less than 10 in stock?"},
    ],
    "text_to_tables": [
        {"query": "A company has 3 departments: HR, Engineering, and Sales. Each department has a head and multiple employees.", 
         "expected": "Departments Table: ['HR', 'Engineering', 'Sales']
Heads Table: ['John', 'Alice', 'Bob']
Employees Table: [..]"},
        {"query": "John, Sarah, and Mike scored 80, 90, and 85 in math respectively.", 
         "expected": "Students Table: ['John', 'Sarah', 'Mike']
Scores Table: [80, 90, 85]"},
        {"query": "The conference had 4 sessions: AI, ML, Data Science, and Cloud Computing.", 
         "expected": "Sessions Table: ['AI', 'ML', 'Data Science', 'Cloud Computing']"},
    ],
    "nl_to_sql_jp": [
        {"query": "注文テーブルから総売上を取得してください。", "expected": "SELECT SUM(total) FROM orders;"},
        {"query": "2020年以降に入社した従業員をすべて見つけてください。", "expected": "SELECT * FROM employees WHERE join_date > '2020-01-01';"},
        {"query": "すべての製品の平均価格を表示してください。", "expected": "SELECT AVG(price) FROM products;"},
    ],
    "sql_to_nl_jp": [
        {"query": "SELECT COUNT(*) FROM customers;", "expected": "顧客の総数はいくつですか？"},
        {"query": "SELECT name FROM students WHERE grade = 'A';", "expected": "A評価を獲得した学生を一覧表示してください。"},
        {"query": "SELECT product_name FROM products WHERE stock < 10;", "expected": "在庫が10未満の製品はどれですか？"},
    ],
    "text_to_tables_jp": [
        {"query": "ある会社には3つの部門があります：人事、エンジニアリング、営業。それぞれの部門には責任者と複数の従業員がいます。",
         "expected": "部門テーブル: ['人事', 'エンジニアリング', '営業']
責任者テーブル: ['ジョン', 'アリス', 'ボブ']
従業員テーブル: [..]"},
        {"query": "ジョン、サラ、マイクはそれぞれ数学で80点、90点、85点を取得しました。",
         "expected": "学生テーブル: ['ジョン', 'サラ', 'マイク']
成績テーブル: [80, 90, 85]"},
        {"query": "この会議には4つのセッションがありました: AI、機械学習、データサイエンス、クラウドコンピューティング。",
         "expected": "セッションテーブル: ['AI', '機械学習', 'データサイエンス', 'クラウドコンピューティング']"},
    ]
}

def get_test_cases(category="nl_to_sql"):
    """Returns test cases based on category."""
    return TEST_CASES.get(category, TEST_CASES["nl_to_sql"])
