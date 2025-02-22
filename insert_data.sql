-- insert_data.sql

-- Insert random data into Departments
INSERT INTO departments (name) VALUES 
('HR'), ('Engineering'), ('Sales'), ('Marketing');

-- Insert random data into Customers
INSERT INTO customers (name, email) VALUES 
('Alice Johnson', 'alice@example.com'),
('Bob Smith', 'bob@example.com'),
('Charlie Brown', 'charlie@example.com');

-- Insert random data into Employees
INSERT INTO employees (name, department_id, join_date, salary) VALUES 
('John Doe', 1, '2021-05-10', 50000),
('Jane Doe', 2, '2020-03-15', 80000),
('Mike Ross', 3, '2022-08-01', 60000);

-- Insert random data into Orders
INSERT INTO orders (customer_id, total, order_date) VALUES 
(1, 250.75, '2023-06-15'),
(2, 125.50, '2023-07-20'),
(3, 320.40, '2023-09-10');

-- Insert random data into Products
INSERT INTO products (product_name, price, stock) VALUES 
('Laptop', 1200.00, 10),
('Mouse', 25.00, 50),
('Keyboard', 45.00, 30);

-- Insert random data into Students
INSERT INTO students (name, grade) VALUES 
('Emma Watson', 'A'),
('Tom Hardy', 'B'),
('Chris Evans', 'A');
