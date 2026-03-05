-- Creacion de los esquemas necesarios
CREATE SCHEMA IF NOT EXISTS Resources;
CREATE SCHEMA IF NOT EXISTS Departments;
CREATE SCHEMA IF NOT EXISTS Orders_Schema;
CREATE SCHEMA IF NOT EXISTS Users_Schema;

-- Tabla 1
CREATE TABLE IF NOT EXISTS Users_Schema.users(
    user_id SERIAL PRIMARY KEY,
    user_name VARCHAR(50), -- Crear con Python
    user_address VARCHAR(500), -- Crear con Python
    user_age INT -- Crear con Python
);

-- Tabla 2
CREATE TABLE IF NOT EXISTS Resources.aisles(
    aisle_id SERIAL PRIMARY KEY,
    aisle VARCHAR(200)
);

-- Tabla 3
CREATE TABLE IF NOT EXISTS Departments.departments(
    department_id SERIAL PRIMARY KEY,
    department VARCHAR(200)
);

-- Tabla 4
CREATE TABLE IF NOT EXISTS Resources.products(
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(500),
    aisle_id INT,
    department_id INT
);

-- Llave foraena de aisle_id
ALTER TABLE Resources.products
ADD CONSTRAINT fk_aisle_id
FOREIGN KEY (aisle_id) REFERENCES Resources.aisles(aisle_id);

-- Llave foranea de department_id
ALTER TABLE Resources.products
ADD CONSTRAINT fk_department_id
FOREIGN KEY (department_id) REFERENCES Departments.departments(department_id);

-- Tabla 5
CREATE TABLE IF NOT EXISTS Orders_Schema.order(
    order_id SERIAL PRIMARY KEY
);

-- Tabla 6
CREATE TABLE IF NOT EXISTS Orders_Schema.orders(
    order_id INT,
    user_id INT,
    eval_set VARCHAR(6),
    order_number INT,
    order_dow INT,
    order_hour_of_day VARCHAR(2),
    days_since_prior_order FLOAT
);

ALTER TABLE Orders_Schema.orders
ADD CONSTRAINT fk_user_id
FOREIGN KEY (user_id) REFERENCES Users_Schema.users(user_id);

-- Tabla 7
CREATE TABLE IF NOT EXISTS Orders_Schema.order_products_train(
    order_id INT,
    product_id INT,
    add_to_cart_order INT,
    reordered INT
);

ALTER TABLE Orders_Schema.order_products_train
ADD CONSTRAINT fk_order_id_train
FOREIGN KEY (order_id) REFERENCES Orders_Schema.order(order_id);

ALTER TABLE Orders_Schema.order_products_train
ADD CONSTRAINT fk_product_id_train
FOREIGN KEY (product_id) REFERENCES Resources.products(product_id);

-- Tabla 8
CREATE TABLE IF NOT EXISTS Orders_Schema.order_products_prior(
    order_id INT,
    product_id INT,
    add_to_cart_order INT,
    reordered INT
);

ALTER TABLE Orders_Schema.order_products_train
ADD CONSTRAINT fk_order_id_prior
FOREIGN KEY (order_id) REFERENCES Orders_Schema.order(order_id);

ALTER TABLE Orders_Schema.order_products_train
ADD CONSTRAINT fk_product_id_prior
FOREIGN KEY (product_id) REFERENCES Resources.products(product_id);