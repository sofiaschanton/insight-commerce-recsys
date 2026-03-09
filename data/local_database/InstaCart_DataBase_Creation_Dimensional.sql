-- =============================================================
-- DDL: Modelo Dimensional Instacart
-- Proyecto Supabase: nuevo proyecto separado
-- Schema: public (default)
-- Ejecutar en orden: DIM_USER → DIM_PRODUCT → FACT_ORDER_PRODUCTS
-- =============================================================


-- -------------------------------------------------------------
-- 1. DIM_USER
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_user (
    user_key        INTEGER         PRIMARY KEY,   -- user_id del CSV
    user_name       VARCHAR(150)    NULL,          -- a poblar después
    user_address    VARCHAR(300)    NULL,          -- a poblar después
    user_birthdate  DATE            NULL           -- a poblar después
);


-- -------------------------------------------------------------
-- 2. DIM_PRODUCT
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_product (
    product_key     INTEGER         PRIMARY KEY,   -- product_id del CSV
    product_name    VARCHAR(300)    NOT NULL,
    aisle_name      VARCHAR(100)    NOT NULL,      -- desnormalizado
    department_name VARCHAR(100)    NOT NULL       -- desnormalizado
);


-- -------------------------------------------------------------
-- 3. FACT_ORDER_PRODUCTS
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_order_products (
    order_key               INTEGER         NOT NULL,  -- order_id del CSV
    user_key                INTEGER         NOT NULL REFERENCES dim_user(user_key),
    product_key             INTEGER         NOT NULL REFERENCES dim_product(product_key),
    order_dow               SMALLINT        NOT NULL,  -- 0=Lunes ... 6=Domingo
    order_hour_of_day       SMALLINT        NOT NULL,  -- 0 a 23
    days_since_prior_order  FLOAT           NULL,      -- NULL en primera orden
    add_to_cart_order       SMALLINT        NOT NULL,
    reordered               SMALLINT        NOT NULL,  -- 0 = no, 1 = sí
    order_number            INTEGER         NOT NULL,
    get_eval                VARCHAR(10)     NOT NULL
                            CHECK (get_eval IN ('train', 'prior')),

    -- PK compuesta: una orden puede tener múltiples productos
    PRIMARY KEY (order_key, product_key)
);


-- -------------------------------------------------------------
-- 4. ÍNDICES (performance en queries analíticas)
-- -------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_fact_user_key
    ON fact_order_products(user_key);

CREATE INDEX IF NOT EXISTS idx_fact_product_key
    ON fact_order_products(product_key);

CREATE INDEX IF NOT EXISTS idx_fact_order_key
    ON fact_order_products(order_key);

CREATE INDEX IF NOT EXISTS idx_fact_get_eval
    ON fact_order_products(get_eval);