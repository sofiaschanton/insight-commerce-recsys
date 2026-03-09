-- =============================================================
-- DDL: Modelo Dimensional Instacart
-- Proyecto: Insight Commerce - Sistema de Recomendación
-- Destino: Neon PostgreSQL (free tier)
-- Ejecutar en orden: DIM_USER → DIM_PRODUCT → FACT_ORDER_PRODUCTS
-- Filtros aplicados en ETL:
--   - eval_set != 'test'
--   - usuarios con >= 5 órdenes prior Y >= 1 orden train
--   - productos con >= 50 compras globales (MIN_COMPRAS, EDA Sección 3)
--   - LIMIT 10.000 usuarios aptos
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
    order_key               INTEGER         NOT NULL,
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


-- =============================================================
-- QUERIES DE VALIDACIÓN
-- Ejecutar después de la carga del ETL para verificar integridad
-- =============================================================


-- -------------------------------------------------------------
-- V-1. Volumen por tabla
-- -------------------------------------------------------------
SELECT 'dim_user'            AS tabla, COUNT(*) AS filas FROM dim_user
UNION ALL
SELECT 'dim_product'         AS tabla, COUNT(*) AS filas FROM dim_product
UNION ALL
SELECT 'fact_order_products' AS tabla, COUNT(*) AS filas FROM fact_order_products;


-- -------------------------------------------------------------
-- V-2. Distribución de get_eval — no debe haber filas 'test'
-- -------------------------------------------------------------
SELECT get_eval, COUNT(*) AS filas
FROM fact_order_products
GROUP BY get_eval
ORDER BY get_eval;


-- -------------------------------------------------------------
-- V-3. Usuarios con >= 5 órdenes prior — universo del modelo
-- -------------------------------------------------------------
SELECT COUNT(*) AS usuarios_aptos
FROM (
    SELECT user_key
    FROM fact_order_products
    WHERE get_eval = 'prior'
    GROUP BY user_key
    HAVING COUNT(DISTINCT order_key) >= 5
) aptos;


-- -------------------------------------------------------------
-- V-4. Productos con >= 50 compras globales — MIN_COMPRAS EDA S3
-- -------------------------------------------------------------
SELECT COUNT(*) AS productos_aptos
FROM (
    SELECT product_key
    FROM fact_order_products
    GROUP BY product_key
    HAVING COUNT(*) >= 50
) aptos;


-- -------------------------------------------------------------
-- V-5. Usuarios aptos con sus productos aptos — universo final
-- -------------------------------------------------------------
SELECT COUNT(*) AS pares_usuario_producto
FROM fact_order_products f
WHERE f.user_key IN (
    SELECT user_key
    FROM fact_order_products
    WHERE get_eval = 'prior'
    GROUP BY user_key
    HAVING COUNT(DISTINCT order_key) >= 5
)
AND f.product_key IN (
    SELECT product_key
    FROM fact_order_products
    GROUP BY product_key
    HAVING COUNT(*) >= 50
);


-- -------------------------------------------------------------
-- V-6. NULLs en days_since_prior_order — deben existir (primera orden)
-- -------------------------------------------------------------
SELECT
    COUNT(*)                                                        AS total_filas,
    COUNT(*) FILTER (WHERE days_since_prior_order IS NULL)          AS nulls_esperados,
    ROUND(
        COUNT(*) FILTER (WHERE days_since_prior_order IS NULL)
        * 100.0 / COUNT(*), 2
    )                                                               AS pct_nulls
FROM fact_order_products;


-- -------------------------------------------------------------
-- V-7. Integridad referencial — no deben existir huérfanos
-- -------------------------------------------------------------
SELECT COUNT(*) AS huerfanos_user
FROM fact_order_products f
LEFT JOIN dim_user u ON f.user_key = u.user_key
WHERE u.user_key IS NULL;

SELECT COUNT(*) AS huerfanos_product
FROM fact_order_products f
LEFT JOIN dim_product p ON f.product_key = p.product_key
WHERE p.product_key IS NULL;


-- -------------------------------------------------------------
-- V-8. Resumen ejecutivo — resultado esperado post carga
-- -------------------------------------------------------------
SELECT
    (SELECT COUNT(*) FROM dim_user)                 AS usuarios_cargados,
    (SELECT COUNT(*) FROM dim_product)              AS productos_cargados,
    (SELECT COUNT(*) FROM fact_order_products)      AS hechos_cargados,
    (SELECT COUNT(DISTINCT order_key)
     FROM fact_order_products)                      AS ordenes_unicas,
    (SELECT COUNT(DISTINCT user_key)
     FROM fact_order_products
     WHERE get_eval = 'prior')                      AS usuarios_con_prior,
    (SELECT COUNT(DISTINCT user_key)
     FROM fact_order_products
     WHERE get_eval = 'train')                      AS usuarios_con_train;