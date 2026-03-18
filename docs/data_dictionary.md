# Data Dictionary – Modelo Dimensional (Star Schema)

## 1. Objetivo

Documentar las variables y reglas de negocio del dataset dimensional usado por el sistema de recomendación, para asegurar alineación entre:

- Modelo de datos en base (star schema)
- Feature engineering
- Entrenamiento e inferencia

Este documento refleja el modelo definido en `data/local_database/InstaCart_DataBase_Creation_Dimensional.sql`.

---

## 2. Estructura del modelo dimensional

Tablas productivas del modelo estrella:

- `dim_user` (dimensión)
- `dim_product` (dimensión)
- `fact_order_products` (tabla de hechos)

---

## 3. Diccionario por tabla

### 3.1 Tabla: `dim_user`

| Field            | Type         | Description                                                  | Unit  | Nulls | Example              |
| ---------------- | ------------ | ------------------------------------------------------------ | ----- | ----- | -------------------- |
| `user_key`       | integer      | Clave del usuario (surrogate/natural key del dataset fuente) | ID    | No    | 1001                 |
| `user_name`      | varchar(150) | Nombre del usuario (atributo descriptivo, opcional)          | texto | Sí    | Juan Pérez           |
| `user_address`   | varchar(300) | Dirección del usuario (atributo descriptivo, opcional)       | texto | Sí    | Av. Siempre Viva 123 |
| `user_birthdate` | date         | Fecha de nacimiento del usuario (atributo opcional)          | fecha | Sí    | 1992-08-14           |

Regla clave:

- `user_key` es `PRIMARY KEY`.

### 3.2 Tabla: `dim_product`

| Field             | Type         | Description                              | Unit      | Nulls | Example      |
| ----------------- | ------------ | ---------------------------------------- | --------- | ----- | ------------ |
| `product_key`     | integer      | Clave del producto                       | ID        | No    | 24852        |
| `product_name`    | varchar(300) | Nombre del producto                      | texto     | No    | Banana       |
| `aisle_name`      | varchar(100) | Nombre del pasillo (desnormalizado)      | categoría | No    | fresh fruits |
| `department_name` | varchar(100) | Nombre del departamento (desnormalizado) | categoría | No    | produce      |

Regla clave:

- `product_key` es `PRIMARY KEY`.

### 3.3 Tabla: `fact_order_products`

| Field                    | Type        | Description                                              | Unit      | Nulls | Example |
| ------------------------ | ----------- | -------------------------------------------------------- | --------- | ----- | ------- |
| `order_key`              | integer     | Identificador de orden                                   | ID        | No    | 123456  |
| `user_key`               | integer     | Usuario que realizó la orden (`FK -> dim_user.user_key`) | ID        | No    | 1001    |
| `product_key`            | integer     | Producto comprado (`FK -> dim_product.product_key`)      | ID        | No    | 24852   |
| `order_dow`              | smallint    | Día de la semana de la orden                             | 0–6       | No    | 2       |
| `order_hour_of_day`      | smallint    | Hora de la orden                                         | 0–23      | No    | 10      |
| `days_since_prior_order` | float       | Días desde la orden anterior                             | días      | Sí    | 7.0     |
| `add_to_cart_order`      | smallint    | Posición en la que se agregó el producto al carrito      | posición  | No    | 1       |
| `reordered`              | smallint    | Indicador de recompra (0=no, 1=sí)                       | binario   | No    | 1       |
| `order_number`           | integer     | Número de orden del usuario en secuencia histórica       | conteo    | No    | 12      |
| `get_eval`               | varchar(10) | Partición temporal/modelado (`prior` o `train`)          | categoría | No    | prior   |

Reglas clave:

- `PRIMARY KEY (order_key, product_key)`.
- `get_eval` tiene `CHECK` permitido: `prior`, `train`.
- `days_since_prior_order` puede ser `NULL` (esperado en primera orden de un usuario).

---

## 4. Relaciones del modelo estrella

- `fact_order_products.user_key` → `dim_user.user_key`
- `fact_order_products.product_key` → `dim_product.product_key`

Grano de la tabla de hechos:

- Una fila por par `(order_key, product_key)`.

---

## 5. Variables clave para Feature Engineering

Variables derivadas utilizadas por el pipeline de features/modelado (resumen funcional):

### 5.1 Features de usuario

| Field                        | Type            | Description                                              | Unit                 | Nulls | Example |
| ---------------------------- | --------------- | -------------------------------------------------------- | -------------------- | ----- | ------- |
| `user_total_orders`          | numeric/integer | Cantidad total de órdenes históricas del usuario         | conteo               | No    | 18      |
| `user_avg_basket_size`       | float           | Promedio de productos por orden del usuario              | productos/orden      | No    | 9.4     |
| `user_days_since_last_order` | float           | Recencia de compra del usuario                           | días                 | Sí    | 6.0     |
| `user_reorder_ratio`         | float           | Proporción histórica de recompra del usuario             | ratio (0–1)          | No    | 0.62    |
| `user_distinct_products`     | integer         | Cantidad de productos distintos comprados por el usuario | conteo               | No    | 57      |
| `user_segment_code`          | integer         | Segmento del usuario por frecuencia de compra            | categoría codificada | No    | 3       |

### 5.2 Features de producto

| Field                       | Type    | Description                                             | Unit              | Nulls | Example |
| --------------------------- | ------- | ------------------------------------------------------- | ----------------- | ----- | ------- |
| `product_total_purchases`   | integer | Compras globales del producto                           | conteo            | No    | 2560    |
| `product_reorder_rate`      | float   | Tasa global de recompra del producto                    | ratio (0–1)       | No    | 0.71    |
| `product_avg_add_to_cart`   | float   | Posición promedio en carrito del producto               | posición promedio | Sí    | 3.8     |
| `product_unique_users`      | integer | Cantidad de usuarios únicos que compraron el producto   | conteo            | Sí    | 980     |
| `p_department_reorder_rate` | float   | Tasa de recompra promedio del departamento del producto | ratio (0–1)       | Sí    | 0.66    |
| `p_aisle_reorder_rate`      | float   | Tasa de recompra promedio del pasillo del producto      | ratio (0–1)       | Sí    | 0.63    |

### 5.3 Features de interacción usuario-producto

| Field                           | Type           | Description                                                   | Unit              | Nulls | Example      |
| ------------------------------- | -------------- | ------------------------------------------------------------- | ----------------- | ----- | ------------ |
| `up_times_purchased`            | integer        | Veces que el usuario compró ese producto                      | conteo            | No    | 7            |
| `up_reorder_rate`               | float          | Frecuencia de recompra del par usuario-producto               | ratio (0–1)       | No    | 0.39         |
| `up_orders_since_last_purchase` | integer        | Órdenes transcurridas desde última compra del producto        | conteo            | No    | 2            |
| `up_first_order_number`         | integer        | Número de orden en que apareció por primera vez el producto   | secuencia         | No    | 2            |
| `up_last_order_number`          | integer        | Número de orden en que apareció por última vez el producto    | secuencia         | No    | 16           |
| `up_avg_add_to_cart_order`      | float          | Posición promedio del producto en carrito para ese usuario    | posición promedio | Sí    | 2.1          |
| `up_days_since_last`            | float          | Días desde última compra del par usuario-producto             | días              | Sí    | 9.0          |
| `up_avg_days_between_orders`    | float          | Promedio de días entre compras del mismo producto por usuario | días              | Sí    | 12.5         |
| `up_delta_days`                 | float          | Diferencia entre recencia y ciclo promedio del par            | días              | Sí    | -3.5         |
| `u_favorite_department`         | string/integer | Departamento favorito histórico del usuario                   | categoría         | Sí    | produce      |
| `u_favorite_aisle`              | string/integer | Pasillo favorito histórico del usuario                        | categoría         | Sí    | fresh fruits |

### 5.4 Variable objetivo

| Field   | Type             | Description                                                              | Unit          | Nulls | Example |
| ------- | ---------------- | ------------------------------------------------------------------------ | ------------- | ----- | ------- |
| `label` | smallint/integer | Objetivo del modelo: recompra esperada del par usuario-producto en train | binario (0/1) | No    | 1       |

---

## 6. Reglas de dataset y calidad de datos

Reglas estructurales:

1. Integridad referencial obligatoria de `user_key` y `product_key` desde hechos hacia dimensiones.
2. Unicidad de `dim_user.user_key` y `dim_product.product_key`.
3. Unicidad de par `(order_key, product_key)` en hechos.
4. `get_eval` solo admite valores `prior` y `train`.

Reglas funcionales del proyecto (ETL/modelado):

1. Se excluyen filas `test` del dataset fuente.
2. Universo de usuarios aptos: al menos 5 órdenes `prior` y al menos 1 orden `train`.
3. Universo de productos aptos: al menos 50 compras globales.
4. `days_since_prior_order` nulo es esperado en primeras órdenes.

---

## 7. Definición de terminado (DoD)

- Documento versionado en repositorio en `docs/data_dictionary.md`.
- Tablas del modelo dimensional estrella documentadas (`dim_user`, `dim_product`, `fact_order_products`).
- Variables clave de feature engineering y reglas de dataset documentadas para uso de Data Science e Ingeniería.
