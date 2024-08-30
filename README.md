# RWA Financial Model

## Construcción de Portafolio Replicante Autofinanciado para Valuación de Obligaciones de Deuda Tokenizadas

### Índice
1. [Introducción](#introducción)
2. [Fundamentos Teóricos](#fundamentos-teóricos)
3. [Modelado de Procesos Estocásticos](#modelado-de-procesos-estocásticos)
4. [Implementación del Modelo](#implementación-del-modelo)
5. [Optimización y Análisis del Portafolio](#optimización-y-análisis-del-portafolio)
6. [Aplicación On-Chain](#aplicación-on-chain)
7. [Limitaciones y Consideraciones Futuras](#limitaciones-y-consideraciones-futuras)

### Introducción

Este proyecto implementa un modelo financiero para la construcción de portafolios replicantes autofinanciados, enfocado en la valuación de obligaciones de deuda tokenizadas en el contexto de las finanzas descentralizadas (DeFi).

### Fundamentos Teóricos

#### Modelo de Black-Scholes-Merton

El modelo se basa en la teoría de Black-Scholes-Merton, que utiliza procesos de Wiener y ecuaciones diferenciales estocásticas (SDEs) para modelar los retornos de activos financieros.

- **Procesos de Wiener en Finanzas**: El movimiento Browniano o proceso de Wiener es fundamental para modelar la dinámica de precios de activos en tiempo continuo.
- **Ecuación de Black-Scholes**: Deriva una ecuación diferencial parcial cuya solución proporciona el precio de opciones financieras.

### Modelado de Procesos Estocásticos

#### Movimiento Browniano Geométrico (GBM)

El valor de cada préstamo se modela utilizando un GBM, descrito por la siguiente ecuación diferencial estocástica:

$dS_t = \mu S_t dt + \sigma S_t dW_t$

Donde:
- $S_t$: Valor del préstamo en el tiempo t
- $\mu$: Deriva (tasa de rendimiento esperada)
- $\sigma$: Volatilidad
- $W_t$: Proceso de Wiener estándar

#### Implementación Numérica: Esquema de Euler-Maruyama

Para la simulación numérica, utilizamos el esquema de Euler-Maruyama:

$S_{t+\Delta t} = S_t \exp\left((\mu - \frac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z_t\right)$

Donde:
- $\Delta t$: Paso de tiempo
- $Z_t \sim N(0,1)$: Variable aleatoria normal estándar

### Implementación del Modelo

#### Datos de Entrada
Los datos se obtienen del archivo `cleaned_loans.csv`, que incluye:
- `loan_id`: Identificador único del préstamo
- `base_interest_rate`: Tasa de interés base (%)
- `loan_duration_days`: Duración del préstamo en días
- `compounding_periods`: Número de períodos de capitalización
- `normalized_cumulative_return`: Retorno acumulativo normalizado

#### Estimación de Parámetros
- Deriva ($\mu$): $\mu = \text{base\_interest\_rate} / 100$
- Volatilidad ($\sigma$): Estimada mediante máxima verosimilitud sobre retornos logarítmicos

#### Simulación de Monte Carlo
1. Generar trayectorias para cada préstamo usando el GBM discretizado.
2. Calcular el valor del portafolio para cada trayectoria y tiempo.

### Optimización y Análisis del Portafolio

#### Construcción del Portafolio Replicante
$\Pi = \sum_{i=1}^n w_i S_i$

#### Optimización de Markowitz
$\max_{w} \quad w^T \mu - \frac{\lambda}{2} w^T \Sigma w$

Sujeto a: $\sum_{i=1}^n w_i = 1$, $w_i \geq 0$

#### Análisis de Sensibilidad
Cálculo de "griegas":
- Delta: $\Delta = \frac{\partial \Pi}{\partial S}$
- Gamma: $\Gamma = \frac{\partial^2 \Pi}{\partial S^2}$
- Theta: $\Theta = \frac{\partial \Pi}{\partial t}$

### Aplicación On-Chain

#### Componentes Necesarios
1. Smart Contract
2. Oráculo
3. Mecanismo de Gobernanza
4. Sistema de Monitoreo
5. Integración con Protocolos DeFi

#### Flujo de Ejecución
1. Actualización de datos de mercado
2. Recálculo de pesos óptimos y métricas de riesgo
3. Trigger de rebalanceo si se superan umbrales
4. Ejecución de transacciones de compra/venta

### Limitaciones y Consideraciones Futuras

- Asunciones de normalidad y estacionariedad
- Potencial extensión para incluir saltos y colas pesadas
- Consideraciones de liquidez y costos de transacción en DeFi
- Balance entre precisión y simplicidad en la implementación numérica
