package main

import (
    "fmt"
    "math/rand"
    "time"

    "PC2/lr"
)

// generateData crea dos slices x e y con N puntos y ruido.
func generateData(N int) ([]float64, []float64) {
    x := make([]float64, N)
    y := make([]float64, N)
    rand.Seed(time.Now().UnixNano())
    for i := range x {
        x[i] = rand.Float64() * 100
        y[i] = 2*x[i] + 5 + rand.NormFloat64()
    }
    return x, y
}

// MeanSquaredError calcula el MSE para los coeficientes m, b.
func MeanSquaredError(x, y []float64, m, b float64) float64 {
    n := float64(len(x))
    var sum float64
    for i := range x {
        diff := y[i] - (m*x[i] + b)
        sum += diff * diff
    }
    return sum / n
}

// RSquared calcula el coeficiente de determinación R².
func RSquared(x, y []float64, m, b float64) float64 {
    n := float64(len(x))
    var sumErr, sumTot, sumY float64

    // media de y
    for _, yi := range y {
        sumY += yi
    }
    meanY := sumY / n

    // SSE y SST
    for i := range x {
        err := y[i] - (m*x[i] + b)
        sumErr += err * err

        dev := y[i] - meanY
        sumTot += dev * dev
    }
    if sumTot == 0 {
        return 0
    }
    return 1 - sumErr/sumTot
}

// result agrupa resultados de una corrida de entrenamiento.
type result struct {
    label    string
    m, b     float64
    duration time.Duration
    mse, r2  float64
}

// train ejecuta fn en x, y, mide tiempo y calcula métricas.
func train(label string, fn func([]float64, []float64) (float64, float64), x, y []float64, out chan<- result) {
    start := time.Now()
    m, b := fn(x, y)
    dur := time.Since(start)

    mse := MeanSquaredError(x, y, m, b)
    r2 := RSquared(x, y, m, b)

    out <- result{label, m, b, dur, mse, r2}
}

func main() {
    const N = 20_000_000

    // generamos datos
    x, y := generateData(N)

    // canal para recibir dos resultados
    ch := make(chan result, 2)

    // lanzamos ambas regresiones en paralelo
    go train("Secuencial", lr.LinearRegression, x, y, ch)
    go train("Concurrente", lr.ConcurrentLinearRegression, x, y, ch)

    // imprimimos los dos resultados al recibirlos
    for i := 0; i < 2; i++ {
        res := <-ch
        fmt.Printf("%s:\n", res.label)
        fmt.Printf("  m=%.4f, b=%.4f\n", res.m, res.b)
        fmt.Printf("  Tiempo de entrenamiento: %.3fs\n", res.duration.Seconds())
        fmt.Printf("  MSE: %.6f\n", res.mse)
        fmt.Printf("  R²:  %.6f\n\n", res.r2)
    }
}
